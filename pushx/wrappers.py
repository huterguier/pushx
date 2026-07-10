"""Action-space and observation wrappers for PushEnv, composable with gxm's own wrappers.

For discrete-direction actions, no wrapper is needed here — compose gxm's own
`gxm.wrappers.Discretize` directly on top of any of these, e.g.:

    Discretize(RelativeAction(PushEnv(assembly)), actions=jnp.array([...]))
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from gxm.core import Timestep
from gxm.spaces import Box
from gxm.typing import Key
from gxm.wrappers import EnvironmentWrapper

from pushx.env import PushEnv
from pushx.model import SCENE_SIZE, PushState


class NormalizeAction(EnvironmentWrapper[PushState]):
    """Actions are given in [-1, 1]^2 and rescaled to the world-coordinate bounds PushModel expects."""

    wrapped: PushEnv

    def __init__(self, wrapped: PushEnv, unwrap: bool = True):
        super().__init__(wrapped, unwrap=unwrap)
        self.action_space = Box(low=-jnp.ones(2), high=jnp.ones(2), shape=(2,))

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        return self.wrapped.init(key)

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        return self.wrapped.reset(key, state)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        world_action = (action + 1.0) * 0.5 * SCENE_SIZE
        state, timestep = self.wrapped.step(key, state, world_action)
        return state, dataclasses.replace(timestep, action=action)


class RelativeAction(EnvironmentWrapper[PushState]):
    """Actions are a delta from the pusher's current position, rather than an absolute world target."""

    wrapped: PushEnv

    def __init__(self, wrapped: PushEnv, unwrap: bool = True, scale: float = 0.3):
        super().__init__(wrapped, unwrap=unwrap)
        self.scale = scale
        self.action_space = Box(low=-jnp.full(2, scale), high=jnp.full(2, scale), shape=(2,))

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        return self.wrapped.init(key)

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        return self.wrapped.reset(key, state)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        world_action = self.wrapped.pusher_position(state) + action
        state, timestep = self.wrapped.step(key, state, world_action)
        return state, dataclasses.replace(timestep, action=action)


class VelocityAction(EnvironmentWrapper[PushState]):
    """Actions are interpreted as a direct pusher velocity command.

    PushModel.step converts its (absolute-position) action into velocity as
    `(target_pos - pusher_pos) * 5`, clamped. Inverting that conversion here lets the
    wrapper-facing action act as a direct velocity command with no PushModel changes needed.
    """

    wrapped: PushEnv

    def __init__(self, wrapped: PushEnv, unwrap: bool = True, max_speed: float = 3.0):
        super().__init__(wrapped, unwrap=unwrap)
        self.max_speed = max_speed
        self.action_space = Box(low=-jnp.full(2, max_speed), high=jnp.full(2, max_speed), shape=(2,))

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        return self.wrapped.init(key)

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        return self.wrapped.reset(key, state)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        target_pos = self.wrapped.pusher_position(state) + action / 5.0
        state, timestep = self.wrapped.step(key, state, target_pos)
        return state, dataclasses.replace(timestep, action=action)


class PixelObservation(EnvironmentWrapper[PushState]):
    """Replaces the raw-state observation with a rendered (height, width, 3) pixel image."""

    wrapped: PushEnv

    def __init__(self, wrapped: PushEnv, unwrap: bool = True, width: int = 64, height: int = 64):
        super().__init__(wrapped, unwrap=unwrap)
        self.width = width
        self.height = height
        self.observation_space = Box(low=0.0, high=1.0, shape=(height, width, 3))

    def _render_timestep(self, state: PushState, timestep: Timestep) -> Timestep:
        pixels = self.wrapped.render(state, self.width, self.height)
        return dataclasses.replace(timestep, next_obs=pixels, true_next_obs=pixels)

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        state, timestep = self.wrapped.init(key)
        return state, self._render_timestep(state, timestep)

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        state, timestep = self.wrapped.reset(key, state)
        return state, self._render_timestep(state, timestep)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        state, timestep = self.wrapped.step(key, state, action)
        return state, self._render_timestep(state, timestep)
