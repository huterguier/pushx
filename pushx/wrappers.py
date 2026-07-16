"""Action-space and observation wrappers for PushEnv, composable with gxm's own wrappers.

For discrete-direction actions, no wrapper is needed here — compose gxm's own
`gxm.wrappers.Discretize` directly on top of any of these, e.g.:

    Discretize(RelativeAction(PushEnv(assembly)), actions=jnp.array([...]))
"""
from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
from gxm.core import Timestep
from gxm.spaces import Box
from gxm.typing import Key
from gxm.wrappers import EnvironmentWrapper
from gxm.wrappers.wrapper import WrapperState

from pushx.env import PushEnv
from pushx.model import SCENE_SIZE, PushState


class NormalizeAction(EnvironmentWrapper[Any]):
    """Actions are given in [-1, 1]^2 and rescaled to the world-coordinate bounds PushModel expects."""

    def __init__(self, wrapped: PushEnv | EnvironmentWrapper):
        super().__init__(wrapped)
        self.action_space = Box(low=-jnp.ones(2), high=jnp.ones(2), shape=(2,))

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        return self.wrapped.init(key)

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        return self.wrapped.reset(key, state)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        world_action = (action + 1.0) * 0.5 * SCENE_SIZE
        state, timestep = self.wrapped.step(key, state, world_action)
        return state, dataclasses.replace(timestep, action=action)


class RelativeAction(EnvironmentWrapper[Any]):
    """Actions are a delta from the pusher's current position, rather than an absolute world target."""

    def __init__(self, wrapped: PushEnv | EnvironmentWrapper, scale: float = 0.3):
        super().__init__(wrapped)
        self.scale = scale
        self.action_space = Box(low=-jnp.full(2, scale), high=jnp.full(2, scale), shape=(2,))

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        return self.wrapped.init(key)

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        return self.wrapped.reset(key, state)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        world_action = cast(PushEnv, self.wrapped).pusher_position(state) + action
        state, timestep = self.wrapped.step(key, state, world_action)
        return state, dataclasses.replace(timestep, action=action)


class VelocityAction(EnvironmentWrapper[Any]):
    """Actions are interpreted as a direct pusher velocity command.

    PushModel.step converts its (absolute-position) action into velocity as
    `(target_pos - pusher_pos) * 5`, clamped. Inverting that conversion here lets the
    wrapper-facing action act as a direct velocity command with no PushModel changes needed.
    """

    def __init__(self, wrapped: PushEnv | EnvironmentWrapper, max_speed: float = 3.0):
        super().__init__(wrapped)
        self.max_speed = max_speed
        self.action_space = Box(low=-jnp.full(2, max_speed), high=jnp.full(2, max_speed), shape=(2,))

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        return self.wrapped.init(key)

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        return self.wrapped.reset(key, state)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        target_pos = cast(PushEnv, self.wrapped).pusher_position(state) + action / 5.0
        state, timestep = self.wrapped.step(key, state, target_pos)
        return state, dataclasses.replace(timestep, action=action)


class PixelObservation(EnvironmentWrapper[Any]):
    """Replaces the raw-state observation with a rendered (height, width, 3) pixel image."""

    def __init__(self, wrapped: PushEnv | EnvironmentWrapper, width: int = 64, height: int = 64):
        super().__init__(wrapped)
        self.width = width
        self.height = height
        self.observation_space = Box(low=0.0, high=1.0, shape=(height, width, 3))

    def _render_timestep(self, state: PushState, timestep: Timestep) -> Timestep:
        pixels = cast(PushEnv, self.wrapped).render(state, self.width, self.height)
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LingerResetState(WrapperState):
    steps: jnp.ndarray
    limit: jnp.ndarray
    linger: jnp.ndarray
    started: jnp.ndarray


class LingerReset(EnvironmentWrapper[LingerResetState]):
    """Reset a fixed number of steps *after* success instead of terminating on it.

    Designed for collecting data (e.g. world-model training) where terminating
    on success would cut every episode at the goal and leave no coverage of
    near-goal dynamics. Success never terminates: it is reported in
    ``info["success"]`` and ``terminated`` is always False. Instead the episode
    truncates — and, like ``gxm.wrappers.TimeLimit``, resets itself — ``linger``
    steps after success is first reached, or at ``time_limit`` steps, whichever
    comes first. The linger countdown starts at the first success and does not
    restart if the assembly leaves and re-enters the goal region.

    ``time_limit`` and ``linger`` may each be an int (fixed), a ``(low, high)``
    tuple sampled uniformly from ``[low, high)`` per episode, or ``None`` to
    disable that reset condition entirely (with ``time_limit=None``, an episode
    that never succeeds never resets). Randomizing both desynchronizes resets
    across parallel environments and avoids placing the first success at a
    fixed offset from every episode boundary.
    """

    def __init__(
        self,
        wrapped: PushEnv | EnvironmentWrapper,
        time_limit: int | tuple[int, int] | None = (200, 400),
        linger: int | tuple[int, int] | None = (20, 60),
    ):
        super().__init__(wrapped)
        self.time_limit = time_limit
        self.linger = linger

    @staticmethod
    def _sample(key: Key, value: int | tuple[int, int] | None) -> jnp.ndarray:
        if value is None:
            return jnp.array(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        if isinstance(value, tuple):
            return jax.random.randint(key, (), value[0], value[1])
        return jnp.array(value, dtype=jnp.int32)

    def _new_state(self, key: Key, wrapped_state) -> LingerResetState:
        key_limit, key_linger = jax.random.split(key)
        return LingerResetState(
            wrapped_state=wrapped_state,
            steps=jnp.array(0, dtype=jnp.int32),
            limit=self._sample(key_limit, self.time_limit),
            linger=self._sample(key_linger, self.linger),
            started=jnp.array(False, dtype=jnp.bool),
        )

    def _mark_success(self, timestep: Timestep) -> Timestep:
        return dataclasses.replace(
            timestep,
            terminated=jnp.zeros_like(timestep.terminated),
            info={**timestep.info, "success": timestep.terminated},
        )

    def init(self, key: Key) -> tuple[LingerResetState, Timestep]:
        key_init, key_sample = jax.random.split(key)
        wrapped_state, timestep = self.wrapped.init(key_init)
        return self._new_state(key_sample, wrapped_state), self._mark_success(timestep)

    def reset(self, key: Key, state: LingerResetState) -> tuple[LingerResetState, Timestep]:
        key_reset, key_sample = jax.random.split(key)
        wrapped_state, timestep = self.wrapped.reset(key_reset, state.wrapped_state)
        return self._new_state(key_sample, wrapped_state), self._mark_success(timestep)

    def step(
        self, key: Key, state: LingerResetState, action: jnp.ndarray
    ) -> tuple[LingerResetState, Timestep]:
        key_step, key_reset, key_sample = jax.random.split(key, 3)
        wrapped_state, timestep = self.wrapped.step(key_step, state.wrapped_state, action)

        success = timestep.terminated
        started = state.started | success
        linger = jnp.where(started, state.linger - 1, state.linger)
        steps = state.steps + 1
        expired = (steps >= state.limit) | (started & (linger <= 0))

        step_state = LingerResetState(
            wrapped_state=wrapped_state,
            steps=steps,
            limit=state.limit,
            linger=linger,
            started=started,
        )
        step_timestep = self._mark_success(timestep)

        reset_wrapped_state, reset_timestep = self.wrapped.reset(key_reset, wrapped_state)
        reset_state = self._new_state(key_sample, reset_wrapped_state)
        reset_timestep = Timestep(
            next_obs=reset_timestep.next_obs,
            true_next_obs=timestep.next_obs,
            action=action,
            reward=timestep.reward,
            terminated=jnp.zeros_like(timestep.terminated),
            truncated=jnp.ones_like(timestep.truncated),
            info={**timestep.info, "success": success},
        )
        return jax.lax.cond(
            expired,
            lambda: (reset_state, reset_timestep),
            lambda: (step_state, step_timestep),
        )
