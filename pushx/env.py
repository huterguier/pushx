from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from gxm import Environment, Timestep
from gxm.spaces import Box

from pushx import rewards, termination
from pushx.dynamics import WALL_HI, WALL_LO, PushDynamics, PushState
from pushx.rewards import RewardFn
from pushx.shapes import Assembly
from pushx.termination import SuccessFn
from pushx.typing import Array, Key


@dataclass(frozen=True)
class PushEnv(Environment[PushState]):
    """``goal`` fixes the target pose as ``((x, y), angle)`` and ``spawn`` the
    initial ``(block_pos, block_angle, pusher_pos)``, up to ``spawn_jitter``,
    instead of sampling them per episode; see :class:`PushDynamics`."""

    assembly: Assembly
    reward_fn: RewardFn = rewards.sparse_success
    success_fn: SuccessFn = field(default_factory=termination.pose_threshold)
    goal: tuple[tuple[float, float], float] | None = None
    spawn: tuple[tuple[float, float], float, tuple[float, float]] | None = None
    spawn_jitter: tuple[float, float] = (0.1, 0.3)
    id: str = "PushX"

    observation_space: Box = field(init=False, repr=False, compare=False)
    action_space: Box = field(init=False, repr=False, compare=False)
    _dynamics: PushDynamics = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        dynamics = PushDynamics(
            self.assembly,
            goal=self.goal,
            spawn=self.spawn,
            spawn_jitter=self.spawn_jitter,
        )
        object.__setattr__(self, "_dynamics", dynamics)
        object.__setattr__(self, "observation_space", dynamics.observation_space)
        object.__setattr__(self, "action_space", dynamics.action_space)

    def error(self, state: PushState) -> tuple[Array, Array]:
        root_idx = self._dynamics._stroke_indices[0]
        pos = state.sim_state.polygon.position[root_idx]
        world_angle = state.sim_state.polygon.rotation[root_idx]
        # world_angle = assembly_angle + stroke_0.angle; goal_angle is an assembly angle
        assembly_angle = world_angle - self.assembly.strokes[0].angle
        pos_err = jnp.linalg.norm(pos - state.goal_pos)
        period = 2 * jnp.pi / self.assembly.symmetry_order
        raw = (assembly_angle - state.goal_angle) % period
        angle_err = jnp.minimum(raw, period - raw)
        return pos_err, angle_err

    def success(self, state: PushState) -> Array:
        return self.success_fn(self, state)

    def out_of_bounds(self, state: PushState) -> Array:
        """Whether any stroke's center has left the arena, or gone NaN. The fixated
        pusher out-muscles the walls, so it can crush the assembly through one; past
        that point the welds tear and jax2d eventually emits NaNs, so ``step``
        truncates the episode here."""
        stroke_indices = jnp.array(self._dynamics._stroke_indices)
        pos = state.sim_state.polygon.position[stroke_indices]
        return jnp.any((pos < WALL_LO) | (pos > WALL_HI) | jnp.isnan(pos))

    def pusher_position(self, state: PushState) -> Array:
        return state.sim_state.circle.position[self._dynamics._pusher_idx]

    def assembly_position(self, state: PushState) -> Array:
        return state.sim_state.polygon.position[self._dynamics._stroke_indices[0]]

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        state, step = self._dynamics.init(key)
        return state, Timestep(
            next_obs=step.next_obs,
            true_next_obs=step.next_obs,
            action=step.action,
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info={},
        )

    def reset(self, key: Key, state: PushState) -> tuple[PushState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: PushState, action: Array
    ) -> tuple[PushState, Timestep]:
        new_state, dynamics_step = self._dynamics.step(key, state, action)
        success = self.success(new_state)
        reward = self.reward_fn(self, state, new_state)
        return new_state, Timestep(
            next_obs=dynamics_step.next_obs,
            true_next_obs=dynamics_step.next_obs,
            action=action,
            reward=reward,
            terminated=success,
            truncated=self.out_of_bounds(new_state),
            info={},
        )

    def render(self, state: PushState, width: int = 64, height: int = 64) -> Array:
        return self._dynamics.render(state, width, height)
