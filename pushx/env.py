from __future__ import annotations

import jax.numpy as jnp
from gxm import Environment, Timestep
from gxm.typing import Key

from pushx import rewards, termination
from pushx.dynamics import WALL_HI, WALL_LO, PushDynamics, PushState
from pushx.rewards import RewardFn
from pushx.shapes import Assembly
from pushx.termination import SuccessFn


class PushEnv(Environment[PushState]):
    def __init__(
        self,
        assembly: Assembly,
        reward_fn: RewardFn | None = None,
        success_fn: SuccessFn | None = None,
        id: str = "PushX",
    ):
        self.id = id
        self._dynamics = PushDynamics(assembly)
        self.observation_space = self._dynamics.observation_space
        self.action_space = self._dynamics.action_space
        self._reward_fn = reward_fn or rewards.sparse_success
        self._success_fn = success_fn or termination.pose_threshold()

    def error(self, state: PushState) -> tuple[jnp.ndarray, jnp.ndarray]:
        root_idx = self._dynamics._stroke_indices[0]
        pos = state.sim_state.polygon.position[root_idx]
        world_angle = state.sim_state.polygon.rotation[root_idx]
        # world_angle = assembly_angle + stroke_0.angle; goal_angle is an assembly angle
        assembly_angle = world_angle - self._dynamics._assembly.strokes[0].angle
        pos_err = jnp.linalg.norm(pos - state.goal_pos)
        period = 2 * jnp.pi / self._dynamics._assembly.symmetry_order
        raw = (assembly_angle - state.goal_angle) % period
        angle_err = jnp.minimum(raw, period - raw)
        return pos_err, angle_err

    def success(self, state: PushState) -> jnp.ndarray:
        return self._success_fn(self, state)

    def out_of_bounds(self, state: PushState) -> jnp.ndarray:
        """Whether any stroke's center has left the arena, or gone NaN. The fixated pusher
        out-muscles the walls, so it can crush the assembly through one; past that point
        the welds tear and jax2d eventually emits NaNs, so ``step`` truncates the episode here."""
        pos = state.sim_state.polygon.position[jnp.array(self._dynamics._stroke_indices)]
        return jnp.any((pos < WALL_LO) | (pos > WALL_HI) | jnp.isnan(pos))

    def pusher_position(self, state: PushState) -> jnp.ndarray:
        return state.sim_state.circle.position[self._dynamics._pusher_idx]

    def assembly_position(self, state: PushState) -> jnp.ndarray:
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
        return self.init(key)

    def step(self, key: Key, state: PushState, action: jnp.ndarray) -> tuple[PushState, Timestep]:
        new_state, dynamics_step = self._dynamics.step(key, state, action)
        success = self.success(new_state)
        reward = self._reward_fn(self, state, new_state)
        return new_state, Timestep(
            next_obs=dynamics_step.next_obs,
            true_next_obs=dynamics_step.next_obs,
            action=action,
            reward=reward,
            terminated=success,
            truncated=self.out_of_bounds(new_state),
            info={},
        )

    def render(self, state: PushState, width: int = 64, height: int = 64) -> jnp.ndarray:
        return self._dynamics.render(state, width, height)
