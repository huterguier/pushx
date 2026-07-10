from __future__ import annotations

import jax.numpy as jnp
from gxm import Environment, Timestep
from gxm.typing import Key

from pushx import rewards, termination
from pushx.model import PushModel, PushState
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
        self._model = PushModel(assembly)
        self.observation_space = self._model.observation_space
        self.action_space = self._model.action_space
        self._reward_fn = reward_fn or rewards.sparse_success
        self._success_fn = success_fn or termination.pose_threshold()

    def error(self, state: PushState) -> tuple[jnp.ndarray, jnp.ndarray]:
        root_idx = self._model._stroke_indices[0]
        pos = state.sim_state.polygon.position[root_idx]
        world_angle = state.sim_state.polygon.rotation[root_idx]
        # world_angle = assembly_angle + stroke_0.angle; goal_angle is an assembly angle
        assembly_angle = world_angle - self._model._assembly.strokes[0].angle
        pos_err = jnp.linalg.norm(pos - state.goal_pos)
        period = 2 * jnp.pi / self._model._assembly.symmetry_order
        raw = (assembly_angle - state.goal_angle) % period
        angle_err = jnp.minimum(raw, period - raw)
        return pos_err, angle_err

    def success(self, state: PushState) -> jnp.ndarray:
        return self._success_fn(self, state)

    def pusher_position(self, state: PushState) -> jnp.ndarray:
        return state.sim_state.circle.position[self._model._pusher_idx]

    def init(self, key: Key) -> tuple[PushState, Timestep]:
        state, step = self._model.init(key)
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
        new_state, model_step = self._model.step(key, state, action)
        success = self.success(new_state)
        reward = self._reward_fn(self, state, new_state)
        return new_state, Timestep(
            next_obs=model_step.next_obs,
            true_next_obs=model_step.next_obs,
            action=action,
            reward=reward,
            terminated=success,
            truncated=jnp.bool_(False),
            info={},
        )

    def render(self, state: PushState, width: int = 64, height: int = 64) -> jnp.ndarray:
        return self._model.render(state, width, height)
