"""Pluggable reward functions for PushEnv, all sharing the RewardFn signature."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp

from pushx.dynamics import PushState

if TYPE_CHECKING:
    from pushx.env import PushEnv

RewardFn = Callable[["PushEnv", PushState, PushState], jnp.ndarray]


def sparse_success(env: PushEnv, state: PushState, new_state: PushState) -> jnp.ndarray:
    return env.success(new_state).astype(jnp.float32)


def dense_distance(pos_weight: float = 1.0, angle_weight: float = 1.0) -> RewardFn:
    def reward_fn(env: PushEnv, state: PushState, new_state: PushState) -> jnp.ndarray:
        pos_err, angle_err = env.error(new_state)
        return -(pos_weight * pos_err + angle_weight * angle_err)

    return reward_fn


def potential_shaping(pos_weight: float = 1.0, angle_weight: float = 1.0) -> RewardFn:
    def reward_fn(env: PushEnv, state: PushState, new_state: PushState) -> jnp.ndarray:
        prev_pos_err, prev_angle_err = env.error(state)
        curr_pos_err, curr_angle_err = env.error(new_state)
        improvement = (prev_pos_err - curr_pos_err) * pos_weight + (prev_angle_err - curr_angle_err) * angle_weight
        return improvement + env.success(new_state).astype(jnp.float32)

    return reward_fn


def pusher_approach(weight: float = 0.5) -> RewardFn:
    """Potential-based shaping on the pusher-to-assembly distance.

    Without this term the agent gets no signal until it happens to move the
    assembly, which makes exploration from scratch (e.g. PPO) extremely slow.
    """

    def reward_fn(env: PushEnv, state: PushState, new_state: PushState) -> jnp.ndarray:
        prev = jnp.linalg.norm(env.pusher_position(state) - env.assembly_position(state))
        curr = jnp.linalg.norm(env.pusher_position(new_state) - env.assembly_position(new_state))
        return weight * (prev - curr)

    return reward_fn


def step_penalty(penalty: float = 0.01) -> RewardFn:
    def reward_fn(env: PushEnv, state: PushState, new_state: PushState) -> jnp.ndarray:
        return -jnp.float32(penalty)

    return reward_fn


def combine(*reward_fns: RewardFn, weights: list[float] | None = None) -> RewardFn:
    weights_ = weights if weights is not None else [1.0] * len(reward_fns)

    def reward_fn(env: PushEnv, state: PushState, new_state: PushState) -> jnp.ndarray:
        return sum(w * f(env, state, new_state) for w, f in zip(weights_, reward_fns))

    return reward_fn
