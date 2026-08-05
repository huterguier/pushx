"""Pluggable success/termination criteria for PushEnv, sharing the SuccessFn signature."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp

from pushx.dynamics import PushState

if TYPE_CHECKING:
    from pushx.env import PushEnv

SuccessFn = Callable[["PushEnv", PushState], jnp.ndarray]


def pose_threshold(pos_threshold: float = 0.1, angle_threshold: float = 0.1) -> SuccessFn:
    def success_fn(env: PushEnv, state: PushState) -> jnp.ndarray:
        pos_err, angle_err = env.error(state)
        return (pos_err < pos_threshold) & (angle_err < angle_threshold)

    return success_fn
