"""Pluggable success/termination criteria for PushEnv, sharing SuccessFn."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pushx.dynamics import PushState
from pushx.typing import Array

if TYPE_CHECKING:
    from pushx.env import PushEnv

type SuccessFn = Callable[[PushEnv, PushState], Array]


def pose_threshold(
    pos_threshold: float = 0.08, angle_threshold: float = 0.35
) -> SuccessFn:
    """Success when the assembly is within `pos_threshold` and `angle_threshold` of
    its goal.

    Defaults match PushT's `eval_state`, rounded: 20px on a 512px window is 0.078
    world units here, and pi/9 is 0.349 rad. One caveat on the position term --
    PushT thresholds a *joint* norm over the agent's position and the block's, so
    it also asks the pusher to be parked; `PushEnv.error` scores the assembly
    alone, which is the more natural question and the one eval_pushx.py reports.
    """

    def success_fn(env: PushEnv, state: PushState) -> Array:
        pos_err, angle_err = env.error(state)
        return (pos_err < pos_threshold) & (angle_err < angle_threshold)

    return success_fn
