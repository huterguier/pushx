"""Extractors decide what each transition contributes to a collected batch.

An extractor is ``(env_state, timestep, action) -> dict`` evaluated inside the
collector's scan, so everything it returns gets a leading ``(num_steps,
num_envs)`` axis in the final batch. Build one via a factory (closing over the
env and any render settings) and pass it to ``Collector``.
"""

from typing import Protocol

import jax
import jax.numpy as jnp
from gxm.wrappers import EnvironmentWrapper
from gxm.wrappers.wrapper import WrapperState

from .collector import Extractor


class Renderable(Protocol):
    """Anything with a PushEnv-style ``render`` — rendering is not part of the gxm protocol."""

    def render(self, state, width: int, height: int) -> jax.Array: ...


def unwrap(state):
    """Peel wrapper states down to the innermost environment state."""
    while isinstance(state, WrapperState):
        state = state.wrapped_state
    return state


def frames(env: Renderable | EnvironmentWrapper, width: int = 64, height: int = 64) -> Extractor:
    """Rendered frames + actions + episode-boundary masks (e.g. for world-model training).

    ``done`` marks episode boundaries (mask these transitions or reset the
    model's latent). ``reward`` is the env's per-step reward — for PushEnv
    that is the success indicator, so it doubles as a success mask.
    """

    def render(env_state):
        pixels = env.render(unwrap(env_state), width=width, height=height)
        return (pixels * 255).astype(jnp.uint8)

    def extractor(env_state, timestep, action):
        return {
            "frames": jax.vmap(render)(env_state),
            "action": action,
            "done": timestep.terminated | timestep.truncated,
            "reward": timestep.reward,
        }

    return extractor
