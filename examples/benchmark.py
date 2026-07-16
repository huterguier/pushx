"""Measure the throughput of generating image observations with a trained policy.

Rolls out many parallel environments with the policy from ``ppo_T.msgpack``
(train first with ``python examples/ppo.py``), rendering a uint8 frame per env
per step — the setup for collecting world-model training data. Prints
environment steps per second with and without rendering.

    python examples/benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from flax import serialization
from gxm.wrappers import AutoReset, TimeLimit
from gxm.wrappers.wrapper import WrapperState
from ppo import ACTION_DELTA, SHAPE, TIME_LIMIT, ActorCritic

from pushx import ASSEMBLIES, PushEnv
from pushx.wrappers import RelativeAction

NUM_ENVS = 1024
NUM_STEPS = 64  # steps per rollout chunk; frames are (NUM_STEPS, NUM_ENVS, H, W, 3)
NUM_CHUNKS = 5
RENDER_SIZE = 64

env = AutoReset(
    TimeLimit(
        RelativeAction(PushEnv(ASSEMBLIES[SHAPE]), scale=ACTION_DELTA),
        time_limit=TIME_LIMIT,
    )
)
network = ActorCritic(env.action_space.shape[0])

key = jax.random.key(0)
state, timestep = jax.vmap(env.init)(jax.random.split(key, NUM_ENVS))
params = network.init(key, timestep.next_obs)
params_path = Path(__file__).parent / f"ppo_{SHAPE}.msgpack"
params = serialization.from_bytes(params, params_path.read_bytes())


def render(state):
    while isinstance(state, WrapperState):  # peel wrapper states down to PushState
        state = state.wrapped_state
    pixels = env.render(state, width=RENDER_SIZE, height=RENDER_SIZE)
    return (pixels * 255).astype(jnp.uint8)


def rollout(key, state, timestep, with_frames: bool):
    def body(carry, key):
        state, timestep = carry
        key_action, key_step = jax.random.split(key)
        dist, _ = network.apply(params, timestep.next_obs)
        action = dist.sample(seed=key_action)
        state, timestep = jax.vmap(env.step)(
            jax.random.split(key_step, NUM_ENVS), state, action
        )
        frames = jax.vmap(render)(state) if with_frames else None
        return (state, timestep), frames

    keys = jax.random.split(key, NUM_STEPS)
    (state, timestep), frames = jax.lax.scan(body, (state, timestep), keys)
    return state, timestep, frames


def measure(with_frames: bool):
    global key, state, timestep
    fn = jax.jit(lambda k, s, t: rollout(k, s, t, with_frames))
    key, key_warmup = jax.random.split(key)
    jax.block_until_ready(fn(key_warmup, state, timestep))  # compile

    start = time.perf_counter()
    for _ in range(NUM_CHUNKS):
        key, key_chunk = jax.random.split(key)
        state, timestep, frames = fn(key_chunk, state, timestep)
    jax.block_until_ready((state, frames))
    elapsed = time.perf_counter() - start
    return NUM_CHUNKS * NUM_STEPS * NUM_ENVS / elapsed


if __name__ == "__main__":
    sps_frames = measure(with_frames=True)
    sps_plain = measure(with_frames=False)
    frame_mb = RENDER_SIZE * RENDER_SIZE * 3 / 1e6
    print(f"{NUM_ENVS} envs, {RENDER_SIZE}x{RENDER_SIZE} uint8 frames")
    print(f"with rendering:    {sps_frames / 1e3:8.0f}k steps/s ({sps_frames * frame_mb / 1e3:.1f} GB/s of frames)")
    print(f"without rendering: {sps_plain / 1e3:8.0f}k steps/s")
