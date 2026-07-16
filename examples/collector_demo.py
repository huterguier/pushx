"""Smoke-test the ``collector`` package against the PushX + PPO setup.

Reproduces ``examples/generate.py`` with the generic components: a mixture of
uniform random envs and an ensemble of PPO checkpoints drives 1024 parallel
environments, and a frames extractor records frames/actions/masks.

    python examples/collector_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import cast

import jax
import jax.numpy as jnp
from collector import Collector, extractors, policies
from flax import serialization
from gxm.spaces import Box
from ppo import ACTION_DELTA, SHAPE, ActorCritic

from pushx import ASSEMBLIES, PushEnv
from pushx.wrappers import LingerReset, RelativeAction

NUM_ENVS = 1024
NUM_STEPS = 64
RENDER_SIZE = 64
RANDOM_FRACTION = 0.125

env = LingerReset(RelativeAction(PushEnv(ASSEMBLIES[SHAPE]), scale=ACTION_DELTA))
action_space = cast(Box, env.action_space)
network = ActorCritic(action_space.shape[0])


def ppo_policy(key, params, obs):
    dist, _ = network.apply(params, obs)
    return dist.sample(seed=key)


def load_checkpoints(template):
    """Stack all saved checkpoints into one pytree with a leading axis."""
    paths = sorted(
        Path(__file__).parent.glob(f"ppo_{SHAPE}_[0-9]*.msgpack"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not paths:  # no per-epoch checkpoints yet; fall back to the final params
        paths = [Path(__file__).parent / f"ppo_{SHAPE}.msgpack"]
    checkpoints = [serialization.from_bytes(template, p.read_bytes()) for p in paths]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *checkpoints)


if __name__ == "__main__":
    key = jax.random.key(0)
    _, timestep = env.init(key)
    checkpoints = load_checkpoints(network.init(key, timestep.next_obs))
    num_checkpoints = jax.tree.leaves(checkpoints)[0].shape[0]

    collector = Collector(
        env=env,
        policy=policies.Mixture(
            [policies.Random(action_space), policies.Ensemble(ppo_policy, checkpoints)],
            [RANDOM_FRACTION, 1.0 - RANDOM_FRACTION],
        ),
        extractor=extractors.frames(env, width=RENDER_SIZE, height=RENDER_SIZE),
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
    )

    state = collector.init(key)
    collect = jax.jit(collector.collect)

    key, key_warmup = jax.random.split(key)
    jax.block_until_ready(collect(key_warmup, state))  # compile

    num_chunks = 5
    batch = {}
    start = time.perf_counter()
    for _ in range(num_chunks):
        key, key_chunk = jax.random.split(key)
        state, batch = collect(key_chunk, state)
    jax.block_until_ready(batch)
    sps = num_chunks * NUM_STEPS * NUM_ENVS / (time.perf_counter() - start)

    print(f"{num_checkpoints} checkpoints + {int(NUM_ENVS * RANDOM_FRACTION)} random envs")
    for name, x in batch.items():
        print(f"{name:8s} {x.dtype} {x.shape}")
    print(f"done fraction:    {float(batch['done'].mean()):.4f}")
    print(f"success fraction: {float(batch['reward'].mean()):.4f}")
    print(f"{sps / 1e3:.0f}k frames/s")
