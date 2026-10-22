"""Stream world-model training data from parallel rollouts of mixed policies.

``generate`` scans NUM_STEPS environment steps over NUM_ENVS parallel
environments and returns uint8 frames plus actions and episode-boundary masks,
keeping everything on device — call it in a loop and feed each batch straight
into a world-model gradient step, carrying ``(state, timestep)`` between calls.

Each environment is driven by one of the policy checkpoints saved by
``python examples/ppo.py`` (``checkpoints/ppo_T_{epoch}.msgpack``, from untrained to
converged), except for a RANDOM_FRACTION of environments that take uniform
random actions. Episodes reset via ``pushx.wrappers.LingerReset``: a few dozen
steps after success or at a randomized time limit, so resets stay
desynchronized across environments. Success never terminates an episode (it is
returned as the ``success`` mask); ``done`` marks the reset boundaries, whose
transitions should be masked (or used to reset the world model's latent).

Run this file directly to smoke-test batch shapes, composition, and throughput:

    python examples/generate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from flax import serialization
from gxm.wrappers.wrapper import WrapperState
from ppo import ACTION_DELTA, CHECKPOINTS, SHAPE, ActorCritic

from pushx import ASSEMBLIES, PushEnv
from pushx.wrappers import LingerReset, RelativeAction

NUM_ENVS = 1024
NUM_STEPS = 64  # steps per batch; frames are (NUM_STEPS, NUM_ENVS, H, W, 3)
RENDER_SIZE = 64
RANDOM_FRACTION = 0.125  # fraction of envs driven by uniform random actions

env = LingerReset(RelativeAction(PushEnv(ASSEMBLIES[SHAPE]), scale=ACTION_DELTA))
network = ActorCritic(env.action_space.shape[0])


def load_policies(template):
    """Stack all saved checkpoints into one pytree with a leading axis."""
    paths = sorted(
        CHECKPOINTS.glob(f"ppo_{SHAPE}_[0-9]*.msgpack"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not paths:  # no per-epoch checkpoints yet; fall back to the final params
        paths = [CHECKPOINTS / f"ppo_{SHAPE}.msgpack"]
    policies = [serialization.from_bytes(template, p.read_bytes()) for p in paths]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *policies)


def render(state):
    while isinstance(state, WrapperState):  # peel wrapper states down to PushState
        state = state.wrapped_state
    pixels = env.render(state, width=RENDER_SIZE, height=RENDER_SIZE)
    return (pixels * 255).astype(jnp.uint8)


def make_generate(policies):
    num_policies = jax.tree.leaves(policies)[0].shape[0]
    # Block assignment: env i runs checkpoint i * K // B; the first
    # RANDOM_FRACTION of envs are overridden with uniform random actions.
    policy_idx = jnp.arange(NUM_ENVS) * num_policies // NUM_ENVS
    is_random = jnp.arange(NUM_ENVS) < int(NUM_ENVS * RANDOM_FRACTION)

    def act(key, obs):
        key_policies, key_random = jax.random.split(key)

        def sample(params, key):
            dist, _ = network.apply(params, obs)
            return dist.sample(seed=key)

        # The policy MLP is tiny: evaluating every checkpoint on the full batch
        # and selecting per env is cheaper than gathering per-env parameters.
        actions = jax.vmap(sample)(
            policies, jax.random.split(key_policies, num_policies)
        )
        actions = jnp.take_along_axis(actions, policy_idx[None, :, None], axis=0)[0]
        random_actions = jax.random.uniform(
            key_random,
            (NUM_ENVS, env.action_space.shape[0]),
            minval=env.action_space.low,
            maxval=env.action_space.high,
        )
        return jnp.where(is_random[:, None], random_actions, actions)

    def generate(key, state, timestep):
        def body(carry, key):
            state, timestep = carry
            key_action, key_step = jax.random.split(key)
            actions = act(key_action, timestep.next_obs)
            state, timestep = jax.vmap(env.step)(
                jax.random.split(key_step, NUM_ENVS), state, actions
            )
            batch = {
                "frames": jax.vmap(render)(state),
                "action": actions,
                "done": timestep.truncated,
                "success": timestep.info["success"],
            }
            return (state, timestep), batch

        keys = jax.random.split(key, NUM_STEPS)
        (state, timestep), batch = jax.lax.scan(body, (state, timestep), keys)
        return state, timestep, batch

    return generate


if __name__ == "__main__":
    import time

    key = jax.random.key(0)
    state, timestep = jax.vmap(env.init)(jax.random.split(key, NUM_ENVS))
    policies = load_policies(network.init(key, timestep.next_obs))
    num_policies = jax.tree.leaves(policies)[0].shape[0]
    generate = jax.jit(make_generate(policies))

    key, key_warmup = jax.random.split(key)
    jax.block_until_ready(generate(key_warmup, state, timestep))  # compile

    num_chunks = 5
    start = time.perf_counter()
    for _ in range(num_chunks):
        key, key_chunk = jax.random.split(key)
        state, timestep, batch = generate(key_chunk, state, timestep)
    jax.block_until_ready(batch)
    sps = num_chunks * NUM_STEPS * NUM_ENVS / (time.perf_counter() - start)

    print(f"{num_policies} checkpoints + {int(NUM_ENVS * RANDOM_FRACTION)} random envs")
    for name, x in batch.items():
        print(f"{name:8s} {x.dtype} {x.shape}")
    print(f"done fraction:    {float(batch['done'].mean()):.4f}")
    print(f"success fraction: {float(batch['success'].mean()):.4f}")
    print(f"{sps / 1e3:.0f}k frames/s")
