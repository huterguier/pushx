"""Train a PPO agent on PushX with ``rlax`` (https://github.com/huterguier/rlax).

Run with:

    python examples/ppo.py

With the default settings (50M steps, ~20 minutes on an RTX 4090) the agent
solves ~90% of greedy episodes on "T"; a final greedy rollout is saved as
``ppo_T.gif``. Two choices matter far more than the PPO
hyperparameters here: the pusher_approach reward term (without it there is no
learning signal until the assembly is accidentally hit) and the pusher-relative
delta action space from ``pushx.wrappers.RelativeAction`` (absolute position
targets plateau at <10% success).
"""

from pathlib import Path

from rlax import PPO, PPOConfig

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from gxm.wrappers import AutoReset, TimeLimit
from gxm.wrappers.wrapper import WrapperState

from pushx import ASSEMBLIES, PushEnv, rewards
from pushx.wrappers import ActionRepeat, RelativeAction

SHAPE = "P"  # change to "X", "L", "H", "I", "+", or any key in ASSEMBLIES
ACTION_DELTA = 0.5  # policy outputs an offset from the pusher, not an absolute target
ACTION_REPEAT = 10  # hold each action 10 sim steps (dt=0.01) -> 10 Hz control, as PushT does
TIME_LIMIT = 100  # in control steps; at 10 Hz this is 10 seconds per episode


def featurize(obs):
    """Expand the raw observation into network-friendly features.

    Raw obs: [assembly_pos(2), assembly_angle, pusher_pos(2), goal_pos(2), goal_angle].
    Angles are raw radians (unbounded, wrapping at ±π), which an MLP handles
    poorly — encode them as sin/cos, including of the angle *difference* that
    success is measured on, and add goal/pusher positions relative to the block.
    """
    pos, angle = obs[..., 0:2], obs[..., 2]
    pusher, goal_pos, goal_angle = obs[..., 3:5], obs[..., 5:7], obs[..., 7]
    angle_diff = angle - goal_angle
    return jnp.concatenate(
        [
            pos,
            pusher,
            goal_pos,
            pusher - pos,
            goal_pos - pos,
            jnp.stack(
                [
                    jnp.sin(angle),
                    jnp.cos(angle),
                    jnp.sin(goal_angle),
                    jnp.cos(goal_angle),
                    jnp.sin(angle_diff),
                    jnp.cos(angle_diff),
                ],
                axis=-1,
            ),
        ],
        axis=-1,
    )


class ActorCritic(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, obs):
        x = featurize(obs)
        x = nn.tanh(nn.Dense(features=256)(x))
        x = nn.tanh(nn.Dense(features=256)(x))
        mean = ACTION_DELTA * nn.tanh(
            nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        )
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.7), (self.n_actions,)
        )
        std = jnp.broadcast_to(jnp.exp(log_std), mean.shape)
        value = nn.Dense(features=1)(x)

        return distrax.MultivariateNormalDiag(mean, std), value.squeeze(-1)


def save_gif(env, params, network, key, path):
    """Roll out one greedy episode and save it as a GIF."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    def render(state):
        while isinstance(state, WrapperState):  # peel wrapper states down to PushState
            state = state.wrapped_state
        return env.render(state, width=200, height=200)

    render = jax.jit(render)
    step = jax.jit(env.step)

    key, key_init = jax.random.split(key)
    state, timestep = jax.jit(env.init)(key_init)
    frames = [np.array(render(state))]
    for _ in range(TIME_LIMIT):
        dist, _ = network.apply(params, timestep.next_obs)
        key, key_step = jax.random.split(key)
        state, timestep = step(key_step, state, dist.mode())
        frames.append(np.array(render(state)))
        if bool(timestep.done):
            break

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    plt.tight_layout()
    image = ax.imshow(frames[0], interpolation="nearest")
    anim = animation.FuncAnimation(
        fig, lambda i: image.set_data(frames[i]), frames=len(frames), interval=1000 / (100 / ACTION_REPEAT)
    )
    anim.save(path, writer="pillow", fps=60)
    plt.close(fig)


if __name__ == "__main__":
    num_steps = int(5e7)
    num_epochs = 10
    learning_rate = 1e-4
    config = PPOConfig(
        num_envs=256,
        num_steps_rollout=64,
        num_minibatches=8,
        num_update_epochs=4,
        gamma=0.99,
        lam=0.95,
        ratio_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=1e-4,
        normalize_advantage=True,
        max_grad_norm=0.5,
        value_clip=0.2,
    )

    # Potential shaping alone gives no signal until the assembly moves; the
    # pusher_approach term rewards closing in on it, which PPO needs to get
    # off the ground from scratch. The extra sparse_success weight makes
    # actually finishing the episode worth far more than the shaping terms.
    reward_fn = rewards.combine(
        rewards.potential_shaping(),
        rewards.pusher_approach(weight=0.5),
        rewards.sparse_success,
        weights=[1.0, 1.0, 4.0],
    )
    env = TimeLimit(
        RelativeAction(
            # inside RelativeAction: the absolute target is held for the whole
            # control step, rather than the delta re-applying every sim step
            ActionRepeat(
                PushEnv(ASSEMBLIES[SHAPE], reward_fn=reward_fn),
                repeat=ACTION_REPEAT),
            scale=ACTION_DELTA),
        time_limit=TIME_LIMIT,
    ).seal()
    network = ActorCritic(env.action_space.shape[0])
    num_optimizer_steps = (
        config.num_loops(num_steps) * config.num_update_epochs * config.num_minibatches
    )
    optimizer = optax.adam(optax.linear_schedule(learning_rate, 0.0, num_optimizer_steps))

    ppo = PPO(config, AutoReset(env), network, optimizer)

    key = jax.random.key(0)
    key, key_init, key_eval, key_gif = jax.random.split(key, 4)
    ppo_state = ppo.init(key_init)
    Path(f"ppo_{SHAPE}_0.msgpack").write_bytes(serialization.to_bytes(ppo_state.params))
    print("epoch  0:", ppo.evaluate(key_eval, ppo_state, num_steps=TIME_LIMIT, num_envs=256))
    for epoch in range(1, num_epochs + 1):
        key, key_train, key_eval = jax.random.split(key, 3)
        ppo_state = ppo.train(key_train, ppo_state, num_steps=num_steps // num_epochs)
        Path(f"ppo_{SHAPE}_{epoch}.msgpack").write_bytes(serialization.to_bytes(ppo_state.params))
        print(f"epoch {epoch:2d}:", ppo.evaluate(key_eval, ppo_state, num_steps=TIME_LIMIT, num_envs=256))
    Path(f"ppo_{SHAPE}.msgpack").write_bytes(serialization.to_bytes(ppo_state.params))
    save_gif(env, ppo_state.params, network, key_gif, f"ppo_{SHAPE}.gif")
