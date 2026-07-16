"""Watch a trained PPO policy push the assembly, resetting after every episode.

Train first with ``python examples/ppo.py`` (saves ``ppo_T.msgpack``), then:

    python examples/watch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from ppo import ACTION_DELTA, SHAPE, ActorCritic

from pushx import ASSEMBLIES, PushEnv
from pushx.wrappers import RelativeAction

RENDER_SIZE = 64
TIME_LIMIT = 300
RESET_ON_SOLVE = False  # keep watching after success until the time limit resets

env = RelativeAction(PushEnv(ASSEMBLIES[SHAPE]), scale=ACTION_DELTA)
network = ActorCritic(env.action_space.shape[0])

key = jax.random.key(0)
state, timestep = jax.jit(env.init)(key)
params = network.init(key, timestep.next_obs)
params_path = Path(__file__).parent / f"ppo_{SHAPE}.msgpack"
params = serialization.from_bytes(params, params_path.read_bytes())

# Sample rather than take the mode: the deterministic action can stall in
# fixed points that the training-time action noise escapes.
policy = jax.jit(lambda key, obs: network.apply(params, obs)[0].sample(seed=key))
render = jax.jit(lambda state: env.render(state, width=RENDER_SIZE, height=RENDER_SIZE))
step = jax.jit(env.step)
reset = jax.jit(env.reset)
step_count = 0

fig, ax = plt.subplots(figsize=(5, 5))
ax.axis("off")
title = plt.title(f"Trained policy pushing {SHAPE}", fontsize=10)
plt.tight_layout()
img_display = ax.imshow(np.array(render(state)), interpolation="nearest")


def update(_):
    global state, timestep, key, step_count
    key, key_action, key_step = jax.random.split(key, 3)
    state, timestep = step(key_step, state, policy(key_action, timestep.next_obs))
    step_count += 1

    if (RESET_ON_SOLVE and timestep.terminated) or step_count >= TIME_LIMIT:
        outcome = "Solved" if timestep.terminated else "Episode over"
        title.set_text(f"{outcome} after {step_count} steps! Resetting...")
        fig.canvas.draw()
        key, key_reset = jax.random.split(key)
        state, timestep = reset(key_reset, state)
        step_count = 0
        title.set_text(f"Trained policy pushing {SHAPE}")

    img_display.set_data(np.array(render(state)))
    return (img_display,)


ani = animation.FuncAnimation(fig, update, interval=16, blit=True)
plt.show()
