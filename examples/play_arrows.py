import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gxm.wrappers import Discretize

from pushx import ASSEMBLIES, PushEnv
from pushx.wrappers import PixelObservation, VelocityAction

SHAPE = "X"  # change to "T", "L", "H", "I", "+", or any key in ASSEMBLIES
RENDER_SIZE = 200
SPEED = 2.5  # pusher velocity while a direction key is held (max_speed clips at 3.0)

# Discrete action set: 0=stay, 1=right, 2=left, 3=up, 4=down.
DIRECTIONS = jnp.array(
    [
        [0.0, 0.0],
        [SPEED, 0.0],
        [-SPEED, 0.0],
        [0.0, SPEED],
        [0.0, -SPEED],
    ]
)
KEY_TO_ACTION = {"right": 1, "left": 2, "up": 3, "down": 4}

base_env = PushEnv(ASSEMBLIES[SHAPE])
env = PixelObservation(
    Discretize(VelocityAction(base_env), actions=DIRECTIONS), width=RENDER_SIZE, height=RENDER_SIZE
)
step_jit = jax.jit(env.step)
reset_jit = jax.jit(env.reset)

key = jax.random.key(0)
state, timestep = env.init(key)
step_count = 0
action_index = 0  # currently held direction; 0 = stay

fig, ax = plt.subplots(figsize=(5, 5))
ax.axis("off")
title = plt.title(f"Arrow keys to push {SHAPE}  ({RENDER_SIZE}×{RENDER_SIZE}px)", fontsize=10)
plt.tight_layout()

img_display = ax.imshow(np.array(timestep.next_obs), interpolation="nearest")


def on_key_press(event):
    global action_index
    if event.key in KEY_TO_ACTION:
        action_index = KEY_TO_ACTION[event.key]


def on_key_release(event):
    global action_index
    if event.key in KEY_TO_ACTION and KEY_TO_ACTION[event.key] == action_index:
        action_index = 0


fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)


def update(_):
    global state, key, step_count
    key, sk = jax.random.split(key)
    state, timestep = step_jit(sk, state, jnp.array(action_index))
    step_count += 1

    if timestep.terminated:
        title.set_text(f"Solved in {step_count} steps! Resetting...")
        fig.canvas.draw()
        key, rk = jax.random.split(key)
        state, timestep = reset_jit(rk, state)
        step_count = 0
        title.set_text(f"Arrow keys to push {SHAPE}  ({RENDER_SIZE}×{RENDER_SIZE}px)")

    img_display.set_data(np.array(timestep.next_obs))
    return (img_display,)


ani = animation.FuncAnimation(fig, update, interval=16, blit=True)
plt.show()
