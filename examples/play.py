import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from pushx import ASSEMBLIES, PushEnv
from pushx.wrappers import PixelObservation

SHAPE = "X"  # change to "T", "L", "H", "I", "+", or any key in ASSEMBLIES
RENDER_SIZE = 200

env = PixelObservation(PushEnv(ASSEMBLIES[SHAPE]), width=RENDER_SIZE, height=RENDER_SIZE)
step_jit = jax.jit(env.step)
reset_jit = jax.jit(env.reset)

key = jax.random.key(0)
state, timestep = env.init(key)
step_count = 0

mouse_pos = np.array([1.0, 1.0])

fig, ax = plt.subplots(figsize=(5, 5))
ax.axis("off")
title = plt.title(f"Move mouse to push {SHAPE}  ({RENDER_SIZE}×{RENDER_SIZE}px)", fontsize=10)
plt.tight_layout()

img_display = ax.imshow(
    np.array(timestep.next_obs), interpolation="nearest"
)


def on_move(event):
    if event.inaxes:
        mouse_pos[0] = event.xdata / RENDER_SIZE * 2.0
        mouse_pos[1] = (1.0 - event.ydata / RENDER_SIZE) * 2.0


fig.canvas.mpl_connect("motion_notify_event", on_move)


def update(_):
    global state, key, step_count
    key, sk = jax.random.split(key)
    action = jnp.array(mouse_pos)
    state, timestep = step_jit(sk, state, action)
    step_count += 1

    if timestep.terminated:
        title.set_text(f"Solved in {step_count} steps! Resetting...")
        fig.canvas.draw()
        key, rk = jax.random.split(key)
        state, timestep = reset_jit(rk, state)
        step_count = 0
        title.set_text(f"Move mouse to push {SHAPE}  ({RENDER_SIZE}×{RENDER_SIZE}px)")

    img_display.set_data(np.array(timestep.next_obs))
    return (img_display,)


ani = animation.FuncAnimation(fig, update, interval=16, blit=True)
plt.show()
