<div align="center">
    <img src="https://github.com/huterguier/pushx/blob/main/images/pushx.png" width="170">
</div>

# PushT Environments in JAX
[![PyPI version](https://img.shields.io/badge/pypi-not_available-red.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-1d8a50.svg)](https://opensource.org/licenses/MIT)

[`pushx`](https://github.com/huterguier/pushx) implements PushT-style manipulation tasks in [JAX](https://github.com/jax-ml/jax),
built on top of [`gxm`](https://github.com/huterguier/gxm) and the [`jax2d`](https://github.com/MichaelTMatthews/Jax2D) physics engine.
A circular pusher shoves a rigid 2D assembly, built out of welded rectangular strokes, toward a target pose.
The whole `init` / `step` / `render` pipeline is `jit`, `vmap` and `scan` friendly.

## Features
- 🔤 **Configurable Shapes:** Assemblies are built from welded rectangular strokes, so any letter- or symbol-shaped target (`T`, `X`, `I`, `L`, `+`, `H`, ...) is just a small declarative description away.
- 🎯 **Pluggable Rewards & Success Criteria:** Swap in sparse, dense, or shaped rewards, and choose how strict "solved" needs to be, without touching the environment itself.
- 🧩 **Composable Wrappers:** Action-space transforms (normalized, relative, velocity-based, discrete) and pixel observations are `gxm.Wrapper`s, so they stack freely with each other and with `gxm`'s own built-in wrappers.
- 🖼️ **JAX-native Rendering:** A hand-rolled signed-distance-field rasterizer renders the scene directly in JAX, no external graphics dependencies required.

## Quick Start

```python
import jax
import jax.numpy as jnp
from pushx import ASSEMBLIES, PushEnv

env = PushEnv(ASSEMBLIES["X"])
key = jax.random.key(0)

state, timestep = env.init(key)
state, timestep = jax.jit(env.step)(key, state, jnp.array([1.0, 1.0]))
image = env.render(state, width=64, height=64)
```

Or through `gxm`'s registry:
```python
import gxm
env = gxm.make("PushX/X")
```

### Rewards & Success Criteria
Both are plain, swappable callables passed to `PushEnv`.
```python
from pushx import rewards, termination

env = PushEnv(
    ASSEMBLIES["X"],
    reward_fn=rewards.combine(rewards.dense_distance(), rewards.step_penalty(0.01)),
    success_fn=termination.pose_threshold(pos_threshold=0.1, angle_threshold=0.1),
)
```
`pushx.rewards` includes `sparse_success`, `dense_distance`, `potential_shaping`, `step_penalty`, and
`combine(*reward_fns, weights=None)` for mixing several together.
`pushx.termination` includes `pose_threshold(pos_threshold, angle_threshold)`.

By default the goal pose and the initial state are sampled every episode, as in PushT. Pass
`goal=((x, y), angle)` and/or `spawn=(block_pos, block_angle, pusher_pos)` to fix them; a fixed
spawn is jittered by `spawn_jitter=(distance, angle)`:
```python
env = PushEnv(ASSEMBLIES["T"], goal=((0.6, 0.6), 0.0), spawn=((1.3, 1.3), 0.0, (1.7, 1.7)))
```

### Action & Observation Wrappers
The default action space is an absolute 2D world-coordinate pusher target. `pushx.wrappers` provides
alternatives, each a `gxm.Wrapper` and therefore composable with one another and with `gxm`'s own wrappers.
```python
from gxm.wrappers import Discretize
from pushx.wrappers import NormalizeAction, RelativeAction, VelocityAction, PixelObservation

env = NormalizeAction(PushEnv(ASSEMBLIES["X"]))   # actions in [-1, 1]^2
env = RelativeAction(PushEnv(ASSEMBLIES["X"]))    # actions are a delta from the pusher's position
env = VelocityAction(PushEnv(ASSEMBLIES["X"]))    # actions are a direct velocity command

# Discrete directions: stack gxm's Discretize on top of any continuous action wrapper
directions = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
env = Discretize(VelocityAction(PushEnv(ASSEMBLIES["X"])), actions=directions)

# Pixel observations instead of the raw-state vector
env = PixelObservation(PushEnv(ASSEMBLIES["X"]), width=64, height=64)
```
Time limits and auto-resetting are handled the same way, via `gxm`'s own wrappers.
```python
from gxm.wrappers import AutoReset, TimeLimit
env = AutoReset(TimeLimit(PushEnv(ASSEMBLIES["X"]), time_limit=200))
```

## Examples
```bash
python examples/play.py         # mouse-driven pusher
python examples/play_arrows.py  # arrow-key discrete pusher
python examples/ppo.py          # train a PPO agent (fully jit-compiled)
```
The PPO example uses [`rlax`](https://github.com/huterguier/rlax), which is included in the `examples` extra.
```bash
pip install "pushx[examples] @ git+https://github.com/huterguier/pushx"
```

## Installation
`pushx` can be installed directly from this GitHub repository.
```bash
pip install git+https://github.com/huterguier/pushx
```
By default `pushx` installs a CPU-only build of JAX. For CUDA support, include the optional dependency.
```bash
pip install "pushx[cuda] @ git+https://github.com/huterguier/pushx"
```
