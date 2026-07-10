from __future__ import annotations

import jax.numpy as jnp

from pushx.model import PUSHER_RADIUS, SCENE_SIZE
from pushx.shapes import Assembly


def _render_pixels(
    state,
    assembly: Assembly,
    stroke_indices: list[int],
    pusher_idx: int,
    width: int = 64,
    height: int = 64,
) -> jnp.ndarray:
    """JAX-native renderer. Returns (H, W, 3) float32 in [0, 1]. Jit/vmap-compatible."""
    xs = jnp.linspace(0, SCENE_SIZE, width)
    ys = jnp.linspace(SCENE_SIZE, 0, height)  # flip: row 0 = top of image = high y
    grid_x, grid_y = jnp.meshgrid(xs, ys)
    pixels = jnp.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)

    edge_width = SCENE_SIZE / (2 * min(width, height))

    img = jnp.ones((height, width, 3))  # white background

    goal_pos = state.goal_pos
    goal_angle = state.goal_angle
    cos_g, sin_g = jnp.cos(goal_angle), jnp.sin(goal_angle)
    R_goal = jnp.array([[cos_g, -sin_g], [sin_g, cos_g]])

    # Goal: union SDF across all strokes, single composite
    goal_sdf = jnp.full((height, width), jnp.inf)
    for stroke in assembly.strokes:
        pos = goal_pos + R_goal @ jnp.array(stroke.offset)
        angle = goal_angle + stroke.angle
        goal_sdf = jnp.minimum(goal_sdf, _rect_sdf(_rect_local(pixels, pos, angle), stroke.size))
    goal_alpha = _smoothstep(edge_width, -edge_width, goal_sdf) * 0.4
    img = goal_alpha[..., None] * jnp.array([0.298, 0.686, 0.314]) + (1 - goal_alpha[..., None]) * img

    # Assembly: union SDF across all strokes, single fill + border composite
    assembly_sdf = jnp.full((height, width), jnp.inf)
    for stroke, idx in zip(assembly.strokes, stroke_indices):
        pos = state.sim_state.polygon.position[idx]
        angle = state.sim_state.polygon.rotation[idx]
        assembly_sdf = jnp.minimum(assembly_sdf, _rect_sdf(_rect_local(pixels, pos, angle), stroke.size))
    border_thickness = 1.5 * edge_width
    fill_alpha = _smoothstep(edge_width, -edge_width, assembly_sdf)
    inner_alpha = _smoothstep(edge_width, -edge_width, assembly_sdf + border_thickness)
    img = fill_alpha[..., None] * jnp.array([0.420, 0.549, 0.749]) + (1 - fill_alpha[..., None]) * img
    img = (fill_alpha - inner_alpha)[..., None] * jnp.array([0.290, 0.392, 0.569]) + (1 - (fill_alpha - inner_alpha))[..., None] * img

    pusher_pos = state.sim_state.circle.position[pusher_idx]
    alpha = _circle_alpha(pixels, pusher_pos, PUSHER_RADIUS, edge_width)
    img = alpha[..., None] * jnp.array([0.878, 0.878, 0.878]) + (1 - alpha[..., None]) * img

    return img


def _smoothstep(edge0: float, edge1: float, x: jnp.ndarray) -> jnp.ndarray:
    t = jnp.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rect_sdf(p_local: jnp.ndarray, size: tuple[float, float]) -> jnp.ndarray:
    q = jnp.abs(p_local) - jnp.array([size[0] / 2, size[1] / 2])
    return jnp.linalg.norm(jnp.maximum(q, 0), axis=-1) + jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0)


def _rect_local(pixels: jnp.ndarray, center: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    cos_a, sin_a = jnp.cos(-angle), jnp.sin(-angle)
    R_inv = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return jnp.einsum("ij,...j->...i", R_inv, pixels - center)


def _circle_alpha(
    pixels: jnp.ndarray,
    center: jnp.ndarray,
    radius: float,
    edge_width: float,
) -> jnp.ndarray:
    sdf = jnp.linalg.norm(pixels - center, axis=-1) - radius
    return _smoothstep(edge_width, -edge_width, sdf)
