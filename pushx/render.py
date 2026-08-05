from __future__ import annotations

import jax.numpy as jnp

from pushx.model import PUSHER_RADIUS, SCENE_SIZE
from pushx.shapes import Assembly

# pymunk's pygame debug-draw renders every shape two-toned: outline in the exact
# requested color, fill lightened by 1.2x (clamped) -- confirmed by sampling actual
# rendered PushT pixels (block outline (119,136,153) vs fill (143,163,184), pusher
# outline (65,105,225) vs fill (78,126,255): both ratios are 1.20 on every unclipped
# channel). The goal has no separate outline (drawn with a flat pygame.draw.polygon).
_ROYAL_BLUE = jnp.array([0.2549, 0.4118, 0.8824])
_ROYAL_BLUE_FILL = jnp.clip(_ROYAL_BLUE * 1.2, 0.0, 1.0)
_LIGHT_SLATE_GRAY = jnp.array([0.4667, 0.5333, 0.6])
_LIGHT_SLATE_GRAY_FILL = jnp.clip(_LIGHT_SLATE_GRAY * 1.2, 0.0, 1.0)
_LIGHT_GREEN = jnp.array([0.5647, 0.9333, 0.5647])


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
    ys = jnp.linspace(SCENE_SIZE, 0, height)
    grid_x, grid_y = jnp.meshgrid(xs, ys)
    pixels = jnp.stack([grid_x, grid_y], axis=-1)

    edge_width = SCENE_SIZE / (2 * min(width, height))
    border_thickness = 4.0 * edge_width
    # Must exceed border_thickness: the border's inner edge is this SDF eroded inward
    # by border_thickness, so if overlap (how far touching strokes are grown into each
    # other before the union) doesn't reach at least that far, the erosion reaches back
    # out past the overlapped region into the same concave-corner ambiguity that
    # `overlap` exists to avoid, and a stray border line reappears at the weld seam.
    overlap = border_thickness + 2.0 * edge_width

    img = jnp.ones((height, width, 3))

    goal_pos = state.goal_pos
    goal_angle = state.goal_angle
    cos_g, sin_g = jnp.cos(goal_angle), jnp.sin(goal_angle)
    R_goal = jnp.array([[cos_g, -sin_g], [sin_g, cos_g]])

    goal_sdf = jnp.full((height, width), jnp.inf)
    for stroke in assembly.strokes:
        pos = goal_pos + R_goal @ jnp.array(stroke.offset)
        angle = goal_angle + stroke.angle
        size = (stroke.size[0] + overlap, stroke.size[1] + overlap)
        goal_sdf = jnp.minimum(goal_sdf, _rect_sdf(_rect_local(pixels, pos, angle), size))
    goal_alpha = _smoothstep(edge_width, -edge_width, goal_sdf)
    img = goal_alpha[..., None] * _LIGHT_GREEN + (1 - goal_alpha[..., None]) * img

    assembly_sdf = jnp.full((height, width), jnp.inf)
    for stroke, idx in zip(assembly.strokes, stroke_indices):
        pos = state.sim_state.polygon.position[idx]
        angle = state.sim_state.polygon.rotation[idx]
        size = (stroke.size[0] + overlap, stroke.size[1] + overlap)
        assembly_sdf = jnp.minimum(assembly_sdf, _rect_sdf(_rect_local(pixels, pos, angle), size))
    fill_alpha = _smoothstep(edge_width, -edge_width, assembly_sdf)
    inner_alpha = _smoothstep(edge_width, -edge_width, assembly_sdf + border_thickness)
    border_alpha = fill_alpha - inner_alpha
    img = fill_alpha[..., None] * _LIGHT_SLATE_GRAY_FILL + (1 - fill_alpha[..., None]) * img
    img = border_alpha[..., None] * _LIGHT_SLATE_GRAY + (1 - border_alpha[..., None]) * img

    pusher_pos = state.sim_state.circle.position[pusher_idx]
    circle_sdf = jnp.linalg.norm(pixels - pusher_pos, axis=-1) - PUSHER_RADIUS
    circle_fill_alpha = _smoothstep(edge_width, -edge_width, circle_sdf)
    circle_inner_alpha = _smoothstep(edge_width, -edge_width, circle_sdf + border_thickness)
    circle_border_alpha = circle_fill_alpha - circle_inner_alpha
    img = circle_fill_alpha[..., None] * _ROYAL_BLUE_FILL + (1 - circle_fill_alpha[..., None]) * img
    img = circle_border_alpha[..., None] * _ROYAL_BLUE + (1 - circle_border_alpha[..., None]) * img

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
