"""Concrete Assembly definitions for each supported letter/shape."""

from __future__ import annotations

import math

from pushx.dynamics import SCENE_SIZE
from pushx.shapes import Assembly, Stroke, Weld

# ---------------------------------------------------------------------------
# T  (symmetry_order=1 — asymmetric)
#
# Sized and positioned to match stable_worldmodel's PushT `add_tee` exactly (bar
# 120x30px, stem 30x90px on PushT's 512px window), scaled by SCENE_SIZE/512 -- so
# the same world position renders the same shape in both envs, modulo PushT's
# y-down vs pushx's y-up convention (render.py flips y when rasterizing). PushT's
# body origin sits at the bar's top edge (its local (0,0), not the bar's center),
# so the strokes below are offset from that same reference point, not from a
# centroid, to keep the two origins interchangeable under a straight scale+flip.
# ---------------------------------------------------------------------------
_PUSHT_PX_SCALE = SCENE_SIZE / 512.0
_BAR_W, _BAR_H = 120.0 * _PUSHT_PX_SCALE, 30.0 * _PUSHT_PX_SCALE
_STEM_W, _STEM_H = 30.0 * _PUSHT_PX_SCALE, 90.0 * _PUSHT_PX_SCALE

T = Assembly(
    strokes=[
        Stroke(offset=(0.0, -_BAR_H / 2), angle=0.0, size=(_BAR_W, _BAR_H)),
        Stroke(
            offset=(0.0, -(_BAR_H + _STEM_H / 2)), angle=0.0, size=(_STEM_W, _STEM_H)
        ),
    ],
    welds=[
        Weld(
            a=0,
            b=1,
            a_relative_pos=(0.0, -_BAR_H / 2),
            b_relative_pos=(0.0, _STEM_H / 2),
        ),
    ],
    symmetry_order=1,
)

# ---------------------------------------------------------------------------
# X  (symmetry_order=4 — π/2 period; bars at ±45° are visually swappable)
# ---------------------------------------------------------------------------
_X_LEN, _X_THK = 0.45, 0.09

X = Assembly(
    strokes=[
        Stroke(offset=(0.0, 0.0), angle=math.pi / 4, size=(_X_LEN, _X_THK)),
        Stroke(offset=(0.0, 0.0), angle=-math.pi / 4, size=(_X_LEN, _X_THK)),
    ],
    welds=[
        Weld(a=0, b=1, a_relative_pos=(0.0, 0.0), b_relative_pos=(0.0, 0.0)),
    ],
    symmetry_order=4,
)

# ---------------------------------------------------------------------------
# I  (symmetry_order=2 — single tall bar; π rotation is identical)
# ---------------------------------------------------------------------------
_I_W, _I_H = 0.09, 0.45

I = Assembly(
    strokes=[
        Stroke(offset=(0.0, 0.0), angle=0.0, size=(_I_W, _I_H)),
    ],
    welds=[],
    symmetry_order=2,
)

# ---------------------------------------------------------------------------
# L  (symmetry_order=1 — vertical bar with foot extending right)
#
#   Assembly origin = center of vertical bar.
#   Foot attaches at the bottom of the vertical and extends to the right.
# ---------------------------------------------------------------------------
_L_THK = 0.09
_L_H = 0.38
_L_ARM = 0.22

L = Assembly(
    strokes=[
        Stroke(offset=(0.0, 0.0), angle=0.0, size=(_L_THK, _L_H)),
        # Arm left edge flush with vertical left edge: shift center left by _L_THK/2.
        # Arm bottom edge flush with vertical bottom: shift center up by _L_THK/2.
        Stroke(
            offset=(_L_ARM / 2 - _L_THK / 2, -_L_H / 2 + _L_THK / 2),
            angle=0.0,
            size=(_L_ARM, _L_THK),
        ),
    ],
    welds=[
        Weld(
            a=0,
            b=1,
            a_relative_pos=(0.0, -_L_H / 2 + _L_THK / 2),
            b_relative_pos=(-_L_ARM / 2 + _L_THK / 2, 0.0),
        ),
    ],
    symmetry_order=1,
)

# ---------------------------------------------------------------------------
# +  (symmetry_order=4 — two axis-aligned bars crossing at center)
#
#   Bar 1 is at 90° so stroke 0 is horizontal and stroke 1 is vertical.
#   Rotating by π/2 swaps them; identical size → same visual.
# ---------------------------------------------------------------------------
_PLUS_LEN, _PLUS_THK = 0.42, 0.09

PLUS = Assembly(
    strokes=[
        Stroke(offset=(0.0, 0.0), angle=0.0, size=(_PLUS_LEN, _PLUS_THK)),
        Stroke(offset=(0.0, 0.0), angle=math.pi / 2, size=(_PLUS_LEN, _PLUS_THK)),
    ],
    welds=[
        Weld(a=0, b=1, a_relative_pos=(0.0, 0.0), b_relative_pos=(0.0, 0.0)),
    ],
    symmetry_order=4,
)

# ---------------------------------------------------------------------------
# H  (symmetry_order=2 — two verticals connected by a horizontal crossbar)
#
#   Assembly origin = center of bounding box.
#   Two welds: left-vertical↔crossbar and right-vertical↔crossbar.
# ---------------------------------------------------------------------------
_H_THK = 0.09
_H_H = 0.40
_H_W = 0.30  # center-to-center distance between the two verticals

H = Assembly(
    strokes=[
        Stroke(
            offset=(-_H_W / 2, 0.0), angle=0.0, size=(_H_THK, _H_H)
        ),  # left vertical
        Stroke(
            offset=(_H_W / 2, 0.0), angle=0.0, size=(_H_THK, _H_H)
        ),  # right vertical
        # Crossbar spans full center-to-center width so it overlaps into both verticals,
        # eliminating the internal border lines at the junctions.
        Stroke(offset=(0.0, 0.0), angle=0.0, size=(_H_W, _H_THK)),  # crossbar
    ],
    welds=[
        Weld(
            a=0,
            b=2,
            a_relative_pos=(_H_THK / 2, 0.0),
            b_relative_pos=(-_H_W / 2 + _H_THK / 2, 0.0),
        ),
        Weld(
            a=1,
            b=2,
            a_relative_pos=(-_H_THK / 2, 0.0),
            b_relative_pos=(_H_W / 2 - _H_THK / 2, 0.0),
        ),
    ],
    symmetry_order=2,
)

# ---------------------------------------------------------------------------
# P  (symmetry_order=1 — stem with a closed rectangular bowl at the top right)
#
#   Assembly origin = center of stem.
#   Bowl = top bar + middle bar + right vertical; bars are flush with the stem's
#   left edge, the right vertical spans the full bowl height.
# ---------------------------------------------------------------------------
_P_THK = 0.09
_P_H = 0.42
_P_W = 0.28  # total width, stem left edge to bowl right edge
_P_BOWL_H = 0.25  # outer height of the bowl

_P_BAR_X = _P_W / 2 - _P_THK / 2
_P_TOP_Y = _P_H / 2 - _P_THK / 2
_P_MID_Y = _P_H / 2 - _P_BOWL_H + _P_THK / 2
_P_RIGHT_X = _P_W - _P_THK
_P_RIGHT_Y = _P_H / 2 - _P_BOWL_H / 2

P = Assembly(
    strokes=[
        Stroke(offset=(0.0, 0.0), angle=0.0, size=(_P_THK, _P_H)),  # stem
        Stroke(offset=(_P_BAR_X, _P_TOP_Y), angle=0.0, size=(_P_W, _P_THK)),  # top bar
        Stroke(
            offset=(_P_BAR_X, _P_MID_Y), angle=0.0, size=(_P_W, _P_THK)
        ),  # middle bar
        Stroke(
            offset=(_P_RIGHT_X, _P_RIGHT_Y), angle=0.0, size=(_P_THK, _P_BOWL_H)
        ),  # right vertical
    ],
    welds=[
        Weld(a=0, b=1, a_relative_pos=(0.0, _P_TOP_Y), b_relative_pos=(-_P_BAR_X, 0.0)),
        Weld(a=0, b=2, a_relative_pos=(0.0, _P_MID_Y), b_relative_pos=(-_P_BAR_X, 0.0)),
        Weld(
            a=1,
            b=3,
            a_relative_pos=(_P_RIGHT_X - _P_BAR_X, 0.0),
            b_relative_pos=(0.0, _P_TOP_Y - _P_RIGHT_Y),
        ),
        # No middle-bar <-> right-vertical weld: closing the loop over-constrains
        # jax2d's joint solver and produces occasional NaN states.
    ],
    symmetry_order=1,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ASSEMBLIES: dict[str, Assembly] = {
    "T": T,
    "X": X,
    "I": I,
    "L": L,
    "+": PLUS,
    "H": H,
    "P": P,
}
