from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stroke:
    offset: tuple[float, float]   # position offset from assembly origin, in assembly-local frame
    angle: float                   # angle relative to assembly root angle (radians)
    size: tuple[float, float]      # (width, height) in stroke-local frame


@dataclass
class Weld:
    a: int                                # index into Assembly.strokes
    b: int
    a_relative_pos: tuple[float, float]   # attachment point in stroke a's local frame
    b_relative_pos: tuple[float, float]   # attachment point in stroke b's local frame


@dataclass
class Assembly:
    strokes: list[Stroke]
    welds: list[Weld]
    symmetry_order: int = 1  # k-fold rotational symmetry; goal_angle sampled from [−π/k, π/k]


