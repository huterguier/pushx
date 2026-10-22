from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stroke:
    offset: tuple[float, float]  # from assembly origin, in assembly-local frame
    angle: float  # relative to assembly root angle (radians)
    size: tuple[float, float]  # (width, height) in stroke-local frame


@dataclass(frozen=True)
class Weld:
    a: int  # index into Assembly.strokes
    b: int
    a_relative_pos: tuple[float, float]  # attachment point in stroke a's local frame
    b_relative_pos: tuple[float, float]  # attachment point in stroke b's local frame


@dataclass(frozen=True)
class Assembly:
    strokes: tuple[Stroke, ...]
    welds: tuple[Weld, ...]
    # k-fold rotational symmetry; goal_angle sampled from [−π/k, π/k]
    symmetry_order: int = 1
