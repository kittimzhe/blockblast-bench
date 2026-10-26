"""Scoring model for Block Blast.

The official scoring constants are not public; this module implements a
documented approximation whose *mechanics* are faithful:

* +1 point per cell placed.
* Line clears score progressively: 1 line = 10, 2 lines = 30 (10+20),
  3 lines = 60 (10+20+30), ... i.e. ``points_per_line * k`` summed for
  k = 1..lines.
* Combo multiplier: a clear that continues a clear-streak multiplies the
  line points by ``1 + combo_step * (combo - 1)``, capped at
  ``combo_cap`` (combo == 1 means the first clear of a streak -> x1.0).
* Perfect clear (board completely empty after a clear) awards a flat bonus.

All constants live in :class:`ScoringConfig` so the whole benchmark runs
under one transparent rule set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoringConfig:
    points_per_cell: int = 1
    points_per_line: int = 10
    multi_line_progressive: bool = True
    combo_step: float = 0.5
    combo_cap: float = 4.0
    perfect_clear_bonus: int = 300

    def to_dict(self) -> dict:
        return asdict(self)


def combo_multiplier(combo: int, cfg: ScoringConfig) -> float:
    """Multiplier applied to line points given the (already incremented) combo."""
    if combo <= 1:
        return 1.0
    return min(1.0 + cfg.combo_step * (combo - 1), cfg.combo_cap)


def line_points(lines: int, cfg: ScoringConfig) -> int:
    """Base points for clearing ``lines`` rows/columns simultaneously."""
    if lines <= 0:
        return 0
    if cfg.multi_line_progressive:
        return sum(cfg.points_per_line * k for k in range(1, lines + 1))
    return lines * cfg.points_per_line


def compute_move_points(
    *,
    cells: int,
    lines: int,
    combo: int,
    perfect_clear: bool,
    cfg: ScoringConfig = None,
) -> int:
    """Total points awarded for one placement.

    ``combo`` is the streak value *after* this move's increment (>=1 iff
    this move cleared at least one line, else the streak is broken).
    """
    if cfg is None:
        cfg = ScoringConfig()
    total = cells * cfg.points_per_cell
    if lines > 0:
        total += int(round(line_points(lines, cfg) * combo_multiplier(combo, cfg)))
    if perfect_clear:
        total += cfg.perfect_clear_bonus
    return total
