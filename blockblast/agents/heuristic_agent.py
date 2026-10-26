"""Heuristic baseline: greedy clears plus board-shape features.

The evaluation of the board *after* a candidate move combines:

* immediate points (clears + combo),
* contact: how flush the placement is against walls / occupied cells,
* fragmentation: number of connected empty regions (fewer is better),
* tiny regions: empty cells stuck in regions of size <= 3 (dead space),
* bumpiness: spread of per-row and per-column fill counts (flat is better).

Weights are hand-picked; this is a baseline, not a tuned solver.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..env import BOARD_SIZE, Observation, board_is_empty, clear_lines, full_lines, place_cells
from ..scoring import ScoringConfig, compute_move_points
from .base import Action, Agent

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _empty_region_sizes(board: Sequence[Sequence[bool]]) -> List[int]:
    seen = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    sizes: List[int] = []
    for r0 in range(BOARD_SIZE):
        for c0 in range(BOARD_SIZE):
            if board[r0][c0] or seen[r0][c0]:
                continue
            size = 0
            stack = [(r0, c0)]
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in _NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if not board[nr][nc] and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            sizes.append(size)
    return sizes


def _contact_count(board_before: Sequence[Sequence[bool]], cells: Sequence[Tuple[int, int]]) -> int:
    """Orthogonal neighbours of the placed cells that are walls or already filled."""
    contact = 0
    for r, c in cells:
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                contact += 1
            elif board_before[nr][nc]:
                contact += 1
    return contact


def _fill_spread(board: Sequence[Sequence[bool]]) -> float:
    row_fill = [sum(1 for v in row if v) for row in board]
    col_fill = [sum(1 for r in range(BOARD_SIZE) if board[r][c]) for c in range(BOARD_SIZE)]

    def spread(values: Sequence[float]) -> float:
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    return spread(row_fill) + spread(col_fill)


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(
        self,
        scoring: Optional[ScoringConfig] = None,
        w_points: float = 1.0,
        w_contact: float = 1.5,
        w_fragmentation: float = -4.0,
        w_tiny_regions: float = -2.0,
        w_bumpiness: float = -0.8,
    ) -> None:
        self._scoring = scoring or ScoringConfig()
        self.w_points = w_points
        self.w_contact = w_contact
        self.w_fragmentation = w_fragmentation
        self.w_tiny_regions = w_tiny_regions
        self.w_bumpiness = w_bumpiness
        self._rng = random.Random(0)
        self._moves = 0

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed ^ 0xC0FFEE)
        self._moves = 0

    def _evaluate(
        self,
        obs: Observation,
        piece: int,
        row: int,
        col: int,
    ) -> float:
        shape = obs.tray[piece]
        cells = [(row + dr, col + dc) for dr, dc in shape.cells]
        board = place_cells(obs.board, shape, row, col)
        rows, cols = full_lines(board)
        lines = len(rows) + len(cols)
        combo = obs.combo + 1 if lines else 0
        perfect = board_is_empty(board)
        points = compute_move_points(
            cells=shape.size, lines=lines, combo=combo, perfect_clear=perfect, cfg=self._scoring
        )
        if lines:
            board = clear_lines(board, rows, cols)
        regions = _empty_region_sizes(board)
        fragmentation = len(regions)
        tiny_cells = sum(s for s in regions if s <= 3)
        contact = _contact_count(obs.board, cells)
        bump = _fill_spread(board)
        return (
            self.w_points * points
            + self.w_contact * contact
            + self.w_fragmentation * fragmentation
            + self.w_tiny_regions * tiny_cells
            + self.w_bumpiness * bump
        )

    def act(self, obs: Observation, feedback: Optional[str] = None) -> Action:
        best_action: Optional[Action] = None
        best_value = None
        for piece, row, col in obs.legal_actions():
            value = (self._evaluate(obs, piece, row, col), self._rng.random())
            if best_value is None or value > best_value:
                best_value = value
                best_action = Action(piece, row, col)
        assert best_action is not None, "act() called on a terminal observation"
        self._moves += 1
        return best_action

    @property
    def stats(self) -> Dict[str, int]:
        return {"moves": self._moves}
