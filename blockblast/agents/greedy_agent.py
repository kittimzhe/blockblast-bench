"""Greedy baseline: maximises the points of the current move only."""

from __future__ import annotations

import random
from typing import Optional

from ..env import Observation, board_is_empty, full_lines, place_cells
from ..scoring import ScoringConfig, compute_move_points
from .base import Action, Agent


class GreedyAgent(Agent):
    """One-step lookahead: choose the placement with the highest immediate score.

    Tie-breaks prefer placing larger pieces (frees tray space), then random.
    """

    name = "greedy"

    def __init__(self, scoring: Optional[ScoringConfig] = None) -> None:
        self._scoring = scoring or ScoringConfig()
        self._rng = random.Random(0)
        self._moves = 0

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed ^ 0x9E37)
        self._moves = 0

    def act(self, obs: Observation, feedback: Optional[str] = None) -> Action:
        best_action: Optional[Action] = None
        best_key = None
        for piece, row, col in obs.legal_actions():
            shape = obs.tray[piece]
            board = place_cells(obs.board, shape, row, col)
            rows, cols = full_lines(board)
            lines = len(rows) + len(cols)
            combo = obs.combo + 1 if lines else 0
            perfect = board_is_empty(board)
            points = compute_move_points(
                cells=shape.size,
                lines=lines,
                combo=combo,
                perfect_clear=perfect,
                cfg=self._scoring,
            )
            key = (points, shape.size, self._rng.random())
            if best_key is None or key > best_key:
                best_key = key
                best_action = Action(piece, row, col)
        assert best_action is not None, "act() called on a terminal observation"
        self._moves += 1
        return best_action

    @property
    def stats(self):
        return {"moves": self._moves}
