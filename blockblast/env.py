"""Block Blast game environment.

Deterministic, seed-driven 8x8 block puzzle faithful to the mobile game:

* The tray holds 3 pieces; all 3 must be used before the tray refills.
* Pieces cannot be rotated or flipped.
* A fully occupied row or column is cleared (multiple at once possible).
* Consecutive clearing moves build a combo streak; the streak resets on any
  non-clearing placement.
* The game ends when no remaining tray piece fits anywhere on the board.

The environment exposes pure board helpers (:func:`can_place`,
:func:`legal_actions_for`, ...) so agents can simulate placements cheaply
without touching live state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .pieces import SHAPE_CATALOG, Shape
from .scoring import ScoringConfig, compute_move_points

#: An action is (tray_index, row, col) of the piece's top-left cell.
Action = Tuple[int, int, int]

BOARD_SIZE = 8
TRAY_SIZE = 3

Board = Sequence[Sequence[bool]]


class InvalidActionError(ValueError):
    """Raised when a submitted move is illegal in the current state."""


# ---------------------------------------------------------------------------
# Pure board helpers (shared by env + agents for simulation)
# ---------------------------------------------------------------------------

def can_place(board: Board, shape: Shape, row: int, col: int) -> bool:
    n_rows = len(board)
    n_cols = len(board[0])
    if row < 0 or col < 0:
        return False
    if row + shape.rows > n_rows or col + shape.cols > n_cols:
        return False
    for dr, dc in shape.cells:
        if board[row + dr][col + dc]:
            return False
    return True


def legal_actions_for(board: Board, tray: Sequence[Optional[Shape]]) -> List[Action]:
    """All legal (tray_index, row, col) moves for the remaining tray pieces."""
    actions: List[Action] = []
    n_rows = len(board)
    n_cols = len(board[0])
    for idx, shape in enumerate(tray):
        if shape is None:
            continue
        for r in range(n_rows - shape.rows + 1):
            for c in range(n_cols - shape.cols + 1):
                if can_place(board, shape, r, c):
                    actions.append((idx, r, c))
    return actions


def place_cells(board: Board, shape: Shape, row: int, col: int) -> List[List[bool]]:
    """Return a *new* board with the shape placed (no clearing)."""
    new_board = [list(r) for r in board]
    for dr, dc in shape.cells:
        new_board[row + dr][col + dc] = True
    return new_board


def full_lines(board: Board) -> Tuple[List[int], List[int]]:
    """Indices of fully-occupied rows and columns."""
    n_rows = len(board)
    n_cols = len(board[0])
    rows = [r for r in range(n_rows) if all(board[r])]
    cols = [c for c in range(n_cols) if all(board[r][c] for r in range(n_rows))]
    return rows, cols


def clear_lines(board: Board, rows: Sequence[int], cols: Sequence[int]) -> List[List[bool]]:
    """Return a *new* board with the given rows/cols emptied."""
    new_board = [list(r) for r in board]
    n_rows = len(board)
    n_cols = len(board[0])
    for r in rows:
        for c in range(n_cols):
            new_board[r][c] = False
    for c in cols:
        for r in range(n_rows):
            new_board[r][c] = False
    return new_board


def board_is_empty(board: Board) -> bool:
    return not any(any(row) for row in board)


def _board_to_tuple(board: Board) -> Tuple[Tuple[bool, ...], ...]:
    return tuple(tuple(bool(v) for v in row) for row in board)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """An immutable snapshot of the state an agent is allowed to see."""

    board: Tuple[Tuple[bool, ...], ...]
    tray: Tuple[Optional[Shape], ...]
    score: int
    combo: int
    moves: int
    lines_cleared: int
    max_combo: int
    perfect_clears: int

    @property
    def pieces_in_tray(self) -> int:
        return sum(1 for s in self.tray if s is not None)

    def legal_actions(self) -> List[Action]:
        return legal_actions_for(self.board, self.tray)

    def is_game_over(self) -> bool:
        return len(self.legal_actions()) == 0


@dataclass(frozen=True)
class StepResult:
    """Result of one successful placement."""

    action: Action
    piece_name: str
    cells: int
    points: int
    lines: int
    cleared_rows: Tuple[int, ...]
    cleared_cols: Tuple[int, ...]
    combo: int
    perfect_clear: bool
    observation: Observation
    game_over: bool


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BlockBlastEnv:
    """Deterministic Block Blast environment.

    The same ``seed`` plus the same action sequence always yields the same
    game (piece draws use :class:`random.Random(seed)`; Mersenne Twister
    streams are stable across supported Python versions).
    """

    def __init__(
        self,
        seed: int = 0,
        scoring: ScoringConfig = None,
        piece_weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.scoring = scoring or ScoringConfig()
        self._piece_weights = list(piece_weights) if piece_weights is not None else None
        self._seed = seed
        self._rng = random.Random(seed)
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset to an empty board. If ``seed`` is given, re-seed the RNG."""
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
        self._board: List[List[bool]] = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self._tray: List[Optional[Shape]] = [None] * TRAY_SIZE
        self.score = 0
        self.combo = 0
        self.moves = 0
        self.lines_cleared = 0
        self.max_combo = 0
        self.perfect_clears = 0
        self._refill()
        return self.observation()

    # -- state access ------------------------------------------------------

    def observation(self) -> Observation:
        return Observation(
            board=_board_to_tuple(self._board),
            tray=tuple(self._tray),
            score=self.score,
            combo=self.combo,
            moves=self.moves,
            lines_cleared=self.lines_cleared,
            max_combo=self.max_combo,
            perfect_clears=self.perfect_clears,
        )

    def legal_actions(self) -> List[Action]:
        return legal_actions_for(self._board, self._tray)

    def is_game_over(self) -> bool:
        return len(self.legal_actions()) == 0

    # -- gameplay ----------------------------------------------------------

    def _draw_shape(self) -> Shape:
        if self._piece_weights is None:
            return self._rng.choice(SHAPE_CATALOG)
        return self._rng.choices(SHAPE_CATALOG, weights=self._piece_weights, k=1)[0]

    def _refill(self) -> None:
        self._tray = [self._draw_shape() for _ in range(TRAY_SIZE)]

    def step(self, *action: int) -> StepResult:
        """Apply a placement ``(piece_index, row, col)``.

        Raises :class:`InvalidActionError` for out-of-range indices, an
        already-used tray slot, or an overlapping/out-of-bounds placement.
        """
        if len(action) == 1 and isinstance(action[0], (tuple, list)):
            piece, row, col = action[0]
        elif len(action) == 3:
            piece, row, col = action
        else:
            raise InvalidActionError(f"malformed action: {action!r}")

        if not isinstance(piece, int) or not isinstance(row, int) or not isinstance(col, int):
            raise InvalidActionError(
                f"action fields must be integers, got ({piece!r}, {row!r}, {col!r})"
            )
        if not (0 <= piece < TRAY_SIZE):
            raise InvalidActionError(
                f"piece index {piece} out of range 0..{TRAY_SIZE - 1}"
            )
        shape = self._tray[piece]
        if shape is None:
            raise InvalidActionError(f"tray slot {piece} is already used")
        if not can_place(self._board, shape, row, col):
            raise InvalidActionError(
                f"cannot place {shape.name} at (row={row}, col={col}): "
                "out of bounds or overlapping occupied cells"
            )

        # place
        board = place_cells(self._board, shape, row, col)
        cells = shape.size

        # clear full lines (rows and columns simultaneously)
        rows, cols = full_lines(board)
        lines = len(rows) + len(cols)
        if lines > 0:
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            board = clear_lines(board, rows, cols)
        else:
            self.combo = 0

        perfect = board_is_empty(board)
        if perfect:
            self.perfect_clears += 1

        points = compute_move_points(
            cells=cells,
            lines=lines,
            combo=self.combo,
            perfect_clear=perfect,
            cfg=self.scoring,
        )

        self._board = board
        self.score += points
        self.moves += 1
        self.lines_cleared += lines
        self._tray[piece] = None
        if all(s is None for s in self._tray):
            self._refill()

        return StepResult(
            action=(piece, row, col),
            piece_name=shape.name,
            cells=cells,
            points=points,
            lines=lines,
            cleared_rows=tuple(rows),
            cleared_cols=tuple(cols),
            combo=self.combo,
            perfect_clear=perfect,
            observation=self.observation(),
            game_over=self.is_game_over(),
        )
