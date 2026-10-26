"""Text rendering of game state, tuned for both humans and LLMs.

The board render keeps explicit row/column indices and uses '.' / 'X' so a
model (or test) can parse the grid with trivial string handling.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .env import Observation, StepResult
from .pieces import Shape

_BOARD_MARKS = {False: ".", True: "X"}


def render_board(board: Sequence[Sequence[bool]], numbered: bool = True) -> str:
    """Render an 8x8 board. Rows 0-7 top->bottom, cols 0-7 left->right."""
    lines = []
    if numbered:
        header = "    " + " ".join(str(c) for c in range(len(board[0])))
        lines.append(header)
    for r, row in enumerate(board):
        body = " ".join(_BOARD_MARKS[bool(v)] for v in row)
        lines.append(f"{r}   {body}" if numbered else body)
    return "\n".join(lines)


def render_shape(shape: Shape, mark: str = "X") -> str:
    """Mini-grid preview of one shape (indented by 4 spaces)."""
    grid = [[mark if (r, c) in shape.cells else "." for c in range(shape.cols)] for r in range(shape.rows)]
    return "\n".join("    " + " ".join(row) for row in grid)


def render_tray(tray: Sequence[Optional[Shape]]) -> str:
    lines = []
    for idx, shape in enumerate(tray):
        if shape is None:
            lines.append(f"[{idx}] (used)")
            continue
        lines.append(f"[{idx}] {shape.name}  ({shape.rows}x{shape.cols} bounding box, {shape.size} cells)")
        lines.append(render_shape(shape))
    return "\n".join(lines)


def render_observation(obs: Observation) -> str:
    """Full observation as text: board + tray + stats."""
    combo_note = f"x{obs.combo}" if obs.combo > 1 else "0"
    parts = [
        "BOARD (8x8, '.'=empty 'X'=occupied, rows/cols numbered 0-7):",
        render_board(obs.board),
        "",
        "TRAY (place one piece; choose the tray index and the (row, col) of its top-left cell):",
        render_tray(obs.tray),
        "",
        f"score={obs.score}  combo={combo_note}  moves={obs.moves}  "
        f"lines_cleared={obs.lines_cleared}  pieces_in_tray={obs.pieces_in_tray}",
    ]
    return "\n".join(parts)


def render_step(res: StepResult) -> str:
    """One-line human summary of a placement."""
    piece, row, col = res.action
    bits = [
        f"move {res.observation.moves}: placed [{piece}] {res.piece_name} at (row={row}, col={col})",
        f"+{res.cells} cells",
    ]
    if res.lines:
        where = []
        if res.cleared_rows:
            where.append("rows " + ",".join(map(str, res.cleared_rows)))
        if res.cleared_cols:
            where.append("cols " + ",".join(map(str, res.cleared_cols)))
        bits.append(f"cleared {res.lines} line(s) ({'; '.join(where)}) combo x{res.combo}")
    if res.perfect_clear:
        bits.append("PERFECT CLEAR")
    bits.append(f"+{res.points} pts -> score {res.observation.score}")
    return " | ".join(bits)


def board_rows_as_strings(board: Sequence[Sequence[bool]]) -> Sequence[str]:
    """Compact board serialisation for transcripts ('.'/'X' rows)."""
    return ["".join(_BOARD_MARKS[bool(v)] for v in row) for row in board]
