"""Render an episode transcript into an animated GIF.

Frames are drawn as plain index buffers (no imaging dependency):
board cells on an 8x8 grid with a small HUD strip at the bottom.
Each move produces up to three frames:

1. placement  — the new piece highlighted,
2. flash      — cells of full rows/columns highlighted (only when clearing),
3. settle     — the resulting board.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .gifenc import write_gif

# palette indices
BG = 0        # page background
GRID = 1      # grid lines
EMPTY = 2     # empty cell
FILLED = 3    # settled block
NEW = 4       # just-placed piece
FLASH = 5     # clearing cells
HUD = 6       # HUD text pixels
COMBO = 7     # combo accent

PALETTE: List[Tuple[int, int, int]] = [
    (24, 26, 38),     # BG dark navy
    (46, 50, 68),     # GRID
    (34, 37, 52),     # EMPTY
    (86, 156, 214),   # FILLED steel blue
    (255, 196, 0),    # NEW amber
    (255, 255, 255),  # FLASH white
    (164, 170, 190),  # HUD grey
    (255, 99, 132),   # COMBO red/pink
]

CELL = 22      # cell size in px
PAD = 2        # gap between cells
MARGIN = 12    # outer margin
HUD_H = 18     # HUD strip height

# 3x5 pixel font for digits and a few glyphs
_FONT: Dict[str, Tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "x": ("000", "101", "010", "101", "000"),
    "#": ("010", "111", "010", "111", "010"),
    " ": ("000", "000", "000", "000", "000"),
}


def _canvas_size(board_size: int = 8) -> Tuple[int, int]:
    side = MARGIN * 2 + board_size * CELL + (board_size - 1) * PAD
    return side, side + HUD_H


def _new_canvas(width: int, height: int) -> List[int]:
    return [BG] * (width * height)


def _fill_rect(buf: List[int], width: int, x0: int, y0: int, w: int, h: int, colour: int) -> None:
    for y in range(y0, y0 + h):
        row = y * width
        for x in range(x0, x0 + w):
            buf[row + x] = colour


def _draw_text(buf: List[int], width: int, x: int, y: int, text: str, colour: int, scale: int = 2) -> None:
    cx = x
    for ch in text:
        glyph = _FONT.get(ch, _FONT[" "])
        for gy, line in enumerate(glyph):
            for gx, bit in enumerate(line):
                if bit == "1":
                    _fill_rect(buf, width, cx + gx * scale, y + gy * scale, scale, scale, colour)
        cx += (3 + 1) * scale


def _cell_origin(r: int, c: int) -> Tuple[int, int]:
    return (MARGIN + c * (CELL + PAD), MARGIN + r * (CELL + PAD))


def _draw_board(
    buf: List[int],
    width: int,
    board: Sequence[str],
    highlight: Sequence[Tuple[int, int]] = (),
    flash: Sequence[Tuple[int, int]] = (),
    hud: str = "",
    hud_colour: int = HUD,
) -> None:
    n = len(board)
    highlight_set = set(highlight)
    flash_set = set(flash)
    for r in range(n):
        for c in range(n):
            x, y = _cell_origin(r, c)
            if (r, c) in flash_set:
                colour = FLASH
            elif (r, c) in highlight_set:
                colour = NEW
            elif board[r][c] == "X":
                colour = FILLED
            else:
                colour = EMPTY
            _fill_rect(buf, width, x, y, CELL, CELL, colour)
    if hud:
        side = MARGIN * 2 + n * CELL + (n - 1) * PAD
        _draw_text(buf, width, MARGIN, side + 4, hud, hud_colour)


def _piece_cells(rec: dict, prev_board: Sequence[str]) -> List[Tuple[int, int]]:
    """Cells the move added = occupied after placing minus occupied before.

    Reconstructed from the transcript alone: board_after already has lines
    cleared, so we re-derive the pre-clear board from prev + known offsets.
    """
    cells = []
    row0, col0 = rec["row"], rec["col"]
    # we do not store shape cells in the transcript; infer from piece name via catalog
    from .pieces import SHAPE_BY_NAME

    shape = SHAPE_BY_NAME[rec["piece_name"]]
    for dr, dc in shape.cells:
        cells.append((row0 + dr, col0 + dc))
    return cells


def _clear_cells(rec: dict, board_size: int = 8) -> List[Tuple[int, int]]:
    cells = []
    for r in rec.get("cleared_rows", []):
        cells.extend((r, c) for c in range(board_size))
    for c in rec.get("cleared_cols", []):
        cells.extend((r, c) for r in range(board_size))
    return cells


def _apply_place(board: List[str], cells: Sequence[Tuple[int, int]]) -> List[str]:
    grid = [list(row) for row in board]
    for r, c in cells:
        grid[r][c] = "X"
    return ["".join(row) for row in grid]


def episode_frames(
    transcript: Sequence[dict],
    board_size: int = 8,
    place_delay: int = 18,
    flash_delay: int = 12,
    settle_delay: int = 10,
) -> Tuple[List[List[int]], List[int], Tuple[int, int]]:
    """Build GIF frames + delays (in 1/100 s) from a transcript."""
    width, height = _canvas_size(board_size)
    frames: List[List[int]] = []
    delays: List[int] = []
    board = ["." * board_size for _ in range(board_size)]

    def emit(b, highlight=(), flash=(), hud="", hud_colour=HUD, delay=settle_delay):
        buf = _new_canvas(width, height)
        _draw_board(buf, width, b, highlight, flash, hud, hud_colour)
        frames.append(buf)
        delays.append(delay)

    emit(board, hud="#0 0", delay=60)

    for rec in transcript:
        if rec.get("terminal_error"):
            break
        cells = _piece_cells(rec, board)
        placed = _apply_place(board, cells)
        hud = f"#{rec['move']} {rec['score']}"
        emit(placed, highlight=cells, hud=hud, delay=place_delay)
        if rec.get("lines"):
            combo_hud = f"#{rec['move']} {rec['score']} x{rec['combo']}"
            emit(placed, flash=_clear_cells(rec, board_size), hud=combo_hud,
                 hud_colour=COMBO, delay=flash_delay)
        board = rec["board_after"]
        emit(board, hud=hud, delay=settle_delay)

    # hold the final position
    delays[-1] = 150
    return frames, delays, (width, height)


def transcript_to_gif(
    path,
    transcript: Sequence[dict],
    board_size: int = 8,
    speed: float = 1.0,
) -> Tuple[int, Tuple[int, int]]:
    """Render a transcript to an animated GIF; returns (n_frames, (w, h))."""
    frames, delays, size = episode_frames(transcript, board_size)
    if speed != 1.0:
        delays = [max(2, int(d / speed)) for d in delays]
    write_gif(path, size, PALETTE, frames, delays, loop=0)
    return len(frames), size
