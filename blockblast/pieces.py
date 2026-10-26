"""Piece (shape) library for Block Blast.

Mirrors the piece set popularised by 1010!/Block Blast!:

* 1x1 dot
* horizontal & vertical lines of length 2..5   (8 shapes)
* 2x2 square, 3x3 square                        (2 shapes)
* small L: a 2x2 box minus one corner (3 cells)  (4 orientations)
* big L: two length-3 arms from a 3x3 corner (5 cells)  (4 orientations)

Total: 19 distinct shapes. Pieces are never rotated or flipped by the player.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

Cell = Tuple[int, int]  # (row_offset, col_offset), normalised to top-left origin


@dataclass(frozen=True)
class Shape:
    """An immutable polyomino-style piece.

    ``cells`` are (row, col) offsets normalised so that min(row) == min(col) == 0,
    which makes (row, col) placement coordinates unambiguous: placing the shape
    at (r, c) puts the top-left cell of its bounding box at board[r][c].
    """

    name: str
    cells: Tuple[Cell, ...]

    @property
    def rows(self) -> int:
        return 1 + max(r for r, _ in self.cells)

    @property
    def cols(self) -> int:
        return 1 + max(c for _, c in self.cells)

    @property
    def size(self) -> int:
        return len(self.cells)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.name} ({self.rows}x{self.cols}, {self.size} cells)"


def _make(name: str, cells: Iterable[Tuple[int, int]]) -> Shape:
    cell_set = sorted({(int(r), int(c)) for r, c in cells})
    if not cell_set:
        raise ValueError(f"shape {name!r} has no cells")
    min_r = min(r for r, _ in cell_set)
    min_c = min(c for _, c in cell_set)
    normalised = tuple(sorted((r - min_r, c - min_c) for r, c in cell_set))
    return Shape(name=name, cells=normalised)


def _big_L_variants() -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Two length-3 arms sharing a corner of a 3x3 bounding box (5 cells)."""
    return [
        ("L5-A", [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]),  # corner bottom-left
        ("L5-B", [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]),  # corner top-left
        ("L5-C", [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]),  # corner top-right
        ("L5-D", [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]),  # corner bottom-right
    ]


def build_catalog() -> Tuple[Shape, ...]:
    shapes: List[Shape] = [_make("1x1", [(0, 0)])]
    for n in (2, 3, 4, 5):
        shapes.append(_make(f"{n}-line-h", [(0, c) for c in range(n)]))
    for n in (2, 3, 4, 5):
        shapes.append(_make(f"{n}-line-v", [(r, 0) for r in range(n)]))
    shapes.append(_make("2x2", [(r, c) for r in range(2) for c in range(2)]))
    shapes.append(_make("3x3", [(r, c) for r in range(3) for c in range(3)]))
    # Small L: 2x2 box minus one corner (3 cells), four orientations.
    for i, (mr, mc) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        cells = [(r, c) for r in range(2) for c in range(2) if (r, c) != (mr, mc)]
        shapes.append(_make(f"L3-{chr(65 + i)}", cells))
    for name, cells in _big_L_variants():
        shapes.append(_make(name, cells))
    # sanity: no duplicate shapes
    seen = set(s.cells for s in shapes)
    if len(seen) != len(shapes):
        raise RuntimeError("duplicate shape detected in catalog")
    return tuple(shapes)


#: The fixed piece catalog (19 shapes). Order is stable -> deterministic sampling.
SHAPE_CATALOG: Tuple[Shape, ...] = build_catalog()

#: Quick lookup by name.
SHAPE_BY_NAME = {s.name: s for s in SHAPE_CATALOG}


def default_piece_weights() -> List[float]:
    """Uniform weights over the catalog (the benchmark default).

    The real game's piece generator is proprietary and reportedly adaptive;
    uniform sampling keeps runs reproducible and comparable across agents.
    """
    return [1.0] * len(SHAPE_CATALOG)
