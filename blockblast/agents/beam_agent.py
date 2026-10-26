"""Beam-search baseline: exact lookahead within the current tray cycle.

The tray refills only after all 3 pieces are used, and piece draws come
from a seeded RNG that does not advance mid-cycle — so searching all
placements of the *remaining* tray pieces is an exact lookahead horizon.

At each move the agent runs a beam search over sequences of placements:

* state  = (board, used tray slots, cumulative points, combo, contacts)
* value  = cumulative points + contact bonus along the path
           + board-shape features at the leaf (fragmentation, tiny dead
           regions, bumpiness — same features as :class:`HeuristicAgent`)
* prune  to ``beam_width`` best states per level; the first action of the
  best final state is played.

No RNG: ties are broken deterministically by action tuple ordering.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..env import (
    Observation,
    board_is_empty,
    can_place,
    clear_lines,
    full_lines,
    place_cells,
)
from ..pieces import Shape
from ..scoring import ScoringConfig, compute_move_points
from .base import Action, Agent
from .heuristic_agent import _contact_count, _empty_region_sizes, _fill_spread


class _Node:
    """One beam-search state: a prefix of placements in the current tray."""

    __slots__ = ("board", "used", "points", "combo", "contacts", "first", "value")

    def __init__(
        self,
        board: Tuple[Tuple[bool, ...], ...],
        used: Tuple[bool, bool, bool],
        points: float,
        combo: int,
        contacts: float,
        first: Optional[Tuple[int, int, int]],
        value: float,
    ) -> None:
        self.board = board
        self.used = used
        self.points = points
        self.combo = combo
        self.contacts = contacts
        self.first = first
        self.value = value


class BeamAgent(Agent):
    name = "beam"

    def __init__(
        self,
        scoring: Optional[ScoringConfig] = None,
        beam_width: int = 40,
        w_contact: float = 1.5,
        w_fragmentation: float = -4.0,
        w_tiny_regions: float = -2.0,
        w_bumpiness: float = -0.8,
    ) -> None:
        self._scoring = scoring or ScoringConfig()
        self.beam_width = max(1, int(beam_width))
        self.w_contact = w_contact
        self.w_fragmentation = w_fragmentation
        self.w_tiny_regions = w_tiny_regions
        self.w_bumpiness = w_bumpiness
        self._moves = 0

    def reset(self, seed: int) -> None:
        self._moves = 0

    def _leaf_value(self, node: _Node) -> float:
        regions = _empty_region_sizes(node.board)
        fragmentation = len(regions)
        tiny = sum(s for s in regions if s <= 3)
        bump = _fill_spread(node.board)
        return (
            node.points
            + self.w_contact * node.contacts
            + self.w_fragmentation * fragmentation
            + self.w_tiny_regions * tiny
            + self.w_bumpiness * bump
        )

    def _expand(self, node: _Node, tray: Sequence[Optional[Shape]]) -> List[_Node]:
        children: List[_Node] = []
        board = node.board
        n = len(board)
        for slot, shape in enumerate(tray):
            if shape is None or node.used[slot]:
                continue
            for row in range(n - shape.rows + 1):
                for col in range(n - shape.cols + 1):
                    if not can_place(board, shape, row, col):
                        continue
                    cells = [(row + dr, col + dc) for dr, dc in shape.cells]
                    placed = place_cells(board, shape, row, col)
                    rows, cols = full_lines(placed)
                    lines = len(rows) + len(cols)
                    if lines:
                        placed = clear_lines(placed, rows, cols)
                    combo = node.combo + 1 if lines else 0
                    perfect = lines > 0 and board_is_empty(placed)
                    pts = compute_move_points(
                        cells=shape.size,
                        lines=lines,
                        combo=combo if lines else 0,
                        perfect_clear=perfect,
                        cfg=self._scoring,
                    )
                    used = list(node.used)
                    used[slot] = True
                    children.append(_Node(
                        board=tuple(tuple(r) for r in placed),
                        used=tuple(used),
                        points=node.points + pts,
                        combo=combo,
                        contacts=node.contacts + _contact_count(board, cells),
                        first=node.first or (slot, row, col),
                        value=0.0,
                    ))
        return children

    def act(self, obs: Observation, feedback: Optional[str] = None) -> Action:
        root = _Node(
            board=obs.board,
            used=tuple(s is None for s in obs.tray),
            points=0.0,
            combo=obs.combo,
            contacts=0.0,
            first=None,
            value=0.0,
        )
        beam = [root]
        remaining = sum(1 for used, shape in zip(root.used, obs.tray) if not used)
        for _depth in range(remaining):
            children: List[_Node] = []
            for node in beam:
                children.extend(self._expand(node, obs.tray))
            if not children:  # cycle cannot complete: keep the best prefix states
                break
            for c in children:
                c.value = self._leaf_value(c)
            # deterministic ordering: value desc, then lexicographic first action
            children.sort(key=lambda c: (-c.value, c.first or (0, 0, 0)))
            beam = children[: self.beam_width]
        best = beam[0]  # beam is kept sorted best-first
        assert best.first is not None, "act() called on a terminal observation"
        self._moves += 1
        return Action(*best.first)

    @property
    def stats(self) -> Dict[str, int]:
        return {"moves": self._moves}
