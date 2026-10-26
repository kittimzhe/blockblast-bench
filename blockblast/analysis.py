"""Offline analysis of finished runs: decision quality and failure modes.

Everything is recomputed from saved transcripts — the env is deterministic,
so每一步的局面都可以精确重放，再拿 greedy 单步最优作对照。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .env import (
    BlockBlastEnv,
    board_is_empty,
    can_place,
    clear_lines,
    full_lines,
    place_cells,
)
from .scoring import ScoringConfig, compute_move_points

# invalid-attempt classification patterns
_INVALID_KINDS = [
    ("used_slot", re.compile(r"tray slot .* (empty|already used)|no piece in slot")),
    ("out_of_bounds_or_overlap", re.compile(r"out of bounds or overlapping")),
    ("bad_piece_index", re.compile(r"piece index|slot index")),
    ("parse_error", re.compile(r"cannot parse JSON|missing/invalid")),
]


def classify_invalid(message: str) -> str:
    for kind, pat in _INVALID_KINDS:
        if pat.search(message):
            return kind
    return "other"


@dataclass
class MoveAssessment:
    move: int
    actual_points: int
    best_points: int
    cleared: bool
    best_clears: bool


@dataclass
class EpisodeAnalysis:
    seed: int
    score: int
    moves: int
    reason: str
    invalid_kinds: Counter = field(default_factory=Counter)
    assessments: List[MoveAssessment] = field(default_factory=list)
    final_fill: float = 0.0
    final_regions: int = 0

    @property
    def greedy_efficiency(self) -> float:
        """Actual points per move vs greedy-optimal points per move."""
        act = sum(a.actual_points for a in self.assessments)
        best = sum(a.best_points for a in self.assessments)
        return round(act / best, 3) if best else 0.0

    @property
    def missed_clears(self) -> int:
        """Moves where a clear was available but the agent didn't clear."""
        return sum(1 for a in self.assessments if a.best_clears and not a.cleared)


def _best_single_move(env: BlockBlastEnv, scoring: ScoringConfig) -> Tuple[int, bool]:
    """Best immediate points over all legal actions (pure simulation)."""
    obs = env.observation()
    board = [list(row) for row in obs.board]
    combo_next = obs.combo + 1  # a clearing move would carry this combo
    best_points, best_clears = 0, False
    for shape in obs.tray:
        if shape is None:
            continue
        for row in range(8 - shape.rows + 1):
            for col in range(8 - shape.cols + 1):
                if not can_place(board, shape, row, col):
                    continue
                placed = place_cells(board, shape, row, col)
                rows, cols = full_lines(placed)
                lines = len(rows) + len(cols)
                perfect = lines > 0 and board_is_empty(clear_lines(placed, rows, cols))
                points = compute_move_points(
                    cells=shape.size,
                    lines=lines,
                    combo=combo_next if lines else 0,
                    perfect_clear=perfect,
                    cfg=scoring,
                )
                if points > best_points:
                    best_points, best_clears = points, lines > 0
    return best_points, best_clears


def _empty_region_count(board_rows: List[str]) -> int:
    """Number of connected empty regions on the final board (4-neighbour)."""
    n = len(board_rows)
    seen = set()
    regions = 0
    for r in range(n):
        for c in range(n):
            if board_rows[r][c] != "." or (r, c) in seen:
                continue
            regions += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < n and 0 <= nx < n and board_rows[ny][nx] == "." and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
    return regions


def analyze_episode(payload: Dict[str, Any], scoring: Optional[ScoringConfig] = None) -> EpisodeAnalysis:
    scoring = scoring or ScoringConfig()
    seed = payload["seed"]
    ana = EpisodeAnalysis(
        seed=seed,
        score=payload["score"],
        moves=payload["moves"],
        reason=payload.get("reason", "?"),
    )
    env = BlockBlastEnv(seed=seed, scoring=scoring)
    last_board: Optional[List[str]] = None
    for rec in payload.get("transcript", []):
        if rec.get("terminal_error"):
            break
        for msg in rec.get("invalid_attempts", []):
            ana.invalid_kinds[classify_invalid(msg)] += 1
        best_points, best_clears = _best_single_move(env, scoring)
        res = env.step(rec["piece"], rec["row"], rec["col"])
        ana.assessments.append(
            MoveAssessment(
                move=rec["move"],
                actual_points=res.points,
                best_points=max(best_points, res.points),
                cleared=res.lines > 0,
                best_clears=best_clears,
            )
        )
        last_board = rec.get("board_after")
    if last_board:
        cells = sum(row.count("X") for row in last_board)
        ana.final_fill = round(cells / 64.0, 3)
        ana.final_regions = _empty_region_count(last_board)
    return ana


def analyze_run(run_dir: Path, scoring: Optional[ScoringConfig] = None) -> Dict[str, Any]:
    """Analyze every episode of a saved run; returns a JSON-friendly report."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    episodes: List[EpisodeAnalysis] = []
    for seed in summary["seeds"]:
        ep_path = run_dir / "episodes" / f"seed_{seed}.json"
        if not ep_path.exists():
            continue
        payload = json.loads(ep_path.read_text(encoding="utf-8"))
        episodes.append(analyze_episode(payload, scoring))

    invalid_total: Counter = Counter()
    for ep in episodes:
        invalid_total.update(ep.invalid_kinds)
    n_assessed = sum(len(ep.assessments) for ep in episodes)
    eff = (
        round(
            sum(a.actual_points for ep in episodes for a in ep.assessments)
            / max(1, sum(a.best_points for ep in episodes for a in ep.assessments)),
            3,
        )
        if n_assessed
        else 0.0
    )
    clears_taken = sum(1 for ep in episodes for a in ep.assessments if a.cleared)
    clears_available = sum(1 for ep in episodes for a in ep.assessments if a.best_clears)
    return {
        "run_id": summary["run_id"],
        "agent": summary["agent"],
        "model": summary.get("model"),
        "episodes": [
            {
                "seed": ep.seed,
                "score": ep.score,
                "moves": ep.moves,
                "reason": ep.reason,
                "greedy_efficiency": ep.greedy_efficiency,
                "missed_clears": ep.missed_clears,
                "final_fill": ep.final_fill,
                "final_empty_regions": ep.final_regions,
                "invalid_kinds": dict(ep.invalid_kinds),
            }
            for ep in episodes
        ],
        "aggregate": {
            "greedy_efficiency": eff,
            "moves_assessed": n_assessed,
            "clears_taken": clears_taken,
            "clears_available": clears_available,
            "clear_conversion": round(clears_taken / clears_available, 3) if clears_available else None,
            "missed_clears": sum(ep.missed_clears for ep in episodes),
            "invalid_kinds": dict(invalid_total),
            "mean_final_fill": round(
                sum(ep.final_fill for ep in episodes) / len(episodes), 3
            )
            if episodes
            else 0.0,
            "mean_final_empty_regions": round(
                sum(ep.final_regions for ep in episodes) / len(episodes), 1
            )
            if episodes
            else 0.0,
        },
    }


def render_analysis(report: Dict[str, Any]) -> str:
    agg = report["aggregate"]
    lines = [
        f"# analysis: {report['run_id']}",
        f"agent={report['agent']}" + (f" model={report['model']}" if report.get("model") else ""),
        "",
        f"{'seed':>6} {'score':>7} {'moves':>6} {'eff':>6} {'missed':>7} {'fill':>6} {'regions':>8}  reason",
    ]
    for ep in report["episodes"]:
        lines.append(
            f"{ep['seed']:>6} {ep['score']:>7} {ep['moves']:>6} "
            f"{ep['greedy_efficiency']:>6.3f} {ep['missed_clears']:>7} "
            f"{ep['final_fill']:>6.0%} {ep['final_empty_regions']:>8}  {ep['reason']}"
        )
    lines += [
        "",
        f"greedy efficiency : {agg['greedy_efficiency']:.3f}  (actual points / single-step-optimal points)",
        f"clear conversion  : {agg['clears_taken']}/{agg['clears_available']}"
        + (f" = {agg['clear_conversion']:.0%}" if agg["clear_conversion"] is not None else "")
        + "  (clears taken when one was available)",
        f"missed clears     : {agg['missed_clears']}",
        f"final board       : fill {agg['mean_final_fill']:.0%}, {agg['mean_final_empty_regions']} empty regions (fragmentation)",
    ]
    if agg["invalid_kinds"]:
        lines.append(f"invalid moves     : {agg['invalid_kinds']}")
    return "\n".join(lines)
