"""Persistence + leaderboard rendering.

Layout under ``results/``::

    results/
      runs.jsonl                     # one line per run (leaderboard index)
      leaderboard.md                 # regenerated after every run
      <run_id>/summary.json          # run meta + aggregates + episode rows
      <run_id>/episodes/seed_N.json  # full transcript (+ LLM dialog if saved)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .harness import RunResult, aggregates

RUNS_INDEX = "runs.jsonl"
LEADERBOARD = "leaderboard.md"

_TABLE_COLUMNS = [
    ("mean_score", "Mean"),
    ("median_score", "Median"),
    ("max_score", "Max"),
    ("mean_moves", "Moves"),
    ("mean_lines", "Lines"),
    ("best_combo", "Combo"),
    ("invalid_per_move", "Inv/mv"),
    ("tokens_per_move", "Tok/mv"),
]


def _episode_row(ep) -> Dict[str, Any]:
    return {
        "seed": ep.seed,
        "score": ep.score,
        "moves": ep.moves,
        "lines_cleared": ep.lines_cleared,
        "max_combo": ep.max_combo,
        "perfect_clears": ep.perfect_clears,
        "invalid_actions": ep.invalid_actions,
        "reason": ep.reason,
        "duration_s": ep.duration_s,
        "agent_stats": ep.agent_stats,
        "error": ep.error,
    }


def write_run(run: RunResult, results_dir: Path, save_transcripts: bool = True) -> Path:
    """Write summary, per-episode files, append to runs.jsonl, refresh leaderboard."""
    run_dir = Path(results_dir) / run.run_id
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run.run_id,
        "agent": run.agent,
        "model": run.model,
        "episodes": run.episodes,
        "seed_base": run.seed_base,
        "seeds": run.seeds,
        "scoring": run.scoring,
        "started_at": run.started_at,
        "aggregates": aggregates(run),
        "episodes_detail": [_episode_row(e) for e in run.results],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if save_transcripts:
        for ep in run.results:
            payload = {
                "run_id": run.run_id,
                "agent": run.agent,
                "model": run.model,
                "seed": ep.seed,
                "score": ep.score,
                "moves": ep.moves,
                "lines_cleared": ep.lines_cleared,
                "max_combo": ep.max_combo,
                "perfect_clears": ep.perfect_clears,
                "invalid_actions": ep.invalid_actions,
                "reason": ep.reason,
                "duration_s": ep.duration_s,
                "error": ep.error,
                "transcript": ep.transcript,
            }
            if ep.dialog:
                payload["dialog"] = ep.dialog
            (run_dir / "episodes" / f"seed_{ep.seed}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    index_row = {
        "run_id": run.run_id,
        "agent": run.agent,
        "model": run.model,
        "episodes": run.episodes,
        "seed_base": run.seed_base,
        "started_at": run.started_at,
        "aggregates": summary["aggregates"],
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / RUNS_INDEX).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(index_row, ensure_ascii=False) + "\n")

    refresh_leaderboard(results_dir)
    return run_dir


def load_runs(results_dir: Path) -> List[Dict[str, Any]]:
    index = Path(results_dir) / RUNS_INDEX
    if not index.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def leaderboard_md(runs: List[Dict[str, Any]], title: str = "Block Blast Benchmark — Leaderboard") -> str:
    rows = sorted(
        runs, key=lambda r: (r.get("aggregates", {}).get("mean_score", 0), r.get("started_at", "")), reverse=True
    )
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No runs yet — execute `blockblast run --agent ...` first._")
        return "\n".join(lines) + "\n"

    header = (
        "| # | Agent | Model | Eps | Mean | Median | Max | Moves | Lines | Combo | Inv/mv | Tok/mv | Run |"
        "\n|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    lines.append(header)
    for i, row in enumerate(rows, 1):
        agg = row.get("aggregates", {})
        model = row.get("model") or "-"
        run_id = row.get("run_id", "?")
        cells = [
            str(i),
            str(row.get("agent", "?")),
            model,
            str(agg.get("episodes", 0)),
            str(agg.get("mean_score", 0)),
            str(agg.get("median_score", 0)),
            str(agg.get("max_score", 0)),
            str(agg.get("mean_moves", 0)),
            str(agg.get("mean_lines", 0)),
            str(agg.get("best_combo", 0)),
            str(agg.get("invalid_per_move", 0)),
            str(agg.get("tokens_per_move", 0) or "-"),
            run_id,
        ]
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "_Scores are only comparable between runs sharing the same seed set and scoring config._",
    ]
    return "\n".join(lines) + "\n"


def refresh_leaderboard(results_dir: Path) -> Optional[Path]:
    results_dir = Path(results_dir)
    rows = load_runs(results_dir)
    if not rows:
        return None
    path = results_dir / LEADERBOARD
    path.write_text(leaderboard_md(rows), encoding="utf-8")
    return path
