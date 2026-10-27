#!/usr/bin/env python3
"""Large-sample baseline study: how much search budget actually buys.

Why this exists: a first pass on the default 10-episode seed set produced a
*non-monotonic* beam-width curve (width 20 scoring below width 10), which
looked like a real "high-variance regime". Re-running on 50 episodes showed
the curve is in fact cleanly monotonic — the dip was sampling noise.

Episode scores in this game are heavy-tailed (one early move rewrites the
whole game), so this script reports the **standard error** alongside every
mean. That is the number that tells you whether a ranking difference means
anything at all.

Usage:
    python3 examples/baseline_study.py                # 50 episodes
    python3 examples/baseline_study.py --episodes 100 # tighter error bars
    python3 examples/baseline_study.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blockblast.agents import BeamAgent, GreedyAgent, HeuristicAgent, RandomAgent
from blockblast.agents.base import Agent
from blockblast.harness import EvalConfig, run_episode

LADDER: List[tuple] = [
    ("random", lambda: RandomAgent(seed=0)),
    ("greedy", lambda: GreedyAgent()),
    ("heuristic", lambda: HeuristicAgent()),
    ("beam-10", lambda: BeamAgent(beam_width=10)),
    ("beam-20", lambda: BeamAgent(beam_width=20)),
    ("beam-40", lambda: BeamAgent(beam_width=40)),
    ("beam-80", lambda: BeamAgent(beam_width=80)),
]


def study_agent(
    name: str, factory: Callable[[], Agent], episodes: int, seed_base: int
) -> Dict[str, float]:
    cfg = EvalConfig()
    started = time.time()
    scores: List[int] = []
    moves: List[int] = []
    for i in range(episodes):
        result = run_episode(factory(), seed=seed_base + i, cfg=cfg)
        scores.append(result.score)
        moves.append(result.moves)

    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return {
        "agent": name,
        "episodes": episodes,
        "mean": round(statistics.mean(scores), 1),
        "median": round(statistics.median(scores), 1),
        "stdev": round(stdev, 1),
        # standard error of the mean: the honest error bar on `mean`
        "sem": round(stdev / (episodes ** 0.5), 1),
        "max": max(scores),
        "mean_moves": round(statistics.mean(moves), 1),
        "wall_s": round(time.time() - started, 1),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=2024)
    ap.add_argument("--json", type=Path, help="also write the rows as JSON")
    args = ap.parse_args(argv)

    last = args.seed_base + args.episodes - 1
    print(f"# baseline study: {args.episodes} episodes (seeds {args.seed_base}-{last})")
    header = f"{'agent':>12} {'mean':>8} {'median':>8} {'stdev':>8} {'±sem':>7} {'max':>6} {'moves':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, factory in LADDER:
        row = study_agent(name, factory, args.episodes, args.seed_base)
        rows.append(row)
        print(
            f"{row['agent']:>12} {row['mean']:>8.1f} {row['median']:>8.1f} "
            f"{row['stdev']:>8.1f} {row['sem']:>7.1f} {row['max']:>6} "
            f"{row['mean_moves']:>7.1f}   [{row['wall_s']:.0f}s]",
            flush=True,
        )

    print(
        "\nRead the ±sem column before trusting any ranking: with heavy-tailed\n"
        "episode scores, gaps smaller than ~2x the standard error are noise."
    )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
