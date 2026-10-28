#!/usr/bin/env python3
"""Paired A/B analysis for two runs over the same seed set.

Why paired: comparing two 10-episode means head-on wastes the benchmark's
best property. Because the environment is deterministic, both runs played
*the exact same 10 games*, so each seed is a matched pair and the per-seed
difference cancels the (enormous) seed-to-seed variance.

What it reports:
  - per-seed deltas, with truncated episodes flagged and excluded
  - mean/median paired delta and its standard error
  - a bootstrap 95% CI on the paired mean difference
  - an exact two-sided sign test (distribution-free)
  - a Fisher exact test on the invalid-move rate (instruction following)
  - token cost ratio, so a win can be judged against what it cost

Usage:
    python3 examples/compare_runs.py results/<run-a> results/<run-b>
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple

FINISHED = {"board_full", "max_moves"}


def load(run_dir: Path) -> Tuple[Dict[int, dict], dict]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {r["seed"]: r for r in summary["episodes_detail"]}, summary["aggregates"]


def bootstrap_ci(diffs: List[float], iters: int = 20000, seed: int = 7) -> Tuple[float, float]:
    random.seed(seed)
    means = sorted(statistics.mean(random.choices(diffs, k=len(diffs))) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def sign_test(wins: int, n: int) -> float:
    """Exact two-sided sign test p-value."""
    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _hyper(a: int, b: int, c: int, d: int) -> float:
    return comb(a + b, a) * comb(c + d, c) / comb(a + b + c + d, a + c)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on a 2x2 table."""
    observed = _hyper(a, b, c, d)
    total = a + c
    p = 0.0
    for i in range(total + 1):
        j = total - i
        if i <= a + b and j <= c + d:
            pr = _hyper(i, a + b - i, j, c + d - j)
            if pr <= observed * (1 + 1e-9):
                p += pr
    return min(1.0, p)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_a", type=Path, help="baseline run directory")
    ap.add_argument("run_b", type=Path, help="variant run directory")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args(argv)

    a_rows, a_agg = load(args.run_a)
    b_rows, b_agg = load(args.run_b)
    seeds = sorted(set(a_rows) & set(b_rows))
    if not seeds:
        print("error: runs share no seeds — not comparable")
        return 2

    la, lb = args.label_a, args.label_b
    print(f"paired comparison over {len(seeds)} shared seeds\n")
    print(f"{'seed':>6} {la:>10} {lb:>10} {'delta':>8}   note")
    diffs: List[int] = []
    for s in seeds:
        sa, sb = a_rows[s]["score"], b_rows[s]["score"]
        bad = [l for l, r in ((la, a_rows[s]), (lb, b_rows[s]))
               if r["reason"] not in FINISHED]
        note = f"  <- excluded: {'/'.join(bad)} truncated" if bad else ""
        if not bad:
            diffs.append(sb - sa)
        print(f"{s:>6} {sa:>10} {sb:>10} {sb-sa:>+8}{note}")

    if len(diffs) < 2:
        print("\ntoo few clean pairs for statistics")
        return 0

    wins = sum(1 for d in diffs if d > 0)
    mean_d = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    sem = sd / len(diffs) ** 0.5
    lo, hi = bootstrap_ci([float(d) for d in diffs])

    print(f"\nclean pairs       : {len(diffs)} ({lb} wins {wins}, loses {len(diffs)-wins})")
    print(f"mean paired delta : {mean_d:+.1f}  (median {statistics.median(diffs):+.1f})")
    print(f"stdev {sd:.1f}, s.e.m. +/-{sem:.1f}  -> |delta|/sem = {abs(mean_d)/sem:.2f}")
    print(f"95% bootstrap CI  : [{lo:+.0f}, {hi:+.0f}]  "
          f"-> {'excludes' if lo > 0 or hi < 0 else 'INCLUDES'} zero")
    print(f"exact sign test   : p = {sign_test(wins, len(diffs)):.3f}")

    ai = sum(r["invalid_actions"] for r in a_rows.values())
    am = sum(r["moves"] for r in a_rows.values())
    bi = sum(r["invalid_actions"] for r in b_rows.values())
    bm = sum(r["moves"] for r in b_rows.values())
    print(f"\ninstruction following (invalid placements):")
    print(f"  {la:>10}: {ai:>3}/{am:<4} = {ai/am:.4f}")
    print(f"  {lb:>10}: {bi:>3}/{bm:<4} = {bi/bm:.4f}")
    print(f"  Fisher exact p = {fisher_exact(ai, am - ai, bi, bm - bi):.4f}")

    ta, tb = a_agg.get("tokens_per_move", 0), b_agg.get("tokens_per_move", 0)
    if ta and tb:
        print(f"\ntoken cost: {tb/ta:.2f}x  ({ta:.0f} -> {tb:.0f} per move)")
    print(f"completion: {la} {a_agg['valid_completion_rate']:.0%}, "
          f"{lb} {b_agg['valid_completion_rate']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
