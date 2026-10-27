# blockblast-bench

[![tests](https://github.com/kittimzhe/blockblast-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/kittimzhe/blockblast-bench/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/dependencies-zero-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

**[中文文档](README.md)**

A deterministic **Block Blast** (8×8 block puzzle) game environment and **LLM-agent benchmark** in pure stdlib Python. LLMs and classical agents play identical seeded games; the harness scores **spatial planning, long-horizon survival and instruction following**, renders episodes as animated GIFs, and also ships as an [inspect_ai](https://inspect.aisi.org.uk/) Task.

- **Zero dependencies** — the whole thing (env, harness, even the GIF encoder) runs on the Python standard library, Python 3.9+.
- **Fully deterministic** — same seed ⇒ same piece sequence; every agent faces the identical deal, so results are fair and reproducible.
- **Battle-tested** — 107 tests across four layers, including randomized invariant testing, hostile-input fuzzing and golden-pinned engine semantics; CI runs on Python 3.9 and 3.13.

## Quick start

```bash
git clone https://github.com/kittimzhe/blockblast-bench.git
cd blockblast-bench

# watch a live demo (heuristic agent)
python3 -m blockblast demo --agent heuristic --seed 42

# run the classic baselines (random / greedy / heuristic / beam, 10 games each)
bash examples/run_baselines.sh

# evaluate any OpenAI-compatible LLM
DEEPSEEK_API_KEY=sk-... python3 -m blockblast run --agent llm \
    --base-url https://api.deepseek.com --model deepseek-chat \
    --api-key-env DEEPSEEK_API_KEY --episodes 5 --save-dialog

# leaderboard / replay / export an animated GIF
python3 -m blockblast leaderboard
python3 -m blockblast gif results/<run_id>/episodes/seed_2024.json --speed 1.5
```

Live HTML report: **https://kittimzhe.github.io/blockblast-bench/**

## Results ladder (default seeds 2024–2033, 10 episodes)

| # | Agent | Mean | Median | Max | Moves | Lines | Combo |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | beam (width 80) | 3953.2 | 4014.5 | **10140** | 384.8 | 186.0 | 4 |
| 2 | beam (width 40) | 2031.2 | 1627.0 | 6141 | 200.0 | 94.5 | 4 |
| 3 | heuristic | 586.9 | 513.0 | 1142 | 64.0 | 26.9 | 4 |
| 4 | greedy | 169.3 | 151.5 | 358 | 24.7 | 6.7 | 4 |
| 5 | random | 56.9 | 55.5 | 94 | 13.3 | 0.9 | 1 |

**LLM result** (qwen3.7-plus, same seeds, complete run): mean **416.8** / median 405 / **max 862** — comfortably above `greedy`; the best game (87 moves, 37 lines, combo 7) exceeds the heuristic *mean*. All 10 episodes reached a natural game over (`board_full` ×10, 100% completion). Decision analysis: 0.953 single-step efficiency, 96% clear conversion (184/192), 7895 tokens/move.

> **Methodology finding**: this run was completed with `--resume` after endpoint faults truncated 6 episodes. Comparing the truncated snapshot (mean 286.9) with the complete run (mean 416.8): **the mean was depressed by 31% purely by infrastructure loss, while single-step efficiency (0.954 → 0.953) and clear conversion (96% → 96%) barely moved.** Process metrics are immune to truncation; outcome metrics are highly sensitive — so an eval report must publish its completion rate, otherwise "weak model" and "flaky endpoint" are indistinguishable.

Where an agent lands on this ladder is directly interpretable: **below random ⇒ cannot even produce legal placements; above greedy ⇒ one-step payoff planning; near heuristic ⇒ board-shape management; near beam ⇒ planning across tray cycles.**

`beam` runs an exact lookahead within the current tray cycle (the tray refills only after all 3 pieces are used) — 6.7× the heuristic mean at width 80. Width sensitivity: more search lifts the ceiling dramatically (best game 1142 → **10140**, 1004 moves), but the mean enters a high-variance regime beyond width ≈20 — game scores are heavy-tailed, so max/median beat mean as peak-ability indicators on small samples.

## Game rules (faithful recreation)

- 8×8 board; a tray of 3 pieces that refills only when all are used; **no rotations/flips**.
- Any completely filled row or column clears (several at once); consecutive clearing turns build a combo multiplier.
- 19-shape catalog (as in 1010!/Block Blast): 1×1, 2–5 bars, 2×2, 3×3, small/big L — `python3 -m blockblast shapes`.
- Game over when none of the remaining tray pieces fits anywhere.

Scoring is a documented approximation (`ScoringConfig`, tunable): +1/cell placed; progressive line points 10/30/60…; combo ×(1+0.5·(n−1)) capped at ×4; perfect clear +300. All agents play under identical rules, so rankings stay fair even if the constants differ from the official game.

## Evaluating LLMs

Any endpoint speaking `POST {base_url}/chat/completions` works out of the box. The model sees a coordinate-labelled text board plus tray mini-grids and must reply with exactly `{"piece": 0-2, "row": 0-7, "col": 0-7}`.

- **Invalid moves are a measured capability**: illegal replies get error feedback and up to 2 retries per move; persistent failure ends the episode (partial score kept) and feeds `invalid_per_move` / `valid_completion_rate`.
- **Network resilience**: exponential backoff, retryable-status handling, adaptive timeout growth (for slow reasoning models), transient-body detection (200 with empty/non-JSON body counts as retryable).
- **Resume**: `--resume <run_dir>` finishes interrupted episodes from their exact breakpoint — env determinism makes replay lossless; tokens/stats merge across segments.
- **Prompt variants**: `--allow-reasoning` invites analysis before the final JSON line (the parser takes the *last* valid JSON, so candidate moves mentioned mid-reasoning cannot shadow the answer).
- `--save-dialog` stores every prompt/reply for failure analysis.

### inspect_ai Task

`blockblast/inspect_task.py` adapts the benchmark to an inspect_ai Task (one sample per seed, custom solver game loop, `game_score` + `finished_cleanly` scorers):

```bash
./examples/run_inspect.sh openai/gpt-4o-mini -T episodes=10
```

## Decision-quality analysis

`blockblast analyze <run_dir>` replays transcripts deterministically and scores every move against the single-step optimum:

- `greedy_efficiency` — actual points ÷ optimal points (the greedy baseline scores exactly 1.000 by construction: a built-in sanity check);
- `clear_conversion` / `missed_clears` — clears taken when one was available;
- `final_fill` / `final_empty_regions` — death-board forensics;
- `invalid_kinds` — taxonomy of illegal moves.

Finding of note: beam deliberately skips 26% of immediately available clears to complete bigger ones within the tray cycle (74% conversion vs the heuristic's 99%) — clear conversion alone distinguishes short-sighted from long-horizon play.

## Testing strategy

| Layer | What it covers |
|---|---|
| Unit | piece catalog, scoring formula, placement/clear rules, text rendering |
| Randomized invariants | hundreds of random games asserting cell conservation, cleared lines empty, combo/score accounting re-derivable, tray lifecycle, game-over ⇔ no legal action; **fuzzing** proves `step()` only ever raises `InvalidActionError` and never mutates state on rejection; **golden tests** pin fixed-seed outcomes |
| Regression bands | baseline score bands + capability ladder (random < greedy < heuristic) |
| End-to-end | all CLI subcommands, artifact schemas, resume + lineage (chains collapse to their final link), network fault injection (timeout/429/5xx/empty-200), pixel-level GIF decode verification |

## Artifacts

```
results/
├── runs.jsonl                  # one line per run (leaderboard index)
├── leaderboard.md              # auto-refreshed markdown leaderboard
└── <run_id>/
    ├── summary.json            # metadata + aggregates + per-episode summary
    └── episodes/seed_N.json    # full transcript (+ LLM dialog if enabled)
```

`blockblast report` renders a single-file HTML report (leaderboard, per-run KPI cards, inline SVG charts, embedded GIFs — zero JS/external assets).

## Design trade-offs

- **Piece distribution**: the real game's generator is proprietary and possibly adaptive ("pity" mechanics); we sample the 19 shapes uniformly for reproducibility and fairness (`BlockBlastEnv(piece_weights=...)` for custom weights).
- Scoring constants are approximations; rankings are comparable within the same rules and seed set.
- Colors are not modelled — clears depend only on occupancy.
- Unrelated to Block Blast! (Hungry Studio); a research recreation of the mechanics.

## Contributing

PRs welcome — stronger baselines (MCTS), prompt-variant suites, more model results (attach `results/<run_id>/`), HTML report improvements. Please keep the core stdlib-only and make `python -m pytest` pass on 3.9 and 3.13.

## License

MIT
