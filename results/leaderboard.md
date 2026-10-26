# Block Blast Benchmark — Leaderboard

| # | Agent | Model | Eps | Mean | Median | Max | Moves | Lines | Combo | Inv/mv | Tok/mv | Run |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | heuristic | - | 10 | 586.9 | 513.0 | 1142 | 64 | 26.9 | 4 | 0.0 | - | 20261023-113751-heuristic-baseline-ede1 |
| 2 | greedy | - | 10 | 169.3 | 151.5 | 358 | 24.7 | 6.7 | 4 | 0.0 | - | 20261023-113634-greedy-baseline-b072 |
| 3 | llm | qwen3.7-plus | 3 | 98.3 | 114 | 137 | 11.7 | 4.3 | 2 | 0.0286 | 5912.6 | 20261023-150523-llm-qwen3.7-plus-mass-0725 |
| 4 | random | - | 10 | 56.9 | 55.5 | 94 | 13.3 | 0.9 | 1 | 0.0 | - | 20261023-113605-random-baseline-77ca |
| 5 | llm | GLM-5.3 | 3 | 25.7 | 25 | 47 | 3 | 1 | 2 | 0.0 | 4128.1 | 20261023-142549-llm-GLM-5.3-mass-835d |

_Scores are only comparable between runs sharing the same seed set and scoring config._
