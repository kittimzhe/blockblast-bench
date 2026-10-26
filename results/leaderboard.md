# Block Blast Benchmark — Leaderboard

| # | Agent | Model | Eps | Mean | Median | Max | Moves | Lines | Combo | Inv/mv | Tok/mv | Run |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | beam | - | 10 | 2031.2 | 1627.0 | 6141 | 200 | 94.5 | 4 | 0.0 | - | 20261026-222707-beam-beam-v1-ed06 |
| 2 | heuristic | - | 10 | 586.9 | 513.0 | 1142 | 64 | 26.9 | 4 | 0.0 | - | 20261023-113751-heuristic-baseline-ede1 |
| 3 | llm | qwen3.7-plus | 10 | 286.9 | 246.0 | 862 | 32.1 | 12.9 | 7 | 0.0125 | 7549.9 | 20261026-190612-llm-qwen3.7-plus-resume-39e7 |
| 4 | llm | qwen3.7-plus | 10 | 286.9 | 246.0 | 862 | 32.1 | 12.9 | 7 | 0.0125 | 7549.9 | 20261026-185722-llm-qwen3.7-plus-resume-0d90 |
| 5 | llm | qwen3.7-plus | 10 | 286.9 | 246.0 | 862 | 32.1 | 12.9 | 7 | 0.0125 | 7549.9 | 20261026-104649-llm-qwen3.7-plus-mass-v3-692d |
| 6 | greedy | - | 10 | 169.3 | 151.5 | 358 | 24.7 | 6.7 | 4 | 0.0 | - | 20261023-113634-greedy-baseline-b072 |
| 7 | llm | qwen3.7-plus | 3 | 98.3 | 114 | 137 | 11.7 | 4.3 | 2 | 0.0286 | 5912.6 | 20261023-150523-llm-qwen3.7-plus-mass-0725 |
| 8 | random | - | 10 | 56.9 | 55.5 | 94 | 13.3 | 0.9 | 1 | 0.0 | - | 20261023-113605-random-baseline-77ca |
| 9 | llm | GLM-5.3 | 3 | 25.7 | 25 | 47 | 3 | 1 | 2 | 0.0 | 4128.1 | 20261023-142549-llm-GLM-5.3-mass-835d |

_Scores are only comparable between runs sharing the same seed set and scoring config._
