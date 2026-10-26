#!/usr/bin/env bash
# Run the classic baselines (random/greedy/heuristic/beam) on the default seed set.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
EPISODES=${EPISODES:-10}

$PYTHON -m blockblast run --agent random    --episodes "$EPISODES" --tag baseline
$PYTHON -m blockblast run --agent greedy    --episodes "$EPISODES" --tag baseline
$PYTHON -m blockblast run --agent heuristic --episodes "$EPISODES" --tag baseline
$PYTHON -m blockblast run --agent beam      --episodes "$EPISODES" --tag baseline

echo
$PYTHON -m blockblast leaderboard
