#!/usr/bin/env bash
# Run the three classic baselines on the default seed set and print the leaderboard.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
EPISODES=${EPISODES:-10}

$PYTHON -m blockblast run --agent random    --episodes "$EPISODES" --tag baseline
$PYTHON -m blockblast run --agent greedy    --episodes "$EPISODES" --tag baseline
$PYTHON -m blockblast run --agent heuristic --episodes "$EPISODES" --tag baseline

echo
$PYTHON -m blockblast leaderboard
