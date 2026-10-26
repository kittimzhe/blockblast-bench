#!/usr/bin/env bash
# Evaluate an LLM on the benchmark. Examples for common OpenAI-compatible endpoints.
#
#   DEEPSEEK_API_KEY=sk-... ./examples/run_llm.sh deepseek
#   OPENAI_API_KEY=sk-...   ./examples/run_llm.sh openai
#   GLM_API_KEY=...         ./examples/run_llm.sh glm
#   ./examples/run_llm.sh local          # local vLLM / Ollama on :8000
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
EPISODES=${EPISODES:-5}
PRESET=${1:-deepseek}

case "$PRESET" in
  deepseek)
    ARGS=(--base-url https://api.deepseek.com --model deepseek-chat --api-key-env DEEPSEEK_API_KEY)
    ;;
  openai)
    ARGS=(--base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key-env OPENAI_API_KEY)
    ;;
  glm)
    ARGS=(--base-url https://open.bigmodel.cn/api/paas/v4 --model glm-4-flash --api-key-env GLM_API_KEY)
    ;;
  local)
    ARGS=(--base-url http://127.0.0.1:8000/v1 --model local-model --api-key-env LOCAL_API_KEY)
    export LOCAL_API_KEY=${LOCAL_API_KEY:-dummy}
    ;;
  *)
    echo "unknown preset: $PRESET (use deepseek|openai|glm|local)" >&2
    exit 2
    ;;
esac

$PYTHON -m blockblast run --agent llm "${ARGS[@]}" \
  --episodes "$EPISODES" --save-dialog --tag "$PRESET"

echo
$PYTHON -m blockblast leaderboard
