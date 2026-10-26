#!/usr/bin/env bash
# Run the benchmark as an inspect_ai Task (requires Python 3.10+).
#
#   MASS_API_KEY=... ./examples/run_inspect.sh openai-api/mass/qwen3.7-plus
#   OPENAI_API_KEY=... ./examples/run_inspect.sh openai/gpt-4o-mini
#   ./examples/run_inspect.sh mockllm/model        # smoke test, no network
#
# Extra args are passed straight to `inspect eval`, e.g.:
#   ./examples/run_inspect.sh mockllm/model -T episodes=3 -T seed_base=9000
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-mockllm/model}
shift || true

# mass platform preset: inspect's generic OpenAI-compatible provider
if [[ "$MODEL" == openai-api/mass/* ]]; then
  export MASS_BASE_URL=${MASS_BASE_URL:-https://maas.appia.vip/api/open-apis/v1}
fi

RUN="uv run --no-project --with inspect-ai --with openai --python 3.13 --"
export PYTHONPATH=.
exec $RUN inspect eval blockblast/inspect_task.py \
  --model "$MODEL" \
  --log-dir "${INSPECT_LOG_DIR:-./inspect-logs}" \
  "$@"
