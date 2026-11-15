#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VARIANT=${1:?usage: launch_server.sh dspark|dspark_latentspec [server options]}
shift
case "$VARIANT" in
  dspark) ALGORITHM=DSPARK ;;
  dspark_latentspec) ALGORITHM=DSPARK_LATENTSPEC ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}
MODEL_CACHE=${MODEL_CACHE:-/dev/shm/rebuttal-models}
TARGET_DEFAULT=/mydata/Qwen/Qwen3-4B
DRAFT_DEFAULT=/mydata/models/qwen3-4b/$VARIANT/step_2616
if [[ ${USE_MODEL_CACHE:-1} == 1 && -f "$MODEL_CACHE/manifest.json" ]]; then
  TARGET_DEFAULT=$MODEL_CACHE/target
  DRAFT_DEFAULT=$MODEL_CACHE/$VARIANT
  echo "Using verified local checkpoint copies; sources/hashes: $MODEL_CACHE/manifest.json"
fi
exec "$ROOT/.venv/bin/python" -m sglang.launch_server \
  --model-path "${TARGET_MODEL:-$TARGET_DEFAULT}" \
  --speculative-draft-model-path "${DRAFT_MODEL:-$DRAFT_DEFAULT}" \
  --speculative-algorithm "$ALGORITHM" \
  --speculative-num-draft-tokens 8 \
  --served-model-name "qwen3-4b-$VARIANT" \
  --host 127.0.0.1 --port "${PORT:-30000}" --tp-size 1 \
  --trust-remote-code --dtype bfloat16 \
  --weight-loader-disable-mmap --random-seed 980406 \
  --attention-backend "${TARGET_ATTENTION_BACKEND:-flashinfer}" \
  --speculative-draft-attention-backend triton \
  --mem-fraction-static 0.50 --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 --context-length 16384 \
  "$@"
