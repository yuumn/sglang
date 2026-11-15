#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VARIANT=${1:?usage: launch_full_server.sh variant [server options]}
shift
OPTIONS=()
case "$VARIANT" in
  dspark) ALGORITHM=DSPARK ;;
  dspark_latentspec) ALGORITHM=DSPARK_LATENTSPEC ;;
  dflash|dflash-3L) ALGORITHM=DFLASH; OPTIONS+=(--speculative-dflash-prediction-hidden-start 0) ;;
  eagle3) ALGORITHM=EAGLE3; OPTIONS+=(--speculative-num-steps 7 --speculative-eagle-topk 1) ;;
  latentspec) ALGORITHM=LATENTSPEC ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}
exec "$VIRTUAL_ENV/bin/python" -m sglang.launch_server \
  --model-path "${TARGET_MODEL:?Verified target model cache is required}" \
  --speculative-draft-model-path "${DRAFT_MODEL:?Verified draft model cache is required}" \
  --served-model-name "${MODEL_FAMILY:?}-$VARIANT" \
  --speculative-algorithm "$ALGORITHM" --speculative-num-draft-tokens 8 \
  "${OPTIONS[@]}" \
  --host 127.0.0.1 --port "${PORT:-30000}" --tp-size 1 \
  --trust-remote-code --dtype bfloat16 --weight-loader-disable-mmap \
  --random-seed 980406 --attention-backend flashinfer \
  --speculative-draft-attention-backend triton \
  --mem-fraction-static 0.50 --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 --context-length 16384 "$@"
