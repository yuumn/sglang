#!/usr/bin/env bash
set -euo pipefail

SGLANG_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang
SGLANG_ENV=${SGLANG_ROOT}/.sglangenv_07
source "${SGLANG_ENV}/bin/activate"

SGLANG_PYTHON=${SGLANG_ENV}/bin/python
if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  # The checked-in environment may contain a stale absolute interpreter symlink.
  SGLANG_PYTHON=${SGLANG_ROOT}/.venv/bin/python
  if [[ ! -x "${SGLANG_PYTHON}" ]]; then
    echo "No usable Python interpreter found for SGLang." >&2
    exit 1
  fi
  export PYTHONPATH="${SGLANG_ROOT}/python:${SGLANG_ENV}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
fi
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

TARGET_MODEL=${TARGET_MODEL:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_dflash_qwen3_4b_ce-0.1_l1-0.9_20260828_182921/checkpoints/dflash_block7_qwen3_4b/step_2616}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-4b-dflash}



"${SGLANG_PYTHON}" -m sglang.launch_server \
  --model-path ${TARGET_MODEL} \
  --served-model-name $SERVED_MODEL_NAME \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-num-draft-tokens 8 \
  --speculative-dflash-prediction-hidden-start 0 \
  "$@"

