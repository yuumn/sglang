#!/usr/bin/env bash
set -euo pipefail

# SGLANG_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang"
# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd $SGLANG_DIR

VENV_DIR="${SGLANG_DIR}/.venv"
TARGET_MODEL="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B"
DRAFT_MODEL="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec/train_log_checkpoints/train_dflash_qwen3_4b_ce-0.1_l1-0.9_20260828_182921/checkpoints/dflash_block7_qwen3_4b/step_5100"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

# COMPAT_PACKAGES="${SGLANG_DIR}/.mllm_compat_packages"
# mkdir -p "${COMPAT_PACKAGES}"
# ln -sfn "${VENV_DIR}/lib/python3.12/site-packages/xgrammar" \
#   "${COMPAT_PACKAGES}/xgrammar"
export PYTHONPATH="${SGLANG_DIR}/python:${COMPAT_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${VENV_DIR}/lib/python3.12/site-packages/xgrammar${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${SGLANG_DIR}"
source "${VENV_DIR}/bin/activate"

# The venv may have been created on another host with an absolute Python
# symlink. Fall back to the host's compatible Python 3.12 in that case.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  PYTHON_BIN="/usr/local/bin/python3.12"
fi

exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${TARGET_MODEL}" \
  --served-model-name qwen3-4b-dflash \
  --host "${SGLANG_HOST:-0.0.0.0}" \
  --port "${SGLANG_PORT:-30000}" \
  --tp-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-backend triton \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-draft-attention-backend triton \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.20}" \
  --disable-cuda-graph \
  "$@"
