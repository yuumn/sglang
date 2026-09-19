#!/usr/bin/env bash
set -euo pipefail
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang/.sglangenv_07/bin/activate
SGLANG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd $SGLANG_DIR
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

TARGET_MODEL="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B"
DRAFT_MODEL="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec/train_log_checkpoints/train_dflash_qwen3_4b_ce-0.1_l1-0.9_20260828_182921/checkpoints/dflash_block7_qwen3_4b/step_5100"


exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${TARGET_MODEL}" \
  --served-model-name qwen3-4b-dflash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-backend triton \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-draft-attention-backend triton \
  --mem-fraction-static 0.20 \
  --disable-cuda-graph 

