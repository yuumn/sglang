#!/usr/bin/env bash
set -euo pipefail
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang/.sglangenv_07/bin/activate
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

TARGET_MODEL=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B

TARGET_MODEL=${TARGET_MODEL:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B}
# DRAFT_MODEL=${DRAFT_MODEL:-}
DRAFT_MODEL=${DRAFT_MODEL:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_myspec_qwen3_4b_0.1ce-0.9l1_PerfectBlend_20260921_065327/checkpoints/myspec_block7_qwen3_4b/step_2616}

python -m sglang.launch_server \
  --model-path "${TARGET_MODEL}" \
  --served-model-name qwen3-4b-latentspec \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --trust-remote-code \
  --speculative-algorithm LATENTSPEC \
  --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-num-draft-tokens 8
