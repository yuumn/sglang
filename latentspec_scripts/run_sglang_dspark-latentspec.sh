#!/usr/bin/env bash
set -euo pipefail
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang/.sglangenv_07/bin/activate
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY


TARGET_MODEL=${TARGET_MODEL:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_myspec-markov-conf_qwen3_4b_0.1ce-0.9l1_PerfectBlend__20260916_215802/checkpoints/myspec_block7_qwen3_4b/step_2616}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-4b-dspark_latentspec}
PORT=${PORT:-30000}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-8}
# Without a profiled SPS table, compact mode still verifies every proposal and
# only adds confidence/scheduler overhead. Callers with an SPS table can opt in.
export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}

exec python -m sglang.launch_server \
  --model-path "${TARGET_MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp-size 1 \
  --trust-remote-code \
  --speculative-algorithm DSPARK_LATENTSPEC \
  --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}" \
  "$@"
