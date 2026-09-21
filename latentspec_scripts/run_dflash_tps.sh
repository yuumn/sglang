#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source ${SGLANG_DIR}/.sglangenv_07/bin/activate

export SUFFIX=dflash
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd $SCRIPT_DIR
HDD_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec
DFLASH_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec/train_log_checkpoints/train_dflash_qwen3_4b_ce-0.1_l1-0.9_20260828_182921/checkpoints/dflash_block7_qwen3_4b
DRAFT_MODEL_LIST=(
    $DFLASH_DIR
)

export DRAFT_MODEL=

bash run_tps.sh

