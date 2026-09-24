#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR
bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/gpu.sh

# qwen3-4b 1-epoch
EAGLE3_DRAFT_MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_eagle3_qwen3_4b_20260712_005338/checkpoints/eagle3_ttt7_qwen3_4b/step_2616
DFLASH_DRAFT_MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_dflash_qwen3_4b_ce-0.1_l1-0.9_20260828_182921/checkpoints/dflash_block7_qwen3_4b/step_2616
DSPARK_DRAFT_MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_dspark_qwen3_4b_20260705_174128/checkpoints/dspark_block7_qwen3_4b/step_2616
MYSPEC_DRAFT_MDOEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_myspec_qwen3_4b_0.1ce-0.9l1_PerfectBlend_20260921_065327/checkpoints/myspec_block7_qwen3_4b/step_2616
MYSPEC_DSPARK_DRAFT_MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_myspec-markov-conf_qwen3_4b_0.1ce-0.9l1_PerfectBlend__20260916_215802/checkpoints/myspec_block7_qwen3_4b/step_2616

export TARGET_MODEL=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B
export LOWER_MODEL_NAME=qwen3-4b

## eagle3
export DRAFT_MODEL=$EAGLE3_DRAFT_MODEL
bash run_sglang_eagle3.sh 
# SUFFIX=eagle3-epoch1 bash run_tps.sh 
# bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/gpu.sh

## dflash
## dspark



# qwen3-8b 1-epoch
# EAGLE3_DRAFT_MODEL=
DFLASH_DRAFT_MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_qwen3-8b_online_0920/train_log_checkpoints/train_dflash_qwen3_8b_0.1ce-0.9l1_PerfectBlend_20260920_171219/checkpoints/dflash_block7_qwen3_8b/step_2618
# DSPARK_DRAFT_MODEL=
MYSPEC_DRAFT_MDOEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_qwen3-8b_online_0920/train_log_checkpoints/train_myspec_qwen3_8b_0.1ce-0.9l1_PerfectBlend_20260922_040329/checkpoints/myspec_block7_qwen3_8b/step_2618
# MYSPEC_DSPARK_DRAFT_MODEL=
export TARGET_MODEL=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-8B
export LOWER_MODEL_NAME=qwen3-8b

## baseline 
# export SERVED_MODEL_NAME=qwen3-8b-baseline
# bash run_sglang_baseline.sh &
# SUFFIX=baseline bash run_tps.sh 
# bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/gpu.sh

## eagle3

## dflash
# export DRAFT_MODEL=$DFLASH_DRAFT_MODEL
# export SERVED_MODEL_NAME=qwen3-8b-dflash
# bash run_sglang_dflash.sh &
# SUFFIX=dflash-epoch1 bash run_tps.sh 
# bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/gpu.sh

## LatentSpec
# export DRAFT_MODEL=$MYSPEC_DRAFT_MDOEL
# export SERVED_MODEL_NAME=qwen3-8b-latentspec
# bash run_sglang_latentspec.sh 
# SUFFIX=latentspec-epoch1 bash run_tps.sh 
# bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/gpu.sh

