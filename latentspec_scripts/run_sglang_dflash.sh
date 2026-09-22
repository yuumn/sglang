
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang/.sglangenv_07/bin/activate
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

TARGET_MODEL=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B
DRAFT_MODEL=${DRAFT_MODEL:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code/train_log_checkpoints/train_dflash_qwen3_4b_ce-0.1_l1-0.9_20260828_182921/checkpoints/dflash_block7_qwen3_4b/step_2616}

python -m sglang.launch_server \
  --model-path ${TARGET_MODEL} \
  --served-model-name qwen3-4b-dflash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${DRAFT_MODEL}" \
  # --dtype bfloat16 \
  # --attention-backend triton \
  # --speculative-draft-attention-backend triton 
  # --mem-fraction-static 0.20 
  # --disable-cuda-graph 

