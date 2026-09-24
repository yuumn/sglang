
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang/.sglangenv_07/bin/activate
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

TARGET_MODEL=${TARGET_MODEL:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-4b-eagle3 }

python -m sglang.launch_server \
  --model-path ${TARGET_MODEL} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --trust-remote-code \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path ${DRAFT_MODEL} 
  # --speculative-num-draft-tokens 8
  # --dtype bfloat16 \
  # --attention-backend triton \
  # --speculative-draft-attention-backend triton 
  # --mem-fraction-static 0.20 
  # --disable-cuda-graph 
  # --disable-overlap-schedule \
  # --cuda-graph-backend-decode disabled \
  # --cuda-graph-backend-prefill disabled

