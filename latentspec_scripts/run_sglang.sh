
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/sglang/.sglangenv_07/bin/activate
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

TARGET_MODEL=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B

python -m sglang.launch_server \
  --model ${TARGET_MODEL} \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --trust-remote-code 



