#!/usr/bin/env bash
# Method B on the 2x RTX 4090 box.
#   bash rl/launch.sh setup      # install deps + build prompt set
#   bash rl/launch.sh serve      # terminal 1: vLLM generation server on GPU 0
#   bash rl/launch.sh train      # terminal 2: GRPO training on GPU 1
#   bash rl/launch.sh eval CKPT  # serve base + LoRA adapter for evaluation
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
CFG=${CFG:-rl/configs/qwen3_4b.yaml}

case ${1:-} in
  setup)
    uv venv --python 3.11 .venv && uv pip install -e ".[rl,dev]"
    bash scripts/make_datasets.sh all
    .venv/bin/python -m rl.build_dataset --out data/rl/train.jsonl --rows 6000
    ;;
  serve)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/trl vllm-serve --model "$MODEL" \
      --gpu-memory-utilization 0.85 --max-model-len 4096
    ;;
  train)
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m rl.train_grpo --config "$CFG" "${@:2}"
    ;;
  eval)
    CKPT=${2:?checkpoint dir}
    # then: VLLM_BASE_URL=http://localhost:8000/v1 python -m hypocover.eval.run --model vllm:cov ...
    CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve "$MODEL" --enable-lora \
      --lora-modules cov="$CKPT" --max-lora-rank 32 --max-model-len 8192 --port 8000
    ;;
  *) echo "usage: $0 setup|serve|train|eval CKPT"; exit 1 ;;
esac
