"""Method B: GRPO + LoRA with the coverage reward, on 2x RTX 4090.

GPU 0 runs a vLLM generation server, GPU 1 trains (see rl/launch.sh):

  CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-4B-Instruct-2507 \
      --gpu-memory-utilization 0.85 --max-model-len 4096
  CUDA_VISIBLE_DEVICES=1 python -m rl.train_grpo --config rl/configs/qwen3_4b.yaml

Smaller first run: --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 50.
"""
from __future__ import annotations

import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rl/configs/qwen3_4b.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-share", action="store_true", help="ablation: reward 1 for any new valid answer")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.model:
        cfg["model"] = a.model
    if a.max_steps:
        cfg["grpo"]["max_steps"] = a.max_steps
    if a.output_dir:
        cfg["grpo"]["output_dir"] = a.output_dir

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from rl.coverage_reward import coverage_reward, coverage_reward_noshare

    ds = load_dataset("json", data_files=cfg["train_file"], split="train").shuffle(seed=0)
    args = GRPOConfig(**cfg["grpo"])
    peft = LoraConfig(**cfg["lora"]) if cfg.get("lora") else None
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=[coverage_reward_noshare if a.no_share else coverage_reward],
        args=args,
        train_dataset=ds,
        peft_config=peft,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
