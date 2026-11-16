# 分工与日程（截止 2026-11-27 周五）

## 模型选择（全组统一）

| 用途 | 模型 | `--model` 写法 | 说明 |
|---|---|---|---|
| **主力模型**：所有方法对比、消融 | DeepSeek V4.1 Flash | `openrouter:deepseek/deepseek-v4.1-flash` | 便宜、上下文长。不在论文里，所以需要下一行的参照模型 |
| **论文参照**：证明我们的评测和论文一致 | LLaMA-3.3-70B-Instruct | `openrouter:meta-llama/llama-3.3-70b-instruct` | 论文里出现过且最便宜。只跑 memory 基线，和论文 Table 1–3 对数字 |
| **方法 B 的对照**，同时免费调试方法 A | Qwen3-4B-Instruct-2507（本地 vLLM） | `vllm:Qwen/Qwen3-4B-Instruct-2507` | 未训练的原模型，在 4090 上用 `vllm serve` 部署 |
| 可选的重点展示结果 | DeepSeek-R1 | `openrouter:deepseek/deepseek-r1` | 论文里出现过的推理模型，token 用量大，只在 Boolean 和大规模设置上跑 |

论文评测过的 7 个模型（附录 Table 9）：GPT-5、Gemini-2.5-Pro、Claude-Opus-4、DeepSeek-R1、Grok-4、GPT-4o、LLaMA-3.3-70B-Instruct。

规则：
- 同一张对比表里，所有方法使用同一个模型、同一批实例（`--seed 33550336 --n-instances 30`）。
- 跑新设置之前，先用 `--n-instances 5` 试跑，估算成本，并确认基线没有达到上限。如果 V4.1 Flash 的 RR 已经在 95% 以上，就把这个设置从主表里去掉，主实验放在大规模和 Boolean 设置上。
- OpenRouter 上的模型 ID 以官网页面为准，如果改了，同步更新这张表。

## 分工

| 成员 | 负责 | 主要文件 | 视频（约 2.5 分钟） |
|---|---|---|---|
| A | 数据与评测：对齐论文难度设置，跑全部基线（direct / memory / stratified，以及暴力枚举上界），确认 Boolean 验证器缺陷 | `scripts/make_datasets.sh`, `hypocover/tasks/`, `hypocover/eval/` | 问题、基准、失败模式、验证器缺陷 |
| B | 方法 A：分区搜索的调参和改进（让模型估计剩余答案数、用 LLM 自己提出划分方式等） | `hypocover/methods/partition.py` | 方法 A |
| C | 方法 B：在 2×4090 上做 GRPO 训练和评测 | `rl/` | 方法 B |
| D | 实验矩阵、消融、作图、README 定稿 | `hypocover/eval/summarize.py`、新增作图脚本 | 结果、消融、局限 |

## 日程

| 日期 | 目标 |
|---|---|
| 11/17–18 | A：生成全部数据；先用 V4.1 Flash 各跑 5 道题，确认哪些设置还有提升空间；再用 V4.1 Flash 跑完 3 个基线 × 3 个任务；用 LLaMA-3.3-70B 跑 memory 基线和论文对照；B：在 3D tp3 和大规模 3D 上调试分区搜索；C：`rl/launch.sh setup`，用 1.5B 模型跑 50 步，先确认训练流程能通 |
| 11/19–22 | B：迭代方法 A；C：用 4B 模型正式训练（共享奖励和不共享奖励两种）；D：写作图脚本（RR 曲线、简单/复杂假设召回率），整理表格 |
| **11/23** | **冻结方法**，之后只跑实验 |
| 11/24–25 | 完整实验和消融，用一个前沿模型在 Boolean 和大规模 Causal/3D 上跑一次作为重点结果 |
| 11/26 | 每人录自己那段 |
| 11/27 | 剪辑、仓库设为 public、提交 |

## 待办 / 未决

- [ ] causal n6 和 large 的生成器参数需要对齐论文（|H_O|=160）；n6 全枚举很慢，考虑在 GPU 机器上跑或使用 `--n-samples`
- [ ] Boolean 难度和论文对齐：论文 Hard 平均 |H_O| 约 78，含 4–5 个算子的表达式，应该是深度 3；确认使用哪种算子集
- [ ] 方法 A：加入"模型自己估计剩余答案数"的停止规则，以及由 LLM 提出划分方式的变体
- [ ] 方法 B：先单独训练 3D 和 causal；如果 Boolean 的 VR 太低，再加 Boolean
- [ ] 作图：RR 随查询数变化的曲线（每条 `rr_curve` 都已保存在 instances.jsonl）
