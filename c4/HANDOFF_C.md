# C 的交接说明：策略层分支（官方提示 + 学出来的提示后缀选择）

写于 2026-11-25。**一句话结论：这是一个诚实的负结果，没有提升性能。** 详细数字、偏离清单和复盘见 `c4/results_ds/RESULTS.md`。

## 1. 这条线做了什么

- 工作在**官方提示、官方逐层输出、官方评分器**之上，不改输出格式、不改评分。
- 动作 = 在官方提示末尾追加一句话：`none`（不加）、`low`（总方块数 < 2k）、`mid`（= 2k）、`high`（> 2k），k 为俯视图有方块的格子数。
- 学习器：每个动作一个 L2 逻辑回归，预测“这一步是否得到新的有效解”，每步选预测最高者；特征只用历史（已用预算、最近新解率和不合法率、各档已找到数、上一步动作及结果、k），不用真实答案分布。
- 训练：训练集 84 题，每步均匀随机选动作收集轨迹（2 次重复），按题分组交叉验证；先拟合并冻结（`c4/policy_ds.json`），再在验证集比较一次，测试集只跑一次。
- 对比臂：`free_official`、`paper_stratified`（论文 6.5 节分层，用了答案个数）、`tier_static`、`random_level`、`tier_cycle`、`learned`。

## 2. 结果（DeepSeek 官方 API，`deepseek-flash` = V4.1-Flash，max_tokens 1024，官方评分器）

| 臂 | 验证 RR | 测试 RR |
|---|---:|---:|
| free_official | 78.40% | 76.39% |
| **learned** | **78.86%** | **76.93%** |
| paper_stratified | 73.15% | 72.69% |
| tier_static | 67.13% | 68.98% |
| tier_cycle | 68.21% | 68.52% |
| random_level | 62.35% | 65.66% |

- **S1 未通过**：learned − free = +0.54 个百分点，95% CI [-2.0, +3.1]，要求 ≥ +5 且下界 > 0。
- S2、S3 形式上通过但**不要当作胜利**：学出来的策略约 93% 的步骤选 `none`，所以与 free 几乎一样；S3 的 98% 只统计可解析回答，按全部带提示槽位计，其他带提示臂合规率只有 63% 到 70%。
- 带提示的臂截断率 32% 到 38%（free 与 learned 为 15.5%），截断混淆了所有带提示臂的比较，也影响了学习器本身。**不能因此说“分层有害”，也不能说“提示本身无用”。**
- 样本小（验证 12 题、测试 24 题），噪声底（free 自己两次重复相差约 6 个百分点）高于 S1 阈值。

## 3. 与 A、B 的关系，以及不能直接比的数字

- A 的基线：OpenRouter `deepseek/deepseek-v4-flash-0731`，max_tokens 512，测试集 RR 53.55%。**我们的 76.39%（free）不能与它直接比**：模型版本、上限、服务商都不同。我们自己的旁证：旧模型上仅把上限提到 1024，验证集 free 约 59.7%。
- B（高度向量）改的是**输出格式**，绕开了截断和格式错误，在 A 的基线上提升很大，但不算 ML。C 保持官方格式，有学习但没有提升。两者不冲突，理论上可结合，但我们没有做，且不建议在截止前仓促做。
- **“决策层”本质上是在官方提示后追加提示词**，学到的只是“该选哪句后缀”。作为 ML 贡献很单薄，只适合作为严格评估过的负结果；不要写成突破性方法。

## 4. 偏离计划的地方（报告里需要写明）

1. 因 OpenRouter 余额耗尽（402）改用 DeepSeek 官方 API；模型版本因此不同，OpenRouter 的干净结果只作参考（验证集 free 59.72%、paper_stratified 65.90%，未提交）。
2. 策略在新模型的训练轨迹上重新拟合，发生在看到任何验证结果之前；旧 `policy.json` 作废。
3. 被 402 污染的输出已隔离（不提交），其中无任何真实模型回答，测试集没有被泄漏。
4. 第一次 DeepSeek 启动的输出已丢弃；训练轨迹来自两个时段（13:56 到 14:08 和 20:xx），价格口径不同，行为数据不受影响。
5. 换后端后没有重做长度截断的烟雾核对；轨迹里没有保存 `reasoning_tokens`，“关闭思考”未完全验证（一次最小探测显示正常）。
6. 费用是**估算**（高峰价、缓存另算），累计约 $12.98，以 DeepSeek 控制台为准。

## 5. 如何复现（需要自己的密钥，仓库里没有密钥）

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
python -B -m c4.run_all --backend deepseek --model deepseek-flash --price-in 0.30 --price-hit 0.006 --price-out 1.20 --ledger c4/spend_ledger_ds.jsonl --out-root c4/layered_ds --results-dir c4/results_ds --policy-file c4/policy_ds.json --total-cap 13
```

- 已提交：代码（`c4/*.py`）、冻结的 `policy_ds.json`、验证和测试的轨迹与预测、结果与日志。
- **没有提交训练轨迹**（约 19 MB），要重拟合策略需要重新运行第一步收集；`policy_ds.json` 已包含拟合结果。
- 测试：`python -B -m unittest discover -s c4 -p "test_*.py"`。

## 6. 没有提交的文件及原因

- `c4/save_key.py`、`c4/collect_with_key.py`：与密钥输入、存放相关，暂不提交；本仓库不含任何密钥，提交前已扫描 `sk-` 等特征。
- `~/.hypocover_openrouter_key`、`~/.hypocover_deepseek_key`：在用户目录，不在仓库内。**请删除并轮换这两把密钥。**
- 被污染的隔离目录、旧 OpenRouter 的训练轨迹、烟雾测试、临时账本和 `policy.json` 等。
- ChatGPT 生成的 `PROJECTION.md`、`projection_*.py`、`projection-runs/` 不是 C 的文件，也不在此分支。

## 7. 可选后续（需要花钱，且只能算探索性分析）

用 `max_tokens=2048` 重跑 `paper_stratified`（和 free 对照），看截断是否是带提示臂变差的主因，约 $1.5 到 $2.5（估算）。需要先调高 `--total-cap` 或换账本。已知风险：`mid` 的输出中位数已顶到 1024，2048 也未必够。**不能替换上面的冻结结论。**
