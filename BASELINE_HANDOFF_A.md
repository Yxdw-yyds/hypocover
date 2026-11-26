# 基线与早期交接记录（角色 A 原始说明归档）

> 本文件完整保留 `deepseek-3d-baseline-512` 分支最初的基线交接说明，供溯源核对；全组四模块汇总请见仓库根目录 [README.md](README.md)。

---

课程项目占 30%。四人沿用原组队，在 HypoSpace 上开发有新意的 ML 方法，提交公开仓库与 10 分钟小组视频，每个人参与讲解。评分看创新性、重要性与质量。

## 已完成的基线

3×3 俯视图，最大高度 3。两列题每列各有三种高度，共 9 个合法解，给 9 次生成机会。三列题有 27 个解，给 27 次机会。官方逐层提示包含此前可解析且去重的结构。每次生成一个候选，失败和重复同样消耗预算。

模型为 OpenRouter 的 `deepseek/deepseek-v4-flash-0731`，temperature 0.7，top_p 0.8，max_tokens 512，reasoning.enabled 为 false。OpenRouter 自动选择提供商。完整验证集复用了首轮四题的 512-token 记录。

| 集合 | 题数 / 候选 | 可解析率 | VR | NR | RR | 同题 Qwen3-8B RR |
|---|---:|---:|---:|---:|---:|---:|
| 验证 | 12 / 216 | 95.99% | 77.78% | 66.98% | 51.54% | 4.94% |
| 测试 | 24 / 432 | 95.52% | 79.48% | 66.20% | 53.55% | 1.08% |

百分比按题等权。VR 是有效候选比例，NR 是不同候选数除以机会数，RR 是不同合法解数除以全部合法解数。测试两列 RR 62.96%，三列 RR 44.14%。验证与测试总费用约 \$0.1176。测试中 85/432 条触及长度上限。每题只有一条采样轨迹，等价布局可能跨集合，结论限于当前题型。

程序均匀独立采样合法高度的期望 RR 为两列 65.36%、三列 63.90%（等权均值 64.63%），可作为解释该结构化任务难度的参照。

## 评分器用法

`summary.json` 的分组 2、3、0 分别表示两列、三列和全部；metrics 顺序为可解析率、VR、NR、RR。`results/<题号>.jsonl` 逐次保存提示、回答、token、费用、提供商和终止原因；同名 JSON 保存逐题评分。`data/*-inputs.jsonl` 是生成端输入，带枚举答案的 columns 文件仅供评测或训练。

```bash
python score_predictions.py validation predictions.jsonl --output validation-score.json
```

评分器来自冻结的上游版本，含全零层兼容修复。[HypoSpace 论文](https://proceedings.mlr.press/v306/chen26bz.html) · [上游代码版本](https://github.com/CTT-Pavilion/_HypoSpace/tree/c69e9318577b34b5b896996571aefd4ba6053f58/3d)
