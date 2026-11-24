# Trace 字段说明

每题一个 JSONL，每行一个 slot（从 1 开始），字段：

| 字段 | 含义 |
|---|---|
| `question` | 题号 |
| `slot` | 第几次生成（1..budget） |
| `action` | 本次使用的引导动作（`free` / `coord1` / ...） |
| `prompt` | 完整 prompt |
| `response` | **模型原始回答**（不做任何修改） |
| `vec` | 解析出的高度向量；`None` 表示解析失败 |
| `new` | 这个向量是否第一次出现（相对于本 trace 之前的 slot） |
| `vec_original` / `new_original` | 如果派生字段被重解析修正，这里保留旧值 |
| `parse_note` | 重解析说明 |
| `finish_reason` / `usage` / `provider` / `seconds` | API 元信息 |

## 解析器版本

- **v1（旧）**：取回复里**第一个** `H=(...)`；
- **v2（2026-11-24 起）**：取回复里**最后一个完整向量**，因为模型可能先在推理里列候选，最后一行才给答案。

受影响的一个典型 slot：`free/g3_h3_b3_079.jsonl` 第 27 轮。原始回答结尾是 `H=(3,2,3)`，但旧解析器取了推理中提到的 `H=(1,1,1)`，于是被误记为重复。修正后该 slot 的 `vec=[3,2,3]`、`new=true`，旧值保留在 `vec_original` / `new_original`。

## 验证

```powershell
python repro/verify_free_100.py        # 重新解析全部 free trace：648 slots，0 重复，0 缺失
python repro/rebuild_predictions.py    # 从 response 重建 predictions-*.jsonl
```

**结论**：`free` 在 36 题上 648 次生成 0 重复；此前的“1 次重复 / 99.85%”是旧解析器造成的，不是模型行为。
