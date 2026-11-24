# 方法 B：高度向量表示 + 历史去重 + 固定引导

> 3D 假设生成任务（HypoSpace 基准）· 角色 B
> 代码/worktree：`hypocover-deepseek/`（分支 `deepseek-3d-baseline-512`）
> 数据：`hypocover-deepseek/data/columns-{2,3}.json`，split 见 `data/manifest.json`
> 结论：不换模型、不加预算，只换答案表示，把验证集 RR 从 **51.54% → 100.00%**、测试集 **53.55% → 100.00%**，费用降到约 **1/6.6**。

---

## 1. 问题定义

给定一个 3×3 的俯视图 `V ∈ {0,1}^{3×3}`（1 表示该列至少有一个方块），枚举所有满足以下条件的 3D 结构：

1. **俯视一致**：每一列只要有方块，投影就是 1；
2. **重力约束**：高度为 h 的柱子，第 1..h 层连续有方块，不能悬空；
3. **高度上限**：最多 3 层。

设占用列数为 `k`（两列题 `k=2`，三列题 `k=3`）。由于每根柱子的高度独立取 `{1,2,3}`，合法解集合恰好是

```
H_O = {1,2,3}^k ,   |H_O| = 3^k = 9 或 27
```

评测指标（每题预算 `N = |H_O|` 次生成）：

- **VR** = 合法候选数 / N
- **NR** = 不同候选数 / N
- **RR** = 命中的合法解数 / |H_O|（主指标）

---

## 2. 方法

### 2.1 核心思想

原始协议让模型输出逐层 0/1 文本。这种表示有两个问题：

1. **输出空间包含非法解**（第一层全 0、悬空、格式错）；
2. **输出很长**，容易撞 `max_tokens` 被截断。

我们利用 3D 任务的结构性质，把答案压缩成**高度向量**：

```
H = (h_1, h_2, ..., h_k),   h_i ∈ {1,2,3}
```

其中 `h_i` 是第 `i` 根占用柱子的高度（按行扫描顺序）。该表示与合法结构**一一对应**，因此：

- 任何向量都自动满足俯视一致和重力约束 → **VR 天然为 100%**；
- 一条答案只有十几个 token → **不会截断**；
- 模型只需要思考“还有哪个组合没出现”，不需要处理格式。

### 2.2 三个组件

**(1) 表示（representation）**

- 输入：俯视图、占用列列表（固定顺序）、高度范围；
- 输出：一行 `H=(h1,...,hk)`；
- 本地确定性还原成逐层文本，交给官方评分器：

```
第 l 层在位置 p 处为 1  ⇔  h_p ≥ l
```

**(2) 历史（history）**

每次 prompt 列出已经找到的向量：

```
Height vectors already found:
H=(1,1,1), H=(1,1,2), H=(2,1,1)
Your answer must be different from every vector above.
```

模型据此避开重复，把预算用于未出现的组合。

**(3) 引导（guidance）**

在“不要重复”之外，可选一个**坐标约束**作为动作：

- `free`：不加坐标约束，只靠历史去重；
- `coord_fixed`：第 1 轮不加约束，之后按轮次轮流要求“至少在坐标 1、2、…、k 上不同”；
- `coord_random`：随机选一个坐标（供 C 的学习策略做对照）。

C 后续可以在这个动作空间上学习“下一次该要求哪一维不同”。

### 2.3 流程图

```
俯视图 + 已找到的向量
        │
        ▼
  构造 prompt（表示 + 历史 + 可选坐标约束）
        │
        ▼
  调用 LLM（temperature 0.7, top_p 0.8, max_tokens 512, reasoning off）
        │
        ▼
  解析 H=(h1,...,hk)  ──失败──► 该 slot 记空（仍占预算）
        │
        ▼
  向量 → 逐层文本（确定性转换）→ 写入 predictions.jsonl
        │
        ▼
  官方 score_predictions.py 评分（VR / NR / RR）
```

### 2.4 代码结构

| 文件 | 作用 |
|---|---|
| `b/representation.py` | 题目加载、`向量 ↔ 逐层文本`、输出解析 |
| `b/policy.py` | `free` / `coord_fixed` / `coord_random` 的 prompt 构造 |
| `b/run.py` | 批量生成、API 调用、逐轮 trace、输出评分 JSONL |
| `b/compare.py` | 基线 vs 各策略、按两列/三列/整体汇总 |
| `b/check_bijection.py` | 验证 `H_O = {1,2,3}^k` 与 ground truth 一致 |
| `b/check_oracle.py` | 验证“正确答案 → 评分器”可得 100% |
| `b/traces/` | 每题逐轮记录（prompt/回答/向量/是否新解/token/费用/provider） |

---

## 3. 实验设置

| 项 | 值 |
|---|---|
| 模型 | `deepseek/deepseek-v4-flash-0731`（OpenRouter） |
| 解码 | temperature 0.7，top_p 0.8，max_tokens 512，`reasoning.enabled=false` |
| 预算 | 每题 `N = |H_O|`（两列 9，三列 27） |
| 数据 | 验证 12 题（6 两列 + 6 三列），测试 24 题（12 + 12），冻结 split |
| 评分器 | 分支自带 `score_predictions.py`（上游 3D 官方实现，含全零层修复） |
| 基线 | 官方逐层 prompt + 历史（仓库 `results/` 已有 trace） |

---

## 4. 结果

### 4.1 主表

| 集合 | 方法 | 两列 RR | 三列 RR | 整体 RR | parse | VR |
|---|---|---:|---:|---:|---:|---:|
| 验证 | 官方提示基线 | 61.11% | 41.98% | **51.54%** | 95.99% | 77.78% |
| 验证 | **高度向量 free** | 100.00% | 100.00% | **100.00%** | 100% | 100% |
| 验证 | 高度向量 + coord_fixed | 100.00% | 94.44% | **97.22%** | 100% | 100% |
| 测试 | 官方提示基线 | 62.96% | 44.14% | **53.55%** | 95.52% | 79.48% |
| 测试 | **高度向量 free** | 100.00% | 100.00% | **100.00%** | 100% | 100% |
| 测试 | 高度向量 + coord_fixed | 100.00% | 97.53% | **98.77%** | 100% | 100% |

### 4.2 成本与稳定性

| 指标 | 基线 | 高度向量 free |
|---|---:|---:|
| 调用次数（验证+测试） | 648 | 648 |
| 总费用 | $0.1176 | **$0.0177** |
| 平均每条费用 | $0.00018 | **$0.000027** |
| 撞 512 token 上限 | 138 条（21.3%） | **0 条** |
| 解析失败 | 约 4.5% | **0 条** |
| 重复（同向量） | 未统计 | **0 / 648** |
| 平均每题 RR | 0.529 | **1.000** |

### 4.3 关于“唯一失败”的更正

初版结果里测试集有一题 26/27（`g3_h3_b3_079`），看起来像模型在最后一轮重复。检查原始回答后发现是**解析器 bug**，不是模型失败：

- 模型最后一轮先列了几个候选：`H=(1,1,1) is already listed ... Try H=(3,2,3)`，最后一行才给出答案 `H=(3,2,3)`；
- 旧解析器取回复中**第一个** `H=(...)`，于是把 `H=(1,1,1)` 当成答案，误判为重复；
- 修复为“取最后一个完整向量”后重新评分：**648 次生成 0 次重复，验证/测试 RR 都是 100.00%**。

教训：自由文本输出必须取“最终答案”，不能取推理过程里提到的候选。修复在 `b/representation.py:parse_vec`，离线重放脚本是 `repro/rebuild_predictions.py`。

### 4.4 与固定引导的对比

- `free` 优于 `coord_fixed`（验证 +2.78，测试 +1.23 个点）；
- 原因：强制某一维不同，可能把模型推到其他已出现的组合上，浪费预算；
- 结论：引导强度应依赖当前覆盖情况，而不是固定轮流。

---

## 5. 正确性验证

1. **双射检查**：全部 120 题，`{1,2,3}^k` 与 ground truth 完全一致（无多无少）。
2. **Oracle 检查**：把全部正确向量转文本后评分，parse/VR/NR/RR 均为 **1.0**，证明转换与评分格式无误。
3. **基线复算**：用 `score_predictions.py` 重算仓库 trace，验证/测试各项与 `summary.json` 一致（最大差 0.005，仅四舍五入误差）。

---

## 6. 复现

```powershell
cd "D:\Machines Learning Final\hypocover-deepseek"
$py = "D:\Machines Learning Final\hypocover\.venv\Scripts\python.exe"

# 免费检查
& $py b\check_bijection.py
& $py b\check_oracle.py
& $py score_predictions.py validation predictions-validation-oracle.jsonl --output validation-score-oracle.json

# 生成（验证约 $0.006，测试约 $0.012）
& $py -m b.run --split validation --policy free
& $py -m b.run --split test --policy free

# 评分与汇总
& $py score_predictions.py validation predictions-validation-free.jsonl --output validation-score-free.json
& $py score_predictions.py test predictions-test-free.jsonl --output test-score-free.json
& $py b\compare.py
```

---

## 7. 通俗解释（给初学者）

把模型想象成学生，题目是“把 9 种堆法都写出来”：

- **基线**：让学生画图 + 写小作文。学生知道答案，但卷面写不下（超字数）、格式写错、第一层画成全 0，被判无效。**能力没问题，是答题方式拖累。**
- **我们**：把答题卡换成“填几个数字”。学生只要想“还有哪个组合没填”，然后写 `H=(2,1,3)`。卷面不会再错，省下的时间全用来找新组合。

一句话：

> **不是模型变聪明了，而是把问答题改成了填空题，而且填空题的答案空间里没有错误选项。**

---

## 8. 局限与下一步

1. 验证/测试只有 12/24 题，单条轨迹，方差尚未充分估计；建议关键结论重复 2–3 次。
2. `free` 已经满分（验证/测试都 100%），**C 的学习策略在当前设置没有任何提升空间**；若要研究策略，需要加难：
   - 更大网格 / 更高上限 / 更多占用列；
   - 或预算小于 |H_O|（如 27 个答案只给 10 次机会），此时“先问哪个”才重要。
3. 本文结果是**表示层面**的贡献；与基线的差异来自输出格式，不是模型或预算，报告中应明确说明。
4. 不同 provider 存在服务端方差；trace 已记录 provider，建议 C 固定 provider 或分层分析。
5. 高度向量只适用于“每列独立 + 重力”的 3D 任务；causal/boolean 需要各自的表示。

---

## 9. 文件与数据

- 结果文件：`predictions-{validation,test}-{free,coord_fixed}.jsonl`
- 评分文件：`validation-score-free.json`、`test-score-free.json` 等
- 逐题 trace：`b/traces/free/`、`b/traces/coord_fixed/`
- 基线 trace 与 `summary.json`：分支根目录
- 说明文档：`REPORT_B.md`（含更详细的背景与名词表）
