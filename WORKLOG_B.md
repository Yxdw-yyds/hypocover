# 角色 B 工作日志（2026-11-18 → 11-24）

> 记录做过什么、得到什么、踩过什么坑，方便队友接手。
> 复现细节见 `METHOD_AND_RESULTS.md`，方法说明见 `C_NEXT_STEPS.md`。

---

## 1. 目标与角色变化

- 最初 `TEAM.md` 里 B 负责“方法 A：分区搜索（partition search）的调参和改进”；
- 11/24 小组把仓库切到新分支 `deepseek-3d-baseline-512`，分工改为：
  - A：冻结题单、官方提示基线、评分；
  - **B：高度向量表示、固定引导、历史记录**；
  - C：学习“下一次选什么提示”；
  - D：配对比较、曲线、视频。
- 本日志前半段是旧任务（分区搜索）的探索，后半段是新任务（高度向量）的落地。

---

## 2. 时间线

### 11/18：读仓库与论文

- clone `hypocover`，确认是单 commit 的 scaffold：有 `hypocover/`、`rl/`、`tests/`，但没有 `data/`、没有 `results/`；
- 下载 HypoSpace 论文（PMLR v306），整理指标、难度设置、head-tail 理论（`K + N(1-α)`）；
- 写 `PLAN_B.md`：环境、数据、实验矩阵、风险。

### 11/19：环境与数据

- Windows + Python 3.11，建 `.venv`，`pip install -e ".[dev]"`；
- 发现没有 `uv`、没有可用的 bash/WSL，`scripts/make_datasets.sh` 跑不了，改用 PowerShell 等价命令；
- 生成 3D：tp1=3、tp2=9、tp3=27、large=125 个合法解，全部正确；
- 生成 Boolean d2（basic/extended/full）、causal n4；
- causal n5 全枚举 >45 分钟，改用 `--n-samples`；n6/large 未完成；
- **发现**：Boolean hard 上 `max_tokens=2048` 会把 reasoning 截断，解析率从 100% 掉到 45%，导致 1 小时 0 个 instance、白花约 $0.65。

### 11/19–24：旧方法（分区搜索）的探索

在 3D tp3（5 条实例）上对比：

| 方法 | VR | NR | RR | 成本 |
|---|---:|---:|---:|---:|
| memory（论文协议） | 0.837 | 0.830 | 0.822 | $0.246 |
| partition v0 | 0.874 | 0.874 | 0.867 | $0.233 |
| CovTree v1（新写） | 0.859 | 0.859 | 0.859 | $0.126 |

结论：

- 这个设置上 DeepSeek V4.1 Flash 已经 0.82–0.87，接近天花板；
- 分区方法只比 memory 高约 4 个点，5 条样本内差异不显著；
- CovTree 的调用数/成本约是 memory 的一半（138 vs 206 calls），因为它避免了大量无效重试；
- 3D large 跑到一半崩溃：上游 `Structure3D.normalize()` 把 numpy array 传回构造函数，`if layer:` 报 `ValueError`；新分支已修复为 `(self.layers[0] * 0).tolist()`。

### 11/24：切到小组新分支

- 新分支 `deepseek-3d-baseline-512`（Liyuhan）把仓库换成 3D 基线交接包：
  - 冻结 `data/columns-{2,3}.json`，split 84 / 12 / 24；
  - 官方 prompt 基线 + `score_predictions.py`；
  - 模型 `deepseek/deepseek-v4-flash-0731`，reasoning off，max_tokens 512；
  - 基线：验证 RR 51.54%，测试 RR 53.55%。
- 建 worktree `hypocover-deepseek`（分支 `deepseek-baseline`），复算基线：与 `summary.json` 逐项一致（最大差 0.005）。
- 统计基线 trace：648 条，21.3% 撞 512 token 上限，平均 prompt 1120 tokens、completion 307 tokens。

### 11/24：B 的新方法落地

- 发现高度向量与合法解一一对应：`H_O = {1,2,3}^k`，`k=2/3`；
- `b/check_bijection.py` 验证 120 题全部一致；
- `b/representation.py`：向量 ↔ 逐层文本、解析；
- `b/policy.py`：`free` / `coord_fixed` / `coord_random`；
- `b/run.py`：批量跑题、记录 trace、输出评分 JSONL；
- `b/check_oracle.py`：全部正确答案喂评分器 = 1.0；
- 运行验证 12 题 + 测试 24 题，得到：

| 策略 | 验证 RR | 测试 RR | 成本（验证+测试） |
|---|---:|---:|---:|
| 官方 prompt 基线 | 51.54% | 53.55% | $0.1176 |
| **高度向量 free** | **100.00%** | **99.85%** | **$0.0177** |
| 高度向量 + coord_fixed | 97.22% | 98.77% | $0.0192 |

- 截断：基线 138/648 → 我们 0/648；解析失败：约 4.5% → 0。
- 唯一失败：测试 `g3_h3_b3_079`，27 个找到 26 个，最后一轮重复 `H=(1,1,1)`。

---

## 3. 关键结论

1. **这个 3D 任务的主要瓶颈是输出表示，不是搜索策略**：官方逐层文本导致 21% 截断和约 22% 无效答案；换成高度向量后 VR 天然 100%。
2. **`free`（表示 + 历史）已接近饱和**：验证 100%、测试 99.85%，给 C 的学习策略留下的空间很小。
3. 若要让“学习选提示”有意义，需要**更难或预算更紧**的设置（4×4、高度 4、4 列，或预算小于 |H_O|）。
4. 基线 prompt 与我们的 prompt **不同**：同模型、同解码、同预算、同评分器，但 prompt/输出格式是干预本身；跨论文比较需谨慎。

---

## 4. 踩坑记录

| 问题 | 现象 | 处理 |
|---|---|---|
| Windows 无 bash/uv | `make_datasets.sh` 跑不了 | 改 PowerShell；`.venv\Scripts\python.exe` |
| Boolean hard 解析失败 | 1 小时 0 个 instance | `max_tokens` 2048→8192/16384；或缩短输出 |
| 3D large 崩溃 | `ValueError: truth value of array` | 上游 `normalize()` 的 numpy 层；新分支已修 |
| 预算消耗快 | reasoning 默认开，单题 $0.05 | 关 reasoning、缩 prompt；B 的新设置单题约 $0.001 |
| 旧分支代码被删 | 新分支没有 `hypocover/` | 新代码放 `b/`，旧 covtree 留在 `main`/本地 |

---

## 5. 成本

- 旧方法探索：约 **$3.6**（其中 3D large 崩溃 + Boolean 卡住约 $1.5 无效）；
- 基线复算 + B 新方法：约 **$0.06**（基线 $0.1176 由队友承担；我们的 free/fixed 共约 $0.04）；
- 新方法单题成本约 **$0.001**，验证集 12 题约 **$0.006**。

---

## 6. 产物清单

| 类型 | 文件 |
|---|---|
| 方法代码 | `b/representation.py`、`b/policy.py`、`b/run.py`、`b/compare.py` |
| 检查脚本 | `b/check_bijection.py`、`b/check_oracle.py` |
| 基线复算 | `repro/make_baseline_predictions.py`、`repro/verify_baseline_summary.py`、`repro/diag_traces.py` |
| 预测 | `predictions-{validation,test}-{free,coord_fixed}.jsonl` |
| 评分 | `validation-score-*.json`、`test-score-*.json` |
| 逐题记录 | `b/traces/{free,coord_fixed}/*.jsonl` |
| 文档 | `METHOD_AND_RESULTS.md`、`C_NEXT_STEPS.md`、本文件 |
