# HypoCover: coverage-aware hypothesis generation on HypoSpace

Course final project (Part 1.2). We try to raise **Recovery** on the
[HypoSpace](https://proceedings.mlr.press/v306/chen26bz.html) benchmark (ICML 2026),
where LLMs must list *all* hypotheses consistent with underdetermined
observations and tend to collapse onto a few of them.

| | Method | Training | Idea |
|---|---|---|---|
| A | **Partition search** (`hypocover/methods/partition.py`) | none, API models | Split the hypothesis space into disjoint cells by structural constraints, query inside cells chosen by a UCB rule, close cells the model declares `EXHAUSTED`, split cells that keep returning duplicates. Oracle-free (the paper's stratified decoding uses the ground-truth complexity distribution). |
| B | **Coverage-GRPO** (`rl/`) | LoRA + GRPO on Qwen3-4B | Reward a new valid hypothesis with `1/m`, where `m` is how many samples in the GRPO group gave the same answer. A group's total reward equals its marginal coverage, so training shifts probability mass from the head of the distribution to its tail. |

## Layout

```
hypocover/
  tasks/        uniform wrapper over the three official domains (causal, 3d, boolean);
                parsing, validators and canonical keys are the upstream code, unchanged
  llm/          OpenAI / OpenRouter / DeepSeek / local vLLM clients + an offline mock
  methods/      direct, memory (= paper protocol), stratified (paper Sec. 6.5), partition (ours)
  metrics.py    VR / NR / RR exactly as upstream, plus RR curves and simple/complex RR
  eval/run.py   run one config, resumable, writes results/<run>/{instances.jsonl,summary.json}
  eval/summarize.py   all results -> one table
rl/             method B: reward, prompt-set builder, TRL GRPO script, configs, launch.sh
scripts/make_datasets.sh   generate every dataset with fixed seeds via the upstream generators
third_party/HypoSpace/     official code, vendored unmodified (MIT), see VENDORED.md
tests/          round-trip checks of the task wrappers, mock run, reward unit test
```

## Quick start

```bash
uv venv --python 3.11 .venv && uv pip install -e ".[dev]"
cp .env.example .env              # add API keys
bash scripts/make_datasets.sh all # writes data/<task>/*.json (causal n6 is slow)
.venv/bin/python -m pytest -q
```

Run a baseline and our method on the same instances:

```bash
.venv/bin/python -m hypocover.eval.run --task 3d --data data/3d/tp3.json --method memory \
    --model openrouter:deepseek/deepseek-v4.1-flash --n-instances 30
.venv/bin/python -m hypocover.eval.run --task 3d --data data/3d/tp3.json --method partition \
    --model openrouter:deepseek/deepseek-v4.1-flash --n-instances 30
.venv/bin/python -m hypocover.eval.summarize
```

`--model mock:zipf` runs everything offline with a simulated peaked generator (no cost).
Other flags: `--budget-mult` (queries = mult × |H_O|, default 1 as in the paper),
`--method-args '{"alloc":"uniform"}'`, `--min-gt` (default 2, skips trivial instances),
`--workers` (parallel instances).

## Evaluation protocol (keep fixed for every method)

* Budget N = |H_O| hypothesis-producing queries per instance; parse failures are
  retried as upstream does (cost tokens, not budget). Partition search counts its
  `EXHAUSTED` answers against the budget too.
* Same instances for every method: `--seed 33550336 --n-instances 30`.
* Report VR / NR / RR and tokens / cost. Two settings, reported separately:
  *no validator* (default) and *validator-in-the-loop* (`use_validator: true`).
* "Write a program that enumerates H_O" solves these toy domains outright. We treat it
  as a reference upper bound, not as a competitor, because real scientific hypothesis
  spaces cannot be enumerated.

## Datasets

| task | level | file | mean \|H_O\| |
|---|---|---|---|
| 3d | L1 / L2 / L3 / large | `data/3d/tp{1,2,3}.json`, `large.json` | 3 / 9 / 27 / 125 (matches paper) |
| causal | L1 / L2 / L3 / large | `data/causal/n{4,5,6}.json`, `large.json` | to be checked against paper |
| boolean | basic / extended / full | `data/boolean/{basic,extended,full}_d{2,3}.json` | depth 2: 4 / 6.5 / 11; paper's hard is ~78, needs config check |

## Known upstream issue (worth reporting)

Some ground-truth Boolean expressions are rejected by the official validator.
`NOT (x OR y)` is listed in H_O, but `_check_operators_in_ast` reads it as `NOR`,
which the `extended` set does not allow. In the `full` set, `_get_ast_depth` counts
`NOT(NOR)` as one level deeper than the generator does. As a result the best possible
RR under the official scorer is 0.985 on `extended_d2` and **0.545 on `full_d2`**
(computed by `tests/test_tasks.py::ceiling`). The paper's Boolean "collapse" at
higher levels is partly this artefact. We report official RR for comparability and
the ceiling next to it.

## Method B on the GPU box (2× RTX 4090)

```bash
bash rl/launch.sh setup          # deps, datasets, data/rl/train.jsonl
bash rl/launch.sh serve          # terminal 1, GPU 0: trl vllm-serve
bash rl/launch.sh train          # terminal 2, GPU 1: GRPO + LoRA
bash rl/launch.sh eval checkpoints/qwen3_4b_cov   # serve adapter, then eval with --model vllm:cov
```

Ablation: `bash rl/launch.sh train --no-share` (reward 1 for any new valid answer).
Train levels: causal n4/n5, 3d tp2, boolean basic/extended d2. Hard and large levels
are held out for evaluation.

## Team

See [TEAM.md](TEAM.md) for roles and the schedule.

## Credits

Benchmark and official code: Chen et al., *HypoSpace*, ICML 2026
([code](https://github.com/CTT-Pavilion/_HypoSpace), MIT).
