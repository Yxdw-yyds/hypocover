#!/bin/bash
# wait for off-peak (18:00 local, Mon-Fri peak ends), then resume the DeepSeek protocol
cd /d/machinelearingproject/work/hypocover-projection
while [ "$(date +%H)" -lt 0 ] && [ "$(date +%u)" -le 5 ]; do sleep 30; done
echo "start $(date +%H:%M)" >> c4/results_ds/stdout.log
PYTHONIOENCODING=utf-8 python -B -m c4.run_all --backend deepseek --model deepseek-flash --price-in 0.30 --price-hit 0.006 --price-out 1.20 --ledger c4/spend_ledger_ds.jsonl --out-root c4/layered_ds --results-dir c4/results_ds --policy-file c4/policy_ds.json --total-cap 13.0 --max-cost 4.0 >> c4/results_ds/stdout.log 2>&1
