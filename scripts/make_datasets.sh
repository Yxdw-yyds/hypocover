#!/usr/bin/env bash
# Generate every dataset we use, with fixed seeds, by calling the official
# (unmodified) upstream generators. Output: data/<task>/<level>.json
#
# Difficulty levels follow the paper (Tables 1-4):
#   causal  L1/L2/L3 = 4/5/6 nodes;  large = hybrid generator, |H_O| up to ~160
#   3d      L1/L2/L3 = 1/2/3 occupied top-view cells on a 3x3 grid, height 3
#           large    = 3 cells, height 5  (|H_O| = 5^3 = 125)
#   boolean L1/L2/L3 = basic / extended / full operator sets
#
# Usage: bash scripts/make_datasets.sh [causal|3d|boolean|all]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-$ROOT/.venv/bin/python}
SEED=33550336
BOOL_DEPTH=${BOOL_DEPTH:-3}
WHAT=${1:-all}
UP=$ROOT/third_party/HypoSpace

gen_causal() {
  mkdir -p data/causal
  cd "$UP/causal"
  for n in 4 5; do
    [ -f "$ROOT/data/causal/n$n.json" ] || \
      "$PY" generate_causal_dataset.py --nodes $n --seed $SEED --output "$ROOT/data/causal/n$n.json"
  done
  # 6 nodes: enumerate all DAGs once but sample observation combinations
  [ -f "$ROOT/data/causal/n6.json" ] || \
    "$PY" generate_causal_dataset.py --nodes 6 --n-samples 3000 --seed $SEED --output "$ROOT/data/causal/n6.json"
  [ -f "$ROOT/data/causal/large.json" ] || \
    "$PY" generate_causal_dataset_for_large.py --nodes 6 --n-observation-sets 100 --seed $SEED --output "$ROOT/data/causal/large.json"
  cd "$ROOT"
}

gen_3d() {
  mkdir -p data/3d
  cd "$UP/3d"
  for b in 1 2 3; do
    [ -f "$ROOT/data/3d/tp$b.json" ] || \
      "$PY" generate_3d_dataset_complete.py --grid-size 3 --max-height 3 --max-blocks $b --fixed --seed $SEED --output "$ROOT/data/3d/tp$b.json"
  done
  [ -f "$ROOT/data/3d/large.json" ] || \
    "$PY" generate_3d_dataset_complete.py --grid-size 3 --max-height 5 --max-blocks 3 --fixed --seed $SEED --output "$ROOT/data/3d/large.json"
  cd "$ROOT"
}

gen_boolean() {
  mkdir -p data/boolean
  cd "$UP/boolean"
  for ops in basic extended full; do
    [ -f "$ROOT/data/boolean/${ops}_d$BOOL_DEPTH.json" ] || \
      "$PY" boolean_dataset.py --operators $ops --max-depth $BOOL_DEPTH --seed $SEED --output "$ROOT/data/boolean/${ops}_d$BOOL_DEPTH.json"
  done
  cd "$ROOT"
}

case $WHAT in
  causal) gen_causal ;;
  3d) gen_3d ;;
  boolean) gen_boolean ;;
  all) gen_3d; gen_boolean; gen_causal ;;
  *) echo "unknown: $WHAT"; exit 1 ;;
esac
