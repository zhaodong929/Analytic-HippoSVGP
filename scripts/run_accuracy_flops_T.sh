#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python benchmarks/benchmark_accuracy_flops.py \
  --experiment T \
  --device cuda \
  --dtype float64 \
  --t-values 100 300 1000 3000 7000 \
  --fixed-m-for-t 64 \
  --rff-sample-size 1000 \
  --train-iters 300 \
  --optimizer-lr 1e-2 \
  --output-root outputs/benchmarks \
  "$@"
