#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python benchmarks/benchmark_accuracy_flops.py \
  --experiment M \
  --device cuda \
  --dtype float64 \
  --m-values 8 16 32 64 128 \
  --fixed-t-for-m 3000 \
  --rff-sample-size 1000 \
  --rff-sample-size-qb 5000 \
  --flops-rff-sample-size 512 \
  --optimizer-lr 1e-2 \
  --analytic-compile 0 \
  --analytic-compile-mode reduce-overhead \
  --output-root outputs/benchmarks \
  "$@"
