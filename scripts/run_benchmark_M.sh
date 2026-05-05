#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python benchmarks/benchmark_scaling.py \
  --experiment M \
  --mode all \
  --device cuda \
  --dtype float64 \
  --m-values 8 16 32 64 128 \
  --fixed-t-for-m 159 \
  --num-observations 256 \
  --rff-sample-size 1000 \
  --warmup 5 \
  --repeats 20 \
  --analytic-compile 0 \
  --analytic-compile-mode reduce-overhead \
  --output-root outputs/benchmarks \
  --figure-dir outputs/benchmarks/figures \
  "$@"
