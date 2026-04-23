#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python benchmarks/benchmark_scaling.py \
  --experiment T \
  --mode all \
  --device cuda \
  --dtype float64 \
  --t-values 100 300 1000 3000 10000 \
  --fixed-m-for-t 64 \
  --num-observations 256 \
  --rff-sample-size 1000 \
  --warmup 5 \
  --repeats 20 \
  --output-root outputs/benchmarks \
  --figure-dir outputs/benchmarks/figures \
  "$@"
