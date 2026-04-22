#!/usr/bin/env bash
# Phase 1 — broad scan across all optimizers.
# 30 trials × 8 epochs each. Results saved to hpo.db.

set -euo pipefail
cd "$(dirname "$0")/.."

python hpo.py \
  --n-trials 30 \
  --n-epochs  8 \
  --device    1 \
  --study-name mae_phase1 \
  --storage   sqlite:///hpo.db
