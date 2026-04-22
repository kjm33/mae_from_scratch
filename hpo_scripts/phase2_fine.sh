#!/usr/bin/env bash
# Phase 2 — fine search locked to the winning optimizer from phase 1.
# Usage: ./phase2_fine.sh adamw
#
# Replace "adamw" with the optimizer printed at the end of phase 1.
# Valid choices: adamw adam adabelief radam lion sgd madgrad

set -euo pipefail
cd "$(dirname "$0")/.."

OPTIMIZER="${1:?Usage: $0 <optimizer>  e.g. $0 adamw}"

python hpo.py \
  --n-trials  60 \
  --n-epochs  15 \
  --device     1 \
  --optimizer "$OPTIMIZER" \
  --study-name mae_phase2 \
  --storage   sqlite:///hpo.db
