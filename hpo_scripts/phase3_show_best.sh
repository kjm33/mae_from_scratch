#!/usr/bin/env bash
# Phase 3 — print the best parameters from phase 2 (or phase 1 if phase 2 hasn't run).
# Copy the printed values into train.py before starting the full training run.

set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'EOF'
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

for study_name in ("mae_phase2", "mae_phase1"):
    try:
        study = optuna.load_study(study_name=study_name, storage="sqlite:///hpo.db")
        t = study.best_trial
        print(f"=== Best trial from {study_name} (val_loss={t.value:.4f}) ===")
        for k, v in t.params.items():
            print(f"  {k:15s}: {v:.6g}" if isinstance(v, float) else f"  {k:15s}: {v}")
        break
    except Exception:
        continue
else:
    print("No completed studies found in hpo.db. Run phase1_broad.sh first.")
EOF
