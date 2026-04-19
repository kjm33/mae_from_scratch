#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME=${1:?"Usage: $0 <profile_name>"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NCU_DIR="$SCRIPT_DIR/ncu"
mkdir -p "$NCU_DIR"

# Uses profile_step.py instead of train.py — a minimal script with no DALI, no scheduler,
# just WARMUP_STEPS warmup steps then one captured step (BATCH_SIZE=256).
#
# --replay-mode kernel
#   Works because BATCH_SIZE=256 keeps GPU memory small (~1 GB), so per-kernel memory
#   backup to RAM is fast. Use --replay-mode application only when GPU memory is large
#   (e.g. full training batch of 5120+).
#
# --set basic     213 metrics, fast
# --set detailed  ~1000 metrics, good balance
# --set roofline  6679 metrics, full hierarchical roofline (slow)
# --set full      8051 metrics (very slow)

ncu \
    --target-processes all \
    --replay-mode kernel \
    --set detailed \
    -o "$NCU_DIR/$PROFILE_NAME" \
    -f \
    python "$SCRIPT_DIR/profile_step.py"

echo ""
echo "Report saved to: $NCU_DIR/$PROFILE_NAME.ncu-rep"
echo "Open with: ncu-ui $NCU_DIR/$PROFILE_NAME.ncu-rep"
