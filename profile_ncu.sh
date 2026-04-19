#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME=${1:?"Usage: $0 <profile_name>"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NCU_DIR="$SCRIPT_DIR/ncu"
mkdir -p "$NCU_DIR"

# TORCHDYNAMO_DISABLE=1
#   torch.compile captures the training step into CUDA graphs. NCU cannot profile
#   individual kernels inside a graph blob — disabling dynamo makes every kernel visible.
#
# --replay-mode application
#   Replays the entire application from scratch for each metric collection pass, instead
#   of saving/restoring GPU memory per kernel (--replay-mode kernel). Required when GPU
#   memory is large (e.g. big batches): kernel replay backs up device memory to system
#   RAM and overflows, causing LaunchFailed. Slower per-pass but the only mode that works
#   at scale.
#
# --set basic
#   213 metrics — minimises the number of application replays needed.
#   Alternatives (more replays, larger report):
#     --set detailed   ~1000 metrics
#     --set roofline   6679 metrics, full hierarchical roofline charts
#     --set full       8051 metrics
#
# --launch-skip / --launch-count
#   Skip the first 1000 kernel launches (DALI startup + a few eager steps), then
#   capture ~600 launches (~one full forward+backward+optimizer step).

TORCHDYNAMO_DISABLE=1 ncu \
    --target-processes all \
    --replay-mode application \
    --set basic \
    --launch-skip 1000 \
    --launch-count 600 \
    -o "$NCU_DIR/$PROFILE_NAME" \
    -f \
    python "$SCRIPT_DIR/train.py"

echo ""
echo "Report saved to: $NCU_DIR/$PROFILE_NAME.ncu-rep"
echo "Open with: ncu-ui $NCU_DIR/$PROFILE_NAME.ncu-rep"
