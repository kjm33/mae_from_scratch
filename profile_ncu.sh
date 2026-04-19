#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME=${1:?"Usage: $0 <profile_name>"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NCU_DIR="$SCRIPT_DIR/ncu"
mkdir -p "$NCU_DIR"

# TORCHDYNAMO_DISABLE=1
#   torch.compile(mode="max-autotune") captures the training step into CUDA graphs.
#   NCU's kernel-replay mode cannot profile kernels inside a CUDA graph blob — it needs
#   to replay each kernel individually.  Disabling dynamo makes every kernel visible.
#
# --replay-mode kernel
#   Replay each kernel in isolation to collect hardware counters.  Fastest option;
#   works correctly once CUDA graphs are out of the way.
#
# --set detailed
#   Sections: ComputeWorkloadAnalysis, MemoryWorkloadAnalysis (+Chart), LaunchStats,
#   Occupancy, SourceCounters, SpeedOfLight (+RooflineChart), Tile, WorkloadDistribution.
#   ~1000 metrics — good balance between depth and collection time.
#   Alternatives:
#     --set basic     213 metrics, quick sanity check
#     --set roofline  6679 metrics, full hierarchical roofline charts (slow)
#     --set full      8051 metrics, everything (very slow)
#
# --launch-skip / --launch-count
#   Skip the first 1000 kernel launches (covers DALI pipeline startup + a few train
#   steps to reach steady-state without compile warmup), then capture ~600 launches
#   (~one full forward+backward+optimizer step at eager speed).
#   Tune these if the captured range misses the step you want.
#
# --nvtx / --nvtx-include
#   PyTorch emits NVTX ranges for torch.profiler.record_function annotations.
#   Uncomment --nvtx-include to restrict capture to a single named range, e.g.:
#     --nvtx --nvtx-include "forward"   (encoder forward only)
#     --nvtx --nvtx-include "backward"  (backward pass only)
#   This is more precise than skip/count when NVTX markers land on GPU correctly.

TORCHDYNAMO_DISABLE=1 ncu \
    --target-processes all \
    --replay-mode kernel \
    --set basic \
    --launch-skip 1000 \
    --launch-count 600 \
    -o "$NCU_DIR/$PROFILE_NAME" \
    -f \
    python "$SCRIPT_DIR/train.py"

echo ""
echo "Report saved to: $NCU_DIR/$PROFILE_NAME.ncu-rep"
echo "Open with: ncu-ui $NCU_DIR/$PROFILE_NAME.ncu-rep"
