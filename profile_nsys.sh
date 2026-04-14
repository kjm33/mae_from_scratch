#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME=${1:?"Usage: $0 <profile_name>"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NSYS_DIR="$SCRIPT_DIR/nsys"
mkdir -p "$NSYS_DIR"

nsys profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --cuda-graph-trace=node \
    --sample=none \
    --cpuctxsw=none \
    --output="$NSYS_DIR/$PROFILE_NAME" \
    --force-overwrite=true \
    python "$SCRIPT_DIR/train.py"
