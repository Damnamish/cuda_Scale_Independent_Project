#!/usr/bin/env bash
#
# Runs the batch image filter over the sample dataset (or whatever
# input/output dirs are passed in) and tees the output into a
# timestamped log file, which can be used as the assignment's
# "proof of execution" artifact.
#
# Usage: ./run.sh [input_dir] [output_dir] [blur_radius]
#
set -euo pipefail

INPUT_DIR="${1:-data/input}"
OUTPUT_DIR="${2:-data/output}"
BLUR_RADIUS="${3:-2}"

if [ ! -x ./bin/cuda_batch_filter ]; then
    echo "bin/cuda_batch_filter not found or not executable -- run 'make build' first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="proof_of_execution_$(date +%Y%m%d_%H%M%S).log"

./bin/cuda_batch_filter "$INPUT_DIR" "$OUTPUT_DIR" "$BLUR_RADIUS" | tee "$LOG_FILE"

echo ""
echo "Log written to $LOG_FILE"
