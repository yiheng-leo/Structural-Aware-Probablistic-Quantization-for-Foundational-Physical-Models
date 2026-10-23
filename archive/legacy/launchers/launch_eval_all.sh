#!/usr/bin/env bash
# Usage: bash launch_eval_all.sh --model_path ARTIFACT --dataset DATASET --data_path DATA
# Each invocation evaluates one explicit artifact, avoiding stale/hard-coded step files.
set -euo pipefail
exec bash "$(dirname "$0")/quantization/evaluate.sh" "$@"
