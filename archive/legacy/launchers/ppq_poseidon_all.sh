#!/usr/bin/env bash
# Shared MySVD-compatible calibration; quantization only, no inline evaluation.
set -euo pipefail
export QUANT_METHODS="${QUANT_METHODS:-ppq}"
exec bash "$(dirname "$0")/quantization/quantize.sh" "$@"
