#!/usr/bin/env bash
# Evaluate one completed artifact; does not recalibrate or quantize.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${QUANT_PYTHON:-python}" -m quantization.evaluate "$@"
