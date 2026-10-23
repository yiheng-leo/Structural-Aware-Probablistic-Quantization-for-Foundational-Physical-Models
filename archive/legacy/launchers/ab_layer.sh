#!/usr/bin/env bash
# Run each prior under the same complete-trajectory calibration protocol.
set -euo pipefail
cd "$(dirname "$0")"
for prior in ppq block_no_sens block_sens; do
    "${QUANT_PYTHON:-python}" -u -m quantization.quantize --method sapq-layerwise --sensitivity_normalization normalized --prior_mode "$prior" "$@"
done
