#!/usr/bin/env bash
# Evaluate a completed artifact without recalibrating or quantizing it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${QUANT_PYTHON:-python}" -m evaluation.evaluate "$@"
