#!/usr/bin/env bash
#SBATCH --job-name=sapq_layer
#SBATCH --gpus=1
#SBATCH --time=24:00:00
set -euo pipefail
# Submit from the repository root; pass checkpoint/data paths as CLI arguments.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
exec "${QUANT_PYTHON:-python}" -u -m quantization.quantize --method sapq-layerwise --sensitivity_normalization normalized "$@"
