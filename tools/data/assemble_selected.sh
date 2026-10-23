#!/usr/bin/env bash
#SBATCH --job-name=assemble_poseidon_data
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/assemble_data-%j.out
#SBATCH --error=logs/assemble_data-%j.err
set -euo pipefail
# For sbatch, submit from the repository root or set QUANT_REPO_ROOT.
DATA_REPO_ROOT="${QUANT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$DATA_REPO_ROOT"
mkdir -p logs
exec "${QUANT_PYTHON:-python}" -u "$DATA_REPO_ROOT/tools/data/assemble_selected.py" "$@"
