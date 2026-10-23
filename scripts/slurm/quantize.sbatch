#!/usr/bin/env bash
#SBATCH --job-name=poseidon_quantize
#SBATCH --gpus=1
#SBATCH --time=24:00:00
set -euo pipefail
# sbatch spools this script: submit from the repository root, or explicitly set
# QUANT_REPO_ROOT. Matrix submissions already export the repository location.
SCRIPT_REPO_ROOT="${QUANT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$SCRIPT_REPO_ROOT"
exec bash "$SCRIPT_REPO_ROOT/scripts/run_method.sh" "$@"
