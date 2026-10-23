#!/usr/bin/env bash
# The only shell mapping from method identifiers to Python method launchers.
set -euo pipefail
SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_REPO_ROOT"

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_method.sh [--dry-run] METHOD [quantization arguments...]
METHOD: fp uniform dynamic static ppq ppq-layerwise sapq sapq-block
        sapq-layerwise sapq-network brecq repq ptq4vit
Set QUANT_PYTHON to choose the interpreter. All remaining arguments are forwarded.
USAGE
}
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
method="$1"
shift
case "$method" in
    fp) module=quantization.runner; set -- --method fp "$@" ;;
    uniform) module=UniformQ.uniformq ;;
    dynamic) module=DynamicQ.dynamicq ;;
    static) module=StaticQ.staticq ;;
    ppq) module=PPQ.ppq; set -- --variant global "$@" ;;
    ppq-layerwise) module=PPQ.ppq; set -- --variant layerwise "$@" ;;
    sapq) module=SAPQ.sapq; set -- --variant global "$@" ;;
    sapq-block) module=SAPQ.sapq; set -- --variant block "$@" ;;
    sapq-layerwise) module=SAPQ.sapq; set -- --variant layerwise "$@" ;;
    sapq-network) module=SAPQ.sapq; set -- --variant network "$@" ;;
    brecq) module=BRECQ.brecq ;;
    repq) module=RepQ.repq ;;
    ptq4vit) module=PTQ4ViT.ptq4vit ;;
    *) printf 'Unknown quantization method: %s\n' "$method" >&2; exit 2 ;;
esac
command=("${QUANT_PYTHON:-python}" -u -m "$module" "$@")
if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
