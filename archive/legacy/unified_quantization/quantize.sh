#!/usr/bin/env bash
# MySVD-style experiment matrix: calibration -> quantization -> saved artifact.
set -euo pipefail
cd "$(dirname "$0")/.."

QUANT_PYTHON="${QUANT_PYTHON:-python}"
QUANT_RUNNER="${QUANT_RUNNER:-sbatch}" # sbatch, local, or dry-run
QUANT_MODEL_NAME="${QUANT_MODEL_NAME:-Poseidon-L}"
QUANT_MODEL_ROOT="${QUANT_MODEL_ROOT:-checkpoints}"
QUANT_DATA_ROOT="${QUANT_DATA_ROOT:-poseidon_main/camlab-ethz/down_streams}"
QUANT_OUTPUT_ROOT="${QUANT_OUTPUT_ROOT:-quantization_artifacts}"
QUANT_METHODS="${QUANT_METHODS:-sapq ppq brecq uniform dynamic static repq ptq4vit}"
QUANT_TRAJECTORIES="${QUANT_TRAJECTORIES:-64}"
QUANT_RANDOM_CALIBRATION="${QUANT_RANDOM_CALIBRATION:-n}"
QUANT_CALIBRATION_SEEDS="${QUANT_CALIBRATION_SEEDS:-0 1 42 2026 3407}"
QUANT_WEIGHT_BITS="${QUANT_WEIGHT_BITS:-8}"
QUANT_ACTIVATION_BITS="${QUANT_ACTIVATION_BITS:-32}"
QUANT_INIT_BITS="${QUANT_INIT_BITS:-9}"
QUANT_TARGET_BITS="${QUANT_TARGET_BITS:-8}"
QUANT_SEED="${QUANT_SEED:-0}"
QUANT_DEVICE="${QUANT_DEVICE:-cuda}"

# dataset | tag | maximum calibration time (MySVD uses all pairs, including t=t).
CONFIG_LIST=(
    "fluids.incompressible.PiecewiseConstants|NS-PwC|20"
    "fluids.incompressible.VortexSheet|NS-SVS|20"
    "fluids.incompressible.BrownianBridge|NS-BB|20"
    "fluids.compressible.RiemannKelvinHelmholtz|CE-RPUI|20"
    "wave.Gaussians|Wave-Gauss|15"
    "wave.Layer|Wave-Layer|20"
)
if [[ "$QUANT_RANDOM_CALIBRATION" == "n" ]]; then
    QUANT_CALIBRATION_SEEDS="0" # Prefix selection does not depend on seed.
fi
mkdir -p logs/quantization
for config in "${CONFIG_LIST[@]}"; do
    IFS='|' read -r dataset tag final_time <<< "$config"
    for method in $QUANT_METHODS; do
        for calibration_seed in $QUANT_CALIBRATION_SEEDS; do
            selection="fixed"
            if [[ "$QUANT_RANDOM_CALIBRATION" == "y" ]]; then selection="random_seed${calibration_seed}"; fi
            run_id="${tag}_${method}_traj${QUANT_TRAJECTORIES}_${selection}_ftime${final_time}_w${QUANT_WEIGHT_BITS}a${QUANT_ACTIVATION_BITS}_init${QUANT_INIT_BITS}_target${QUANT_TARGET_BITS}_seed${QUANT_SEED}"
            model_path="$QUANT_MODEL_ROOT/$QUANT_MODEL_NAME/no_phyloss/$dataset/origin/none"
            save_path="$QUANT_OUTPUT_ROOT/$QUANT_MODEL_NAME/$dataset/$method/$run_id"
            command=("$QUANT_PYTHON" -u -m quantization.quantize
                --method "$method" --model_name "$QUANT_MODEL_NAME"
                --model_path "$model_path" --dataset "$dataset" --data_path "$QUANT_DATA_ROOT"
                --traj_num "$QUANT_TRAJECTORIES" --random_calibration "$QUANT_RANDOM_CALIBRATION"
                --calibration_seed "$calibration_seed" --final_time "$final_time"
                --weight_bits "$QUANT_WEIGHT_BITS" --activation_bits "$QUANT_ACTIVATION_BITS"
                --init_bits "$QUANT_INIT_BITS" --target_bits "$QUANT_TARGET_BITS"
                --seed "$QUANT_SEED" --DEV "$QUANT_DEVICE" --save_path "$save_path" "$@")
            case "$QUANT_RUNNER" in
                local) "${command[@]}" ;;
                dry-run) printf '%q ' "${command[@]}"; printf '\n' ;;
                sbatch)
                    printf -v wrapped '%q ' "${command[@]}"
                    sbatch --job-name="q_${tag}_${method}" --gpus=1 --time=24:00:00 \
                        --chdir="$PWD" --output="logs/quantization/${run_id}-%j.out" \
                        --wrap="$wrapped"
                    ;;
                *) echo "QUANT_RUNNER must be sbatch, local, or dry-run" >&2; exit 2 ;;
            esac
        done
    done
done
