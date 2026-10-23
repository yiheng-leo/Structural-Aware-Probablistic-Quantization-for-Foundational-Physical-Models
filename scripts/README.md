# 启动与实验矩阵

正式 shell 入口集中在本目录。所有方法名到 Python 模块的映射只维护在 `run_method.sh`；量化结果保存后，使用 `evaluate.sh` 单独评估。

## 单次运行

从仓库根目录执行：

```bash
export QUANT_PYTHON=/path/to/python
bash scripts/run_method.sh sapq --model_path /path/to/checkpoint \
  --dataset fluids.incompressible.PiecewiseConstants --data_path /path/to/data \
  --save_path /path/to/artifact

bash scripts/run_method.sh --dry-run sapq-network --config configs/sapq/network.json
bash scripts/run_method.sh repq --activation_bits 8 --help
```

`--dry-run` 打印实际 Python 命令，不加载模型或数据。`QUANT_PYTHON` 是解释器路径；其余参数按原样传递给 Python CLI，包括 `--config`、`--variant` 和带空格的路径。

支持 `fp`、`uniform`、`dynamic`、`static`、`ppq`、`ppq-layerwise`、`sapq`、`sapq-block`、`sapq-layerwise`、`sapq-network`、`brecq`、`repq`、`ptq4vit`。PPQ/SAPQ 的旧方法名转换为对应目录入口的 `--variant`，数值算法和训练调度不变。`sapq` 对应全网更新与网络输出损失，`sapq-network` 对应逐块更新与网络输出损失，`sapq-block` 对应块内重构。

## 批量矩阵

```bash
QUANT_RUNNER=dry-run QUANT_METHODS='sapq ppq repq' bash scripts/quantize.sh
QUANT_RUNNER=local QUANT_METHODS='ppq-layerwise' bash scripts/quantize.sh --epochs 2
QUANT_RUNNER=sbatch QUANT_PYTHON=/path/to/python bash scripts/quantize.sh
```

保留原六个数据集及其时间范围：NS-PwC、NS-SVS、NS-BB、CE-RPUI、Wave-Gauss、Wave-Layer。默认选择前 64 条轨迹；`QUANT_RANDOM_CALIBRATION=y` 时遍历 `QUANT_CALIBRATION_SEEDS`，固定选择时只运行 seed 0。

主要环境变量：

| 变量 | 默认值或含义 |
| --- | --- |
| `QUANT_RUNNER` | `sbatch`，可选 `local`、`dry-run` |
| `QUANT_METHODS` | `sapq ppq brecq uniform dynamic static repq ptq4vit` |
| `QUANT_MODEL_NAME` / `QUANT_MODEL_ROOT` | `Poseidon-L` / `checkpoints` |
| `QUANT_DATA_ROOT` | `poseidon_main/camlab-ethz/down_streams` |
| `QUANT_OUTPUT_ROOT` | `quantization_artifacts` |
| `QUANT_TRAJECTORIES` | `64` |
| `QUANT_RANDOM_CALIBRATION` | `n` |
| `QUANT_CALIBRATION_SEEDS` | `0 1 42 2026 3407` |
| `QUANT_WEIGHT_BITS` | `8` |
| `QUANT_ACTIVATION_BITS` | 未设置时 RepQ 为 `8`，其余为 `32`；显式设置覆盖该选择 |
| `QUANT_INIT_BITS` / `QUANT_TARGET_BITS` | `9` / `8` |
| `QUANT_SEED` / `QUANT_DEVICE` | `0` / `cuda` |

命令行附加参数传递到每次运行，优先级由 Python CLI 保持。单独调用 `run_method.sh repq` 时，Python 入口也默认使用 A8；显式 CLI 参数或配置可以覆盖该默认值。

## Slurm 与评估

在仓库根目录提交单次任务：

```bash
sbatch scripts/slurm/quantize.sbatch sapq-block --config /path/to/config.json
bash scripts/evaluate.sh --model_path /path/to/artifact \
  --dataset fluids.incompressible.PiecewiseConstants --data_path /path/to/data
```

Slurm 脚本接收方法名及全部 CLI 参数，再调用 `run_method.sh`。从其他目录提交时设置 `QUANT_REPO_ROOT` 为仓库绝对路径。矩阵提交会自动设置该变量，并把日志写入 `logs/quantization/`。GPU、时限等资源可以用 `sbatch` 参数覆盖。

## 历史入口与迁移记录

原根目录的重复 `.sh`、`.sbatch` 入口，以及原 `quantization/quantize.sh`、`evaluate.sh`，保存在 `archive/legacy/launchers/`。RepQ 原始提交脚本保留在 `scripts/legacy/RepQ-ViT/`；VICON 脚本位于 `workflows/vicon/scripts/legacy/`。这些是历史记录，可能依赖旧目录或外部环境，不属于当前正式启动流程。

`migration_manifest.json` 记录本次根目录、数据工具和入口迁移的每个文件、迁移前后 SHA-256，以及保持原样的根目录软链接。数据工具说明见 `tools/README.md`。
