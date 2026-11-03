# 数据与诊断工具

量化和评估入口位于 `scripts/`。本目录保存独立的数据准备、结构检查和历史结果绘图工具。

## 数据准备：`data/`

各脚本保留原来的数据集选择、下载来源和输出格式，没有合并不同历史实现。

| 文件 | 原位置与用途 |
| --- | --- |
| `all_datasets.py` | 原根目录脚本；下载 14 个下游数据集到 `poseidon_main/camlab-ethz/down_streams/` |
| `assemble_all.py` | 原根目录脚本；调用上述数据集自带的 `assemble_data.py`，保留 `.nc` 输出约定 |
| `download_selected.py` / `.sh` | 原根目录脚本；下载选定的 5 个数据集，保留原绝对 `DATA_ROOT` |
| `assemble_selected.py` / `.sh` | 原根目录脚本；组装选定数据集，保留原绝对 `DATA_ROOT` 和 `.nc` 格式 |
| `download.py` | 原 `setting_environment/download.py`；下载 NS 数据集及 W&B 模型，保留其原流程 |
| `download_all_datasets.sh` | 原 `setting_environment/` 脚本；通过 Hugging Face CLI 下载其 20 个数据集列表到 `datasets/` |
| `assemble_all_datasets.sh` | 原 `setting_environment/` 脚本；组装 `datasets/` 子目录，保留 `.h5` 输出约定 |

例如，从仓库根目录运行：

```bash
python tools/data/all_datasets.py
python tools/data/assemble_all.py
QUANT_PYTHON=/path/to/python bash tools/data/download_selected.sh
QUANT_PYTHON=/path/to/python bash tools/data/assemble_selected.sh
```

先检查各实现中的数据路径、列表与所需 Hugging Face/W&B 登录状态。迁移没有下载数据、访问账号或改动 `dataset`、`models` 等现有绝对软链接。两个 selected shell 使用调用环境中的 Python；Slurm 提交请从仓库根目录执行，或设置 `QUANT_REPO_ROOT`。

## 为 main 原版 global SAPQ 准备缓存

`prepare_main_sapq.py` 是给 **main 原始代码** 使用的独立工具。将该脚本复制到服务器的 main 工作目录再执行；它不会切换分支。ritchie 的正式量化入口不依赖这些旧缓存。

从 main 仓库根目录执行，以下环境变量语法适用于 tcsh/csh：

```tcsh
setenv PPQ_MODEL_PATH /cs/research/vecg/EasyPDE/model-store/poseidon/fluids.incompressible.PiecewiseConstants
setenv PPQ_DATA_PATH poseidon/camlab-ethz/down_streams
setenv PPQ_DATASET_NAME fluids.incompressible.PiecewiseConstants
setenv SAPQ_PRIOR_MODE block_sens
python -u tools/prepare_main_sapq.py --cache-root /cs/research/vecg/EasyPDE/sapq-input-cache --device cuda
python -u -m SAPQ.run_sapq_network_global
```

该工具补齐层名 PT、固定校准批次及 Sobolev 敏感度。新校准集遵循 main 默认数据设置：训练轨迹容量 2048、shuffle、512 个 batch、每批 2 个样本；随机种子默认为 0，可用 `--seed` 指定。缺少历史缓存时，这是新抽取的 1024 个样本，不能恢复作者当年的具体抽样。`--frozen-only` 只准备层名及校准批次。

新缓存目录放在 `--cache-root` 下，通过软链接提供 main 所需路径。已有可用目录及缓存会复用；失效的目录软链接会重建，链接原目标的数据不会删除。它同时提供 global 读取的顶层校准文件和旧生成器使用的 `<数据目录名>-calib/` 路径，解决两处路径不一致。敏感度沿用 main 的量化器、Sobolev 损失、梯度归约及归一化，只将逐块重复反向计算合并成每批一次梯度计算；不修改 SAPQ 优化配置。

## 结构检查与绘图

`diagnostics/inspect_layers/inspect_layers.py` 及其保存的 B/L 层清单从原 `inspect_layers/` 整体迁入。脚本按新位置寻找仓库，默认检查 `models/NS-PwC-L`；更换模型时修改脚本中的配置。它是人工诊断工具，不是量化时必须读取的层缓存。

`plotting/SAPQ/` 的两个独立绘图脚本读取 `archive/results/SAPQ/` 下的历史 CSV/JSON，默认输出到相应历史结果的 `plots/` 目录。运行前检查对应实验数据是否存在。

## 归档范围

- 依赖旧 PPQ/SAPQ trainer 的根目录评估、梯度、绘图和调试脚本保存在 `archive/legacy/root/`，保持原文件名，没有将它们宣称为当前可运行工具。
- `all_previous_useless/` 与 `Original/` 分别完整保存在 `archive/legacy/` 同名目录。
- 原根目录 `ppq_artifact/`、`eval_results_longterm/`、`rollout_figures/`、`zzzzz_plot/` 保存在 `archive/results/root/` 同名目录。
- 历史 Slurm/矩阵入口保存在 `archive/legacy/launchers/`；VICON 的外部仓库和 helper 依赖仍属于历史工作流。

逐文件迁移记录见 `scripts/migration_manifest.json`。当前正式工作流不导入归档代码。
