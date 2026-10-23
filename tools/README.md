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

## 结构检查与绘图

`diagnostics/inspect_layers/inspect_layers.py` 及其保存的 B/L 层清单从原 `inspect_layers/` 整体迁入。脚本按新位置寻找仓库，默认检查 `models/NS-PwC-L`；更换模型时修改脚本中的配置。它是人工诊断工具，不是量化时必须读取的层缓存。

`plotting/SAPQ/` 的两个独立绘图脚本读取 `archive/results/SAPQ/` 下的历史 CSV/JSON，默认输出到相应历史结果的 `plots/` 目录。运行前检查对应实验数据是否存在。

## 归档范围

- 依赖旧 PPQ/SAPQ trainer 的根目录评估、梯度、绘图和调试脚本保存在 `archive/legacy/root/`，保持原文件名，没有将它们宣称为当前可运行工具。
- `all_previous_useless/` 与 `Original/` 分别完整保存在 `archive/legacy/` 同名目录。
- 原根目录 `ppq_artifact/`、`eval_results_longterm/`、`rollout_figures/`、`zzzzz_plot/` 保存在 `archive/results/root/` 同名目录。
- 历史 Slurm/矩阵入口保存在 `archive/legacy/launchers/`；VICON 的外部仓库和 helper 依赖仍属于历史工作流。

逐文件迁移记录见 `scripts/migration_manifest.json`。当前正式工作流不导入归档代码。
