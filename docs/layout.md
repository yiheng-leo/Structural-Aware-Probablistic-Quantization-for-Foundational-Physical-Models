# 目录与入口

本次整理保持迁移前 ritchie 的量化算法、网格、损失、采样、重构和训练顺序。
校准选择继续与 MySVD 相同。每种方法只有一份活跃实现；历史材料不被生产代码导入。

## 方法目录

| 目录 | 唯一启动模块 | 算法文件（另有 __init__.py 与已有许可证） |
|---|---|---|
| UniformQ | `UniformQ.uniformq` | quantize.py |
| DynamicQ | `DynamicQ.dynamicq` | quantize.py |
| StaticQ | `StaticQ.staticq` | quantize.py |
| PPQ | `PPQ.ppq` | quantize.py、loss.py |
| SAPQ | `SAPQ.sapq` | quantize.py、loss.py、sensitivity.py、reconstruction.py |
| BRECQ | `BRECQ.brecq` | quantize.py、quantizers.py、poseidon.py、reconstruction.py |
| RepQ | `RepQ.repq` | quantize.py |
| PTQ4ViT | `PTQ4ViT.ptq4vit` | quantize.py、hessian.py、search.py |

启动文件只调用 `quantization.runner.method_main`，量化文件统一暴露：

```python
quantize(model, calibration_loader, args, device, work_dir) -> QuantizationResult
```

方法使用传入的数据和模型，返回模型、量化状态及方法元数据，不构造测试集或保存最终评估结果。

PPQ 用 `--variant global|layerwise`；SAPQ 用 `--variant global|network|block|layerwise`。
SAPQ global 联合更新全网并观察最终输出；network 逐块更新但观察最终输出；
block 观察块输出；layerwise 观察层输出。名称只选择已有实现，不改变这些实现的数学定义。
raw/normalized、先验和敏感度指标均是参数，不再产生额外入口。

## 公共与评估

- `quantization/runner.py`：配置、模型加载、按需构造校准集、方法调用、保存。
- `quantization/calibration.py`：MySVD 轨迹选择、时间对、数据元信息。
- `quantization/common.py`：结果类型、前向协议、设备和选层工具。
- `quantization/layers.py`：稳定的推理模块与量化网格基础函数。
- `quantization/cache.py`、`observers.py`：磁盘校准快照和共用范围统计。
- `quantization/probabilistic.py`：公共数学与 functional 权重原语，不包含方法分发和 epoch 循环。
- `quantization/artifacts.py`、`compat.py`：模型保存与已知旧类路径的加载兼容。
- `evaluation/evaluate.py`：独立评估入口，默认使用 SAFE-SVD 协议。
- `evaluation/poseidon/`、`reporting.py`：全部测试时间对、ScOT Trainer、PDE 物理指标及原模型对照报告。
- `evaluation/rollout.py`、`metrics.py`：显式 `--protocol endpoint` 的 AR 前向及指标。

正式入口：

```bash
python -m PPQ.ppq --config configs/ppq/default.json --model_path ... --data_path ...
python -m SAPQ.sapq --variant block --model_path ... --data_path ...
python -m RepQ.repq --config configs/repq/default.json --model_path ... --data_path ...
python -m evaluation.evaluate --model_path ... --dataset ... --data_path ...
```

JSON 配置提供默认值，显式命令行参数覆盖文件配置。显式 method 与 variant 冲突会报错；
方法入口不允许跨方法调用。RepQ 未显式设置时使用 A8，其余方法使用 A32。
W8A8 模板明确设置激活位宽，不能把不同 W/A 设置当成同一实验。

FP、UniformQ、DynamicQ，以及 A32 的 StaticQ 不读取离线校准数据；其他方法按需求使用同一校准协议。
为了评估 FP，可直接评估原始完整模型文件；需要统一导出时用
`python -m quantization.runner --method fp --model_path ... --save_path ...`。

## 旧路径映射

| 旧内容 | 新位置 |
|---|---|
| `quantization/baselines.py` 中各算法 | 对应方法的 quantize.py 和必要内核 |
| `quantization/probabilistic.py` 中 PPQ/SAPQ 调度 | PPQ/quantize.py、SAPQ/quantize.py |
| 多个 run_ppq / run_sapq / run_poseidon 启动文件 | 各方法单入口；旧文件在 archive/legacy/ |
| SAPQ sensitivity / layerwise cache 等 | SAPQ/sensitivity.py、reconstruction.py |
| BRECQ/quant 中 Poseidon 活跃内核 | BRECQ/quantizers.py、poseidon.py、reconstruction.py |
| RepQ-ViT 下 Poseidon 适配 | RepQ/ |
| `quantization/evaluate.py` | evaluation/evaluate.py、rollout.py、metrics.py |
| 根目录重复评估、梯度调试、旧绘图 | archive/legacy/root/，方法相关历史材料在相应归档 |
| 根目录旧 launchers | archive/legacy/launchers/ |
| 正式矩阵／Slurm | scripts/quantize.sh、evaluate.sh、run_method.sh、slurm/ |
| 独立检查／绘图／下载工具 | tools/diagnostics/、plotting/、data/ |
| 上游分类、检测、ImageNet 程序 | reference/third_party/ |
| VICON 专用流程 | workflows/vicon/ |
| 旧实验产物 | archive/results/ |

`scOT/`、`physical/` 保留路径，它们是模型运行及反序列化依赖。
`train_file/` 和现有 finetuning YAML 保持原路径；量化配置新增于 configs/<method>/。
外部模型／数据软链接保留，不复制集群数据。

## 兼容范围

保留三处小兼容入口：

- `quantization.quantize` 转发公共 runner，供旧自动化命令使用。
- `quantization.evaluate` 转发独立评估模块。
- `quantization.baselines.QuantizedLinear` 重导出新的 layers 类，供已保存模型直接加载。

`artifacts.load_quantized_artifact` 还能解析旧 BRECQ 的 QuantModule、AdaRound、
BaseQuantBlock、Poseidon 包装及 VICON 包装类路径；新模型不依赖 archive/reference。
不承诺任意旧脚本的 `__main__` 自定义类或只保存步长的历史文件可直接作为完整模型加载。
历史 ImageNet 类和整数后端不映射成新 Poseidon 类。

## 验证

`python -m pytest` 只收集 tests/。检查覆盖真实小型 ScOT 的 13 条路线、独立进程保存回载、
统一评估、CLI/config/variant 优先级、无需校准的方法，以及迁移前 QDQ/BRECQ artifact 兼容。
旧 artifact 额外对照可通过 QUANTIZATION_LEGACY_ARTIFACT 和
QUANTIZATION_LEGACY_BRECQ_ARTIFACT 指定 fixture 目录，缺少外部 fixture 时跳过对应检查。

历史源码与输出归档保留；旧代码依赖当时的工作目录、模型、缓存和环境，不作为第二套正式运行入口。
