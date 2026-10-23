# Structural-Aware Probabilistic Quantization for Physical Foundation Models

Poseidon 的各量化方法拥有独立入口，共用 MySVD 完整轨迹校准协议、模型保存和独立评估。

| 方法 | 正式启动入口 |
|---|---|
| UniformQ | `python -m UniformQ.uniformq` |
| DynamicQ | `python -m DynamicQ.dynamicq` |
| StaticQ | `python -m StaticQ.staticq` |
| PPQ | `python -m PPQ.ppq --variant global` / `--variant layerwise` |
| SAPQ | `python -m SAPQ.sapq --variant global` / `network` / `block` / `layerwise` |
| BRECQ | `python -m BRECQ.brecq` |
| RepQ | `python -m RepQ.repq` |
| PTQ4ViT | `python -m PTQ4ViT.ptq4vit` |

在仓库根目录运行。每个方法的 `<name>.py` 只负责启动，`quantize.py` 包含该方法的量化流程。
损失、敏感度、重构与搜索放在对应方法的必要模块中。方法不会自行加载测试集或执行评估。

```bash
python -m pip install -r setting_environment/quantization_requirements.txt

python -m SAPQ.sapq --config configs/sapq/default.json \
  --model_path /path/to/poseidon-checkpoint \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams --DEV cuda \
  --save_path quantization_artifacts/NS-PwC/sapq

python -m evaluation.evaluate \
  --model_path quantization_artifacts/NS-PwC/sapq \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams --DEV cuda \
  --initial_time 0 --final_time 15 --ar_steps 3 \
  --output_path quantization_evaluation/NS-PwC/sapq.json
```

命令行参数覆盖 JSON 配置。`configs/<method>/` 提供可修改的模板；真实模型和数据路径需要自行指定。
默认校准选择训练集前 64 条完整轨迹，支持固定 seed 的随机轨迹选择。
Wave-Gaussians 使用 `--final_time 15`。Uniform、Dynamic 和 FP 不要求离线校准数据；
StaticQ 在量化激活时才收集校准统计。所有正式产物目前为浮点 QDQ，不代表整数硬件加速。

- [目录职责、旧入口迁移和兼容说明](docs/layout.md)
- [校准协议、方法范围、保存格式及评估口径](quantization/README.md)
- [批量与 Slurm 调度](scripts/README.md)
- [方法配置](configs/README.md)
- [第三方原始源码与适配范围](reference/third_party/LAYOUT.md)
- [原 Poseidon 模型说明](docs/poseidon_upstream.md)

```bash
python -m pytest
```

测试只发现 `tests/` 中的当前测试；历史实验和上游测试留在归档／参考目录，不参与默认测试。
