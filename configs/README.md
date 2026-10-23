# 配置

`<method>/default.json` 是对应量化方法的示例配置，PPQ/SAPQ 的其他 JSON 是变体配置。
在仓库根目录执行，例如：

```bash
python -m SAPQ.sapq --config configs/sapq/block.json \
  --model_path /path/to/checkpoint --data_path /path/to/data --DEV cuda
```

显式命令行参数覆盖 JSON 配置；未知参数、非整数轨迹数及跨方法配置会报错。
只声明所需参数，其他参数由入口默认值提供。模型路径、数据路径和输出位置可用 CLI 覆盖。
Wave-Gaussians 需额外设置 `--final_time 15`。

PPQ: global/layerwise。SAPQ: global/network/block/layerwise。
raw/normalized、先验、敏感度类型是配置项，不对应额外 Python 入口。
DynamicQ/StaticQ/RepQ/PTQ4ViT 示例明确使用 W8A8；其他量化模板为权重量化 A32。

根目录原有的 `run*.yaml` 和 `sweep.yaml` 仍是 Poseidon finetuning 配置，
由 train_file/ 和 scOT/train.py 使用，保留其原路径以兼容训练脚本。
