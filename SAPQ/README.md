# Global SAPQ 的显存设置

main 的标准入口为 `python -u -m SAPQ.run_sapq_network_global`，raw 敏感度入口为 `python -u -m SAPQ.run_sapq_network_global_raw`。模型和数据仍通过 `PPQ_MODEL_PATH`、`PPQ_DATA_PATH`、`PPQ_DATASET_NAME` 指定；缺失的标准入口缓存可用仓库根目录下的 `tools/prepare_main_sapq.py` 准备。

两个入口现在复用同一份 FP 模型，冻结模型权重，只优化独立的通道步长。训练中的每个 Monte Carlo 样本采用 PyTorch checkpoint：反向时恢复该样本的原始随机状态，重算需要的张量，避免同时保留所有样本的噪声权重和激活。MC 数量、log-mean-exp likelihood、prior、学习率和早停设置沿用原实验配置；额外的重算会增加训练时间。训练日志会显示 `mc_backward=checkpoint_replay`。

验证 loader 的 batch size 仍为 512，默认每次仅将其中 8 个样本送入 GPU。预测在 CPU 上拼回原 batch 后再计算原来的 L1 / relative L1 和 batch 均值，因此保留原验证样本数及聚合方式。可以进一步缩小前向分块，例如 tcsh/csh 中运行：

```tcsh
setenv PPQ_EVAL_FORWARD_BATCHSIZE 2
python -u -m SAPQ.run_sapq_network_global
```

该变量只控制验证前向分块，不控制校准 batch、MC 数量或验证样本数。likelihood 的直接调用默认保留原堆叠实现供诊断比较；`SAPQTrainerGlobal` 显式启用 `memory_efficient=True`。

数值检查覆盖原始目标、步长梯度、连续 Adam 更新、CPU/CUDA 随机状态、验证分块聚合，以及小型真实 ScOT 模型：

```bash
python -m pytest tests/test_sapq_global_memory.py -q
```

无 CUDA 时会跳过 CUDA 对照。完整 Poseidon-L 的峰值显存需在实际 GPU 上测量。
