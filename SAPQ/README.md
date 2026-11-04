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

## 导出到 ritchie 的统一 evaluation

使用训练时同一份原始 checkpoint 和最终步长，在 main 根目录执行：

```bash
python -u -m SAPQ.export_main_artifact \
  --step_path SAPQ/artifacts_global/fluids.incompressible.PiecewiseConstants/network_block_sens_sobo/sapq_global_step_sizes.pt \
  --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
  --save_path /cs/research/vecg/EasyPDE/quantization_artifacts/NS-PwC/sapq-main
```

`--checkpoint` 省略时采用步长文件中记录的原始模型路径。工具逐通道应用 `round(W/s)*s`，生成 `compressed_model.pt`、`quantization_state.pt` 和 `metadata.json`。这是以浮点张量保存的确定性量化权重，适合当前 evaluation；不做低 bit 整数打包，因此文件体积不会按有效位宽同比下降。

平均 bit 从步长文件旁的 `sapq_global_history.json` 最后一个 epoch 读取，可用 `--history_path` 指定对应历史。缺失时记录为 null，不使用配置中的 target_bits 代替。必须使用同一次运行的步长与历史，以及原始、尚未量化的 checkpoint。导出工具拒绝覆盖已有非空目录。

在 ritchie 工作目录中直接评估仓库外的导出产物：

```bash
python -m evaluation.evaluate \
  --model_path /cs/research/vecg/EasyPDE/quantization_artifacts/NS-PwC/sapq-main \
  --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path poseidon/camlab-ethz/down_streams --DEV cuda \
  --final_time 20 --batch_size 8 --physics_batch_size 10 \
  --output_path quantization_evaluation/NS-PwC/sapq-main.json
```

这里采用 ritchie 默认的 SAFE-SVD 测试协议，输出平均 bit、各指标的 origin/compressed 值与变化百分比。main 日志中的随机验证子集 L1 与这份统一测试结果使用不同评估设置。
