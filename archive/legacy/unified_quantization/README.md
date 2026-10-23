# Poseidon quantization：MySVD 校准协议

本分支把 Poseidon 的量化流程统一成三个阶段：加载同一份校准集、执行量化、保存模型。
评估由 `python -m quantization.evaluate` 单独执行。生产入口不读取旧的 `/lus/...`
范围、步长、Fisher 或校准 batch 缓存，也不在训练期间构造验证/测试集。

校准策略移植自 `physic_model_comp/poseidon/MySVD/helper/calibration_dataset.py`
及 `MySVD/mysvd.py`。参考代码仅用于阅读和对照测试，运行本仓库不依赖 MySVD。
对应的 ScOT 数据集实现已经逐文件比较：时间索引和规范化逻辑一致。

## 校准集的精确定义

- `--traj_num 64`：默认选择训练集前 64 条**完整轨迹**。
- `--random_calibration y --calibration_seed 42`：先加载完整训练划分，然后严格使用
  `random.Random(seed).sample(range(total_trajectories), traj_num)`；保留返回的轨迹顺序。
- 时变问题使用 `fix_input_to_time_step=None`、`time_step_size=1`、
  `max_num_time_steps=final_time-time_back`，保留所有 `t1 <= t2` 的时间对，包括零时间差。
- 椭圆问题和 MySVD 一样，不传上述时间参数。
- `shuffle=False`、`drop_last=False`，没有 `calib_steps`、随机样本截断或隐式子集。
  `final_time=20,time_back=0` 时，每条轨迹有 231 对，64 条共 14,784 个样本。
- `--batch_size` 只控制内存批大小；默认 2。它不减少校准集，且会记录在元数据中。
  算法优化的 batch 大小会影响数值，比较实验应同时固定这个参数。
- Wave-Gauss 的最大时间是 15，需显式 `--final_time 15`，不会静默裁剪成别的协议。

`metadata.json` 记录所选相对/绝对轨迹索引、全部时间对、规范化、实际样本数、
源码摘要、数据文件身份与内容探针、校准指纹以及全部有效参数。
文件内容探针用于防止误用缓存，并非对整个大型数据文件的完整密码学哈希。
每次量化使用独立临时目录，不能误读旧实验的无来源缓存。

## 运行

要求 Python >= 3.10 和适合当前设备的 PyTorch。

```bash
python -m pip install -r setting_environment/quantization_requirements.txt
```

模型路径默认沿 MySVD 的
`checkpoints/Poseidon-L/no_phyloss/<dataset>/origin/none`，数据路径默认
`poseidon_main/camlab-ethz/down_streams`。本仓库原来的模型/数据软链接指向集群，
请用参数提供实际可访问的路径。数据目录既可以是包含多个 `.nc` 文件的目录，也可以是
包含当前任务 `.nc` 文件的目录；底层文件名与 ScOT 一致。

量化并保存（以下路径需替换成实际路径）：

```bash
python -m quantization.quantize \
  --method sapq \
  --model_name Poseidon-L \
  --model_path /path/to/poseidon-checkpoint \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams \
  --traj_num 64 --random_calibration n --calibration_seed 42 \
  --final_time 20 --time_back 0 --batch_size 2 \
  --init_bits 9 --target_bits 8 --sensitivity sobolev \
  --sensitivity_normalization raw --DEV cuda \
  --save_path quantization_artifacts/NS-PwC/sapq
```

独立评估保存的模型（0→15，三次自回归；不重新量化）：

```bash
python -m quantization.evaluate \
  --model_path quantization_artifacts/NS-PwC/sapq \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams \
  --split test --initial_time 0 --final_time 15 --ar_steps 3 \
  --batch_size 16 --DEV cuda \
  --output_path quantization_evaluation/NS-PwC/sapq.json
```

默认评估完整 test split；`--num_trajectories` 可显式选择其确定性前缀并记录协议。
自回归只把模型预测作为下一步输入，固定/掩码通道从已知输入恢复；不会把未来标签喂给模型。
指标报告终点的按样本平均值，末尾不足一批的样本不会获得额外权重。

## 方法和范围

| `--method` | 量化逻辑 |
|---|---|
| `fp` | 保存原浮点模型，供相同独立评估协议使用 |
| `uniform` | 原 PPQ 对照的逐输出通道 max-abs 权重网格，`2*maxabs/(2^bits-1)` |
| `dynamic` | DynamicQ 对称权重量化及可选运行时输入激活量化 |
| `static` | 用完整校准集估计输入激活范围的静态 QDQ |
| `ppq` | 全网 MC 平均 MSE + 参数量加权平均位宽上限，保持当前 PPQ trainer 的实际目标 |
| `ppq-layerwise` | 逐层 MC 平均 MSE + MDL 先验 |
| `sapq` | 全网联合步长优化，MC log-mean-exp 似然 + 可选位宽先验 |
| `sapq-block` | 逐 ScOT/残差块的 Fisher 加权重构 |
| `sapq-layerwise` | 逐层重构和可选位宽先验 |
| `sapq-network` | 逐块调度，但在最终网络输出计算似然；保留该消融的 sum 归约和先验缩放 |
| `brecq` | Poseidon 块包装 + AdaRound 重构，保存后不依赖包装器/原模型 |
| `repq` | 所选 Linear 的 scale/shift 重参数化；ScOT 条件化归一化用显式输入仿射保持等价 |
| `ptq4vit` | 所选 Linear 的 Hessian/Fisher 区间搜索，GELU 后采用 twin uniform 激活量化 |

PPQ/SAPQ 当前只支持权重量化，`--activation_bits` 必须为 32；步长学习与模型权重更新分开。
统一的 Linear 选择规则沿原 `inspect_layers`：排除名字中含 `norm`/`time` 的条件化层。
BRECQ 按自己的块定义处理卷积和 Linear，并在结果元数据中记录实际范围。

`repq`、`ptq4vit` 是明确限定为所选 Linear 的 Poseidon 适配，**没有量化 attention
matmul/softmax**，不能当作原 ImageNet 实现的全算子复现。`static` 是可保存的静态 QDQ
基线；历史 `StaticQ/eager_static.py` 的 QNNPACK 探索并不等同于这个入口。

这些模型保存的是浮点量化/反量化模拟（QDQ），不是打包的整数模型，不能由文件大小或
运行时间推断实际低位整数硬件加速。PPQ/SAPQ 的有效位宽是连续代理
`log2(range/step)`，元数据记录实际参数量加权结果，图例/目录名不再充当位宽证据。

SAPQ 默认保留当前 raw 敏感度协议和 `sigma=sigma0*(1+alpha*sensitivity)`。
`--sensitivity_normalization normalized` 是显式切换。不同粒度保留各自原先的
loss reduction 和先验缩放，不把它们伪装成仅改变训练粒度的完全等价实验。
Sobolev 敏感度保留历史实现的 dataset `std` 广播约定，并写入 metadata；本次不静默改动
历史方法的物理损失定义。评估侧则按物理通道恢复单位。

## 保存格式与评估口径

每个输出目录包含：

- `compressed_model.pt`：完整、可独立加载的 CPU 模型，与 MySVD 保存形式一致。
- `quantization_state.pt`：算法对应的步长、范围、敏感度、历史或其他量化状态。
- `metadata.json`：方法、校准选择、全部参数、代码版本、模型校验和与量化统计。

已有模型默认不覆盖，另选 `--save_path` 或明确加 `--overwrite`。
完整模型文件使用 Python/PyTorch 序列化，只加载可信来源的文件。
成功后清理本次临时缓存；`--keep_work_dir` 可保留，失败时保留用于诊断。

独立评估的 `L1`/`RelL1` 沿 ScOT：在归一化输出上计算，RelL1 的单位是百分数。
物理 Sobolev 指标先反归一化，一阶差分除网格间距；不可压缩流另报告散度与涡量误差。
结果 JSON 附指标定义。这与旧绘图脚本中未除网格间距的 Sobolev 数值不同，不能直接混表。

## 批量调度与旧入口

`quantization/quantize.sh` 的方法/数据集/轨迹/seed 矩阵采用 MySVD shell 风格：

```bash
QUANT_RUNNER=dry-run QUANT_METHODS='sapq ppq brecq' bash quantization/quantize.sh
QUANT_RUNNER=sbatch QUANT_PYTHON=/path/to/python bash quantization/quantize.sh
```

用 `QUANT_MODEL_ROOT`、`QUANT_DATA_ROOT`、`QUANT_OUTPUT_ROOT` 覆盖路径；shell 不激活任何
硬编码 Conda 环境。`QUANT_RUNNER=local` 会顺序运行。固定前缀模式只运行一次，随机模式按 seed 展开。

PPQ/SAPQ/BRECQ/RepQ 的现有主要 Python 入口及批量 shell 已转发统一流程。
SAPQ 的旧 Python 入口保留各自原有的 normalized／raw 默认值；统一入口默认 raw，
可用 `--sensitivity_normalization` 显式指定。旧 ablation shell 默认 normalized。
`StaticQ/run_poseidon.py`、`DynamicQ/run_poseidon.py`、`PTQ4ViT/run_poseidon.py` 提供相同参数。
旧绘图/诊断所需的 global helper 保存在 `SAPQ/legacy_global_helpers.py`，它们不参与新流程。
ImageNet/detection 第三方程序、VICON 专用入口和早期研究原型保留原用途，不能混用这份 Poseidon 校准集。

## 验证

```bash
python -m pytest tests -q
```

测试包括 MySVD helper 的直接对照、完整时间对和随机采样、数学/梯度检查、全部方法在真实
小型 ScOT 上的保存/回载、全新进程独立评估、mask/AR 防止标签泄漏。
合成小模型的测试证明接口与计算链路可运行，不证明真实 Poseidon-L 数据集的精度或速度。
