# KISS-IMU 项目总览

本文档从源码角度解释当前仓库的整体结构、数据流、核心算法模块、运行入口和产物。它和 `README.md`、`RUNNING.md` 的分工如下：

- `README.md`：项目论文背景、快速开始、训练/评估/推理命令。
- `RUNNING.md`：如何在 fresh clone 上跑通，尤其是合成数据 CPU demo 和数据契约。
- 本文档：解释“这个项目由哪些模块组成、每个模块做什么、训练和推理时数据怎样流动”。

## 1. 项目是什么

KISS-IMU 是一个用于惯性里程计的学习系统。输入是原始 IMU 流，模型学习对加速度计和陀螺仪读数进行去噪/修正，再通过 IMU 预积分得到运动轨迹。

项目的关键思想是：

1. 不直接训练网络输出位姿，而是训练网络输出 IMU 修正量和不确定性。
2. 使用 LiDAR Odometry 生成自监督伪标签：相邻点云先做 ICP，再和 IMU 预积分结果一起做 PGO。
3. 对不同运动模式做均衡学习：用 GMM 按运动速度/角速度划分运动模式，对稀有模式增加权重，并用频率门控减少常见模式主导训练。
4. 推理时可将校正后的 IMU、ICP 因子和协方差一起放入 PGO，得到融合后的轨迹。

一句话数据流：

```text
dataset window
  -> IMUNet 校正 IMU
  -> IMUIntegrator 预积分
  -> LOModule ICP 点云配准
  -> pvgo.optimize 融合 IMU/ICP
  -> 选择 GT 或 ICP/PGO 伪标签
  -> loss + GMM 权重 + freq gate
  -> 更新网络
```

## 2. 目录结构

```text
.
├── README.md                  # 论文项目说明和常用命令
├── RUNNING.md                 # 运行指南、合成数据 demo、数据契约
├── PROJECT_OVERVIEW_zh.md     # 本文档
├── requirements.txt           # 原生环境依赖列表
├── scripts/
│   ├── train.sh               # 训练入口
│   ├── evaluate.sh            # checkpoint 指标评估
│   ├── inference.sh           # 保存 ICP/PGO 推理轨迹
│   ├── raw_pvgo.sh            # raw IMU + PVGO baseline
│   └── tsne_encoder.sh        # encoder feature t-SNE 可视化
├── src/
│   ├── train.py               # 主训练流程
│   ├── evaluate.py            # RPE/APE 评估
│   ├── inference.py           # 轨迹推理与保存
│   ├── raw_pvgo.py            # 不使用学习网络的 baseline
│   ├── tsne_encoder.py        # encoder 特征可视化
│   ├── data/                  # 数据集、batch collate
│   ├── models/                # IMU 网络、LO、GMM、PGO
│   ├── training/              # 积分器、loss、日志、辅助控制器
│   └── utils/                 # 参数、点云、overlap 计算
├── tools/
│   └── gen_synth_dataset.py   # 生成合成 DiTer-OS 风格数据
├── examples/
│   └── dataset_layout.md      # 数据格式示例
└── docker/                    # GPU 完整环境和 CPU demo 环境
```

## 3. 核心数据约定

### 3.1 序列目录

每个序列放在 `<data_root>/<sequence>/` 下：

```text
<data_root>/<sequence>/
├── imu.csv
├── gt_pose.csv
└── points/
    ├── data/
    │   ├── 000000.bin
    │   └── ...
    └── timestamps.txt
```

`src/data/seq_dataset.py` 中的 `DATASET_CFG` 描述不同数据集的列索引和格式。当前明确配置了：

- `diter_os`
- `diter++`
- `kitti`

CLI 中虽然还出现了 `mulran`、`yeoncheon` 等名称，但当前 `SeqDataset` 对未配置类型会回退到 `diter_os` 格式。如果要严谨支持新数据集，应先在 `DATASET_CFG` 中加入对应的 IMU 列、GT pose 格式、点云 dtype、外参和重力配置。

### 3.2 一个 dataset item 是什么

`SeqDataset` 把相邻两帧 LiDAR scan 之间的数据作为一个窗口：

```text
item i = scan i -> scan i+1 的窗口
```

每个 item 包含：

| key | 含义 |
| --- | --- |
| `accels`, `gyros` | 窗口内 IMU 加速度和角速度 |
| `imu_ts`, `imu_dts` | IMU 时间戳和每个采样间隔 |
| `scan0`, `scan1` | 相邻两帧点云 |
| `scan1_ts` | KISS-ICP deskew 路径使用的点时间戳 |
| `gt_pose0`, `gt_pose1` | 窗口起点/终点 GT 位姿 |
| `gt_velocity` | 起点速度，按 GT 位置差分估计 |

`src/data/collate.py` 会把变长 IMU 序列 padding 到 batch 最大长度，并保留 `valid_length`。点云数量也可变，所以保持 Python list，不做 tensor stack。

### 3.3 位姿和状态表示

项目中位姿统一使用 7 维向量：

```text
[x, y, z, qx, qy, qz, qw]
```

源码中再转换为 `pypose.SE3` 或 `pypose.SO3` 做李群计算。常见状态包括：

- `pos`：世界系位置。
- `rot`：四元数或 `SO3`。
- `vel`：世界系速度。
- `cov`：IMU 预积分传播的 9x9 协方差。

重力默认是 `[0, 0, 9.81]`。`training/integrator.py` 兼容了不同 pypose 版本中 gravity 参数是 3 维向量或 scalar 的差异。

## 4. 训练主流程

训练入口是 `scripts/train.sh`，它最终进入 `src/train.py`。

### 4.1 初始化

`train.py` 启动后会：

1. 创建 `TrainingMonitor`，写 TensorBoard 日志。
2. 创建 `IMUNet` 和 Adam optimizer。
3. 为每个训练/验证序列创建 `SeqDataset` 和 `DataLoader`。
4. 用训练序列的 IMU 数据拟合 `GmmModule`。
5. 保存 `gmm.joblib`、`gmm_initial_distribution.png`、`gmm_initial_stats.txt`。
6. 初始化学习率 scheduler。

### 4.2 GMM 运动模式建模

`models/gmm.py` 从 IMU 流中提取两维特征：

```text
[linear_speed, angular_speed]
```

它使用 yaw-only 的简化积分估计线速度和角速度，并按 `win_sec=0.2` 做滑动统计。随后：

1. 用 `StandardScaler` 标准化特征。
2. 用 `GaussianMixture` 聚类运动模式。
3. 如果 `K=None`，在 K=2 到 K=7 之间用 BIC 自动选择。
4. 用 class-balanced effective number 计算每个 GMM component 的训练权重。

训练时每个 IMU 窗口会通过 `predict_window(..., reduce="soft-mode")` 得到运动模式 id。该 id 再用于：

- `base_w`：GMM 类别均衡权重。
- `FreqGate`：根据近期使用频率，对常见 component 降低保留概率，对稀有 component 更倾向保留。
- `rare_loss`：对当前 batch 中最稀有的一部分样本额外加权。

### 4.3 IMUNet 做什么

`models/imu_net.py` 中的 `IMUNet` 输入 padding 后的 IMU：

```text
(B, T, 6) = [acc, gyro]
```

网络结构：

```text
Conv1d(kernel=10, stride=5)
  -> GELU + Dropout
  -> GRU(32 -> 64)
  -> GRU(64 -> 128)
  -> decoders
```

解码器输出：

- `acc_noise`：加速度修正。
- `gyr_noise`：角速度修正。
- `acc_cov`：加速度不确定性。
- `gyr_cov`：角速度不确定性。

网络输出的时间长度比原始 IMU 短。`broadcast_to_valid` 会把每个低频 correction 分配回原始有效 IMU 片段：

```text
corrected_acc = raw_acc + acc_noise_broadcast
corrected_gyr = raw_gyr + gyr_noise_broadcast
```

所以模型学习的是“怎样修正 IMU”，不是“直接预测轨迹”。

### 4.4 IMU 预积分

`training/integrator.py` 封装 `pypose.module.IMUPreintegrator`。

训练中会做两次积分：

1. `motion_mode=False`：从当前全局状态开始积分，得到 IMU 轨迹节点 `imu_nodes`、速度 `imu_vels`、协方差 `imu_covs`。
2. `motion_mode=True`：从零位移/零速度状态积分，得到每个窗口的相对运动 `imu_motions`、相对速度变化和相对协方差。

这两组结果分别服务于：

- 和标签轨迹做 loss。
- 作为 PGO 的 IMU factor。

### 4.5 标签从哪里来

训练有两种监督来源。

#### GT supervision

当 `USE_GT=true` 时，`scripts/train.sh` 给 `train.py` 传入 `--use-gt`。此时训练直接使用 `sample['gt_pose1']` 作为标签，不调用 LiDAR ICP，也不调用 PGO。

这条路径适合：

- 跑合成数据 demo。
- 在没有 LiDAR C++ 后端时测试训练链路。
- 做“只看 GMM 均衡学习贡献”的 ablation。

#### 自监督伪标签

默认训练路径中，标签来自 ICP 和 PGO 的二选一：

1. `LOModule` 对每个相邻点云窗口做 ICP，得到 `icp_poses`、`icp_motions` 和 overlap 分数。
2. `pvgo.optimize` 把 IMU 预积分因子和 ICP 因子放进 pypose LM optimizer，得到 `pgo_poses`。
3. 对 PGO 结果也计算 pose-aligned overlap。
4. 每个窗口选择 ICP overlap 和 PGO overlap 中更高的一方作为 label。

因此训练标签不是单一来源，而是：

```text
label_i = argmax_overlap(ICP_i, PGO_i)
```

这种设计让系统在训练时倾向选择几何配准更一致的位姿约束。

### 4.6 损失函数

`training/losses.py` 的 `get_losses` 返回六项：

- `rot_loss`：相对运动旋转误差。
- `vel_loss`：相邻速度变化误差。
- `pos_loss`：相对运动平移误差。
- `rot_cov_loss`：旋转相关的不确定性 NLL 风格项。
- `vel_cov_loss`：速度相关的不确定性 NLL 风格项。
- `pos_cov_loss`：位置相关的不确定性 NLL 风格项。

训练中最终每个样本的总损失形式是：

```text
rot_w   * rot_loss     + cov_r_w * rot_cov_loss
vel_w   * vel_loss     + cov_v_w * vel_cov_loss
pos_w   * pos_loss     + cov_t_w * pos_cov_loss
```

然后再乘上：

- GMM component 权重。
- component convergence active mask。
- frequency gate keep mask。

最后叠加 `rare_boost * rare_loss`，反向传播更新 `IMUNet`。

## 5. LiDAR Odometry 和 PGO

### 5.1 LOModule

`models/lo_module.py` 支持多种点云配准后端：

- `kiss_icp`
- `fast_gicp`
- `small_gicp`
- `g_icp`

其中 `cv2`、`open3d`、`pygicp`、`kiss_icp`、`small_gicp` 都是可选依赖。源码使用 `try/except` 延迟导入，所以没有这些库时仍可 import 项目；只有真正选择对应后端时才需要安装。

`LOModule.forward` 的输出是：

```text
(global_poses, relative_motions, overlap_scores)
```

并且会保存 `scans0_np`、`scans1_np`，供 PGO 后 overlap 重新计算。

`small_gicp` 还支持 `use_submap`，把历史点云维护为局部 submap，用于和当前 scan 配准。

### 5.2 overlap 分数

`utils/overlap_score.py` 使用 KDTree 计算双向最近邻重合率：

```text
overlap = matched_points / total_points
```

该分数用于：

- 训练时在 ICP label 和 PGO label 之间选择。
- 推理时记录 ICP/PGO 的几何一致性。
- 自适应 PGO 权重路径中给 ICP factor 加权。

### 5.3 PVGO / PGO

`models/pvgo.py` 中的 `GraphOptimizer` 把待优化量设为：

- `nodes`：一串 `SE3` 位姿节点。
- `vels`：每个节点的速度。

优化误差包括：

- LO pose-between factor。
- IMU rotation factor。
- IMU velocity factor。
- IMU translation factor。

`optimize(...)` 使用 pypose 的 Levenberg-Marquardt 优化，并在结束后对齐到输入的第一个节点。

推理时如果开启 `--use-adaptive-weight`：

- ICP factor 权重来自 overlap。
- IMU factor 权重来自积分协方差。

否则使用 `--lm-weight` 给定的固定权重。

## 6. 运行入口

### 6.1 训练

```bash
bash scripts/train.sh
```

常用环境变量：

| 变量 | 作用 |
| --- | --- |
| `DATA_DIR` | 数据集根目录 |
| `DATA_TYPE` | 数据集类型 |
| `TRAIN_SEQS` / `VALID_SEQS` | 训练/验证序列 |
| `LO_MODEL` | ICP 后端 |
| `USE_GT` | 是否使用 GT supervision，跳过 ICP/PGO |
| `USE_SUBMAP` | 是否为 small_gicp 使用 submap |
| `TRAIN_RATIO` | 使用训练窗口比例 |
| `DEVICE` | `cuda:0` 或 `cpu` |

输出目录由 `scripts/train.sh` 拼接，通常位于：

```text
results/<data_type>/.../<train_seqs>_valid_<valid_seqs>/<train_ratio>/
```

### 6.2 评估

```bash
CKPT=path/to/best_model.ckpt bash scripts/evaluate.sh
```

`src/evaluate.py` 不运行 ICP/PGO。它做的是：

1. 加载训练好的 `IMUNet`。
2. 从 GT 起始状态积分校正后的 IMU。
3. 和 GT endpoint 比较。
4. 输出平均 RPE translation、RPE rotation、APE。

### 6.3 推理

```bash
CKPT=path/to/best_model.ckpt SEQS="Forest_new" bash scripts/inference.sh
```

`src/inference.py` 会完整运行：

```text
IMUNet -> IMU integration -> ICP -> PGO
```

每个序列输出：

```text
<out-dir>/<seq>/inference.npz
<out-dir>/<seq>/trajectory.png
```

`inference.npz` 中保存：

- `imu_poses`
- `icp_poses`
- `pgo_poses`
- `gt_poses`
- `icp_overlap`
- `pgo_overlap`

### 6.4 raw PVGO baseline

```bash
bash scripts/raw_pvgo.sh
```

`src/raw_pvgo.py` 不使用 `IMUNet` 学习修正，而是直接用 raw IMU 做预积分，再和 ICP 做 PGO。它对应论文中的 raw IMU + PVGO baseline，用于对比“学习 IMU 修正”带来的收益。

### 6.5 encoder t-SNE

```bash
CKPT=path/to/best_model.ckpt bash scripts/tsne_encoder.sh
```

`src/tsne_encoder.py` 会：

1. 加载 checkpoint 和同目录下的 `gmm.joblib`。
2. 抽取 `IMUNet.encoder` 特征。
3. 用 GMM 给 IMU 窗口/特征块分配运动模式 id。
4. 用 t-SNE 投影到 2D。
5. 保存 `tsne_train.png`、`tsne_eval.png`。

它用于观察 encoder 表征是否按运动模式分离。

## 7. 输出产物

一次训练通常会产生：

| 文件/目录 | 含义 |
| --- | --- |
| `best_model.ckpt` | 验证集最优网络 checkpoint |
| `train/<seq>/ckpt/*.ckpt` | 每个 epoch 的训练侧 checkpoint |
| `valid/<seq>/ckpt/*.ckpt` | 每个 epoch 的验证侧 checkpoint |
| `train/<seq>/ckpt/*.png` | 训练轨迹对比图 |
| `valid/<seq>/ckpt/*.png` | 验证轨迹对比图 |
| `gmm.joblib` | 拟合好的 GMM 和 scaler |
| `gmm_initial_distribution.png` | GMM 初始 component 可视化 |
| `gmm_initial_stats.txt` | GMM component 统计 |
| `monitoring/` | TensorBoard 日志 |
| `inference/<seq>/inference.npz` | 推理轨迹数据 |
| `inference/<seq>/trajectory.png` | 推理轨迹图 |
| `tsne/*.png` | encoder t-SNE 可视化 |

## 8. 合成数据 demo

`tools/gen_synth_dataset.py` 可以生成一个 DiTer-OS 风格的合成序列：

```bash
python tools/gen_synth_dataset.py --out data/synth/Synth01
```

它会生成：

- 200 Hz IMU。
- 10 Hz LiDAR scan 时间戳。
- 平面 yaw-only GT 轨迹。
- 多种运动模式：慢直行、快直行、缓转、急转等。
- 每帧点云 `.bin`。

推荐 demo 路径：

```bash
CUDA_VISIBLE_DEVICES="" \
DATA_DIR=$PWD/data/synth DATA_TYPE=diter_os \
TRAIN_SEQS=Synth01 VALID_SEQS=Synth01 \
LO_MODEL=small_gicp DEVICE=cpu EPOCH=10 BATCH_SIZE=4 \
USE_GT=true TRAIN_RATIO=1.0 \
bash scripts/train.sh
```

这条路径使用 GT supervision，所以不需要真实数据集、不需要 GPU，也不需要 LiDAR C++ 后端。

Docker CPU demo 位于：

- `docker/Dockerfile.demo`
- `docker/docker-compose.demo.yml`
- `docker/run_demo.sh`
- `docker/DEMO_zh.md`

## 9. 如何扩展项目

### 9.1 添加新数据集

优先修改 `src/data/seq_dataset.py` 的 `DATASET_CFG`：

1. 配置 `acc_idx`、`gyr_idx`。
2. 配置 `gt_format`，必要时扩展 `_parse_gt_row`。
3. 配置 `lidar_dtype`。
4. 配置 `R_I_L`、`T_I_L`、`gravity`。
5. 用 `examples/dataset_layout.md` 中的方式检查 `SeqDataset` 是否能正常读取。

### 9.2 添加新 LO 后端

修改 `src/models/lo_module.py`：

1. 新增一个 wrapper class，提供 `get_motion(source_raw, target_raw)` 或对应接口。
2. 在 `LOModule.__init__` 中添加 `lo_model` 分支。
3. 保证返回的相对变换是 4x4 矩阵，最后能转换成 `[x,y,z,qx,qy,qz,qw]`。
4. 如需要 overlap，确保 `scans0_np`、`scans1_np` 保存的是配准使用的点云。

### 9.3 改训练目标

训练目标主要在 `src/train.py` 中的 `train(...)` 函数内：

- 改标签来源：调整 `--use-gt` 分支或 ICP/PGO label selection。
- 改损失：修改 `training/losses.py` 或 `per_sample_total` 的权重组合。
- 改运动均衡：修改 `GmmModule` 特征、component 权重或 `FreqGate`。
- 改 PGO 因子：修改 `models/pvgo.py`。

## 10. 实现注意事项

以下是按当前源码观察到的实现细节，理解和修改项目时需要留意：

1. `GMM_COMP_NUM` 在脚本和参数中存在，但 `src/train.py` 当前构造 `GmmModule(..., K=None, win_sec=0.2)`，实际总是走 BIC 自动选 K。如果需要固定 K，应把 `args.gmm_comp_num` 传入 `GmmModule`。
2. `CompConvergenceManager` 已在训练中构造并用于生成 active mask，但当前主训练流程没有调用 `conv_mgr.update(...)`，所以 component 收敛状态不会真正更新。
3. `CovWeightController` 被导入，训练日志中也预留了 `_cov_ctrl` 相关逻辑，但当前主训练没有初始化 `train._cov_ctrl`。实际 loss 权重主要来自命令行的 `cov_*_w`。
4. `SeqDataset` 目前是重构出的轻量实现，和现有脚本接口匹配；如果要复现实验中的完整真实数据集行为，建议重点核对每个数据集的列索引、坐标系、外参和 GT 格式。
5. 完整自监督路径需要可用的 LiDAR 后端库。合成数据 + `USE_GT=true` 路径可以在只有 PyTorch、pypose、scikit-learn 等核心依赖的 CPU 环境下跑通。
6. `src/evaluate.py` 当前写法是 `out_state, _ = integrator.integrate(...)`，但当前 `IMUIntegrator.integrate(...)` 只返回一个 dict。直接运行评估前可能需要把这里改为 `out_state = integrator.integrate(...)`。
7. `src/raw_pvgo.py` 当前向 `SeqDataset` 传入了 `method`、`train_ratio` keyword，而当前 `SeqDataset.__init__` 不接收这些参数。运行 raw baseline 前需要先对齐这条接口。

## 11. 阅读源码的建议顺序

建议按下面顺序读：

1. `src/data/seq_dataset.py` 和 `src/data/collate.py`：先理解一个 batch 长什么样。
2. `src/models/imu_net.py`：理解网络输入输出。
3. `src/training/integrator.py`：理解校正 IMU 如何变成位姿。
4. `src/models/lo_module.py`：理解 ICP 如何产生相对位姿。
5. `src/models/pvgo.py`：理解 PGO 如何融合 IMU 和 ICP。
6. `src/models/gmm.py`：理解运动模式如何划分和加权。
7. `src/training/losses.py`：理解训练目标。
8. `src/train.py`：把所有模块串起来看完整训练循环。
9. `src/evaluate.py`、`src/inference.py`、`src/tsne_encoder.py`：理解训练后的使用方式。
