# 用 Docker 跑通 KISS-IMU 合成数据 Demo（CPU，无需数据集 / 无需 GPU）

这份教程用一套**自包含的 CPU 镜像**跑通 KISS-IMU 的整条训练链路：
合成一段 DiTer-OS 风格的轨迹 → 拟合 GMM 运动分布 → 用 GT 监督训练 IMU 网络 →
输出模型、GMM、轨迹图和 TensorBoard 日志。

它**不需要**真实数据集、不需要 GPU，也不需要那一堆难编译的 LiDAR C++ 库
（open3d / kiss-icp / small_gicp / pygicp）。

> 想跑真实数据集 + 完整 LiDAR ICP/PGO 伪标签的 GPU 流程？那条路用
> `docker/docker-compose.yml`（需要 `sparolab/kiss-imu:v1.0` 镜像和 NVIDIA runtime）。
> 本教程用的是另一套文件：`docker/Dockerfile.demo` + `docker/docker-compose.demo.yml`。

---

## 0. 前置条件

- 已安装 Docker（`docker --version`）和 Compose v2（`docker compose version`）
- 能联网（构建时要拉 `python:3.10-slim` 基础镜像 + pip 包，约 1.5GB）
- 不需要 GPU、不需要 NVIDIA 驱动

---

## 1. 一条命令跑通

在仓库根目录执行：

```bash
docker compose -f docker/docker-compose.demo.yml up --build
```

第一次会构建镜像（拉 CPU 版 PyTorch / pypose 等，约 5~8 分钟），构建完自动：

1. 生成合成数据到 `data/synth/Synth01/`
2. 在 CPU 上训练（默认 10 个 epoch，GT 监督，跳过 LiDAR ICP/PGO）

跑完后日志结尾类似：

```
[Summary] RTE in Synth01 at epoch N: 0.5XX m
[Summary] RRE in Synth01 at epoch N: 12.XX deg
[run_demo] done. Trained artifacts:
  results/diter_os/use_gt/.../Synth01_valid_Synth01/1.0
```

> 这些合成数据上的数值只用来证明流水线能跑通，**不能**和论文在真实数据上的结果相比。

---

## 2. 看结果

所有产物都通过挂载卷写回宿主机的 `results/` 目录。定位结果目录：

```bash
RD=$(find results -name best_model.ckpt -printf '%h\n' | head -1); echo "$RD"
```

里面有：

| 文件 | 说明 |
| --- | --- |
| `best_model.ckpt` | 训练好的 IMU 网络（`train/`、`valid/` 下还有每个 epoch 的 ckpt） |
| `gmm.joblib` + `gmm_initial_distribution.png` + `gmm_initial_stats.txt` | 拟合出的运动分布 GMM 及各分量权重/均值 |
| `train/Synth01/ckpt/000N.png`、`valid/Synth01/ckpt/000N.png` | 俯视轨迹图（GT vs 预测） |
| `monitoring/` | TensorBoard 日志 |

看 TensorBoard（在宿主机本地装个 tensorboard 即可）：

```bash
tensorboard --logdir "$RD/monitoring"
```

---

## 3. 调参再跑（不用重新构建）

镜像已经建好后，改 epoch / batch size 直接用 `run`，仓库是 bind-mount 的，
改代码也无需重建：

```bash
# 跑 20 个 epoch（轨迹会更好看一点）
docker compose -f docker/docker-compose.demo.yml run --rm -e EPOCH=20 demo

# 同时改 batch size
docker compose -f docker/docker-compose.demo.yml run --rm -e EPOCH=20 -e BATCH_SIZE=8 demo
```

进容器里手动玩：

```bash
docker compose -f docker/docker-compose.demo.yml run --rm demo bash
# 容器内当前目录就是 /home/test_ws/src（仓库根）
# 重新生成数据：
python tools/gen_synth_dataset.py --out data/synth/Synth01 --duration 120 --seed 7
# 手动训练（等价于 run_demo.sh 里的命令）：
CUDA_VISIBLE_DEVICES="" DATA_DIR=$PWD/data/synth DATA_TYPE=diter_os \
  TRAIN_SEQS=Synth01 VALID_SEQS=Synth01 LO_MODEL=small_gicp \
  DEVICE=cpu EPOCH=10 BATCH_SIZE=4 USE_GT=true TRAIN_RATIO=1.0 \
  bash scripts/train.sh
```

合成数据可调的参数：`--duration`、`--acc-bias`、`--gyr-bias`、`--acc-noise`、
`--gyr-noise`、`--n-points`、`--seed`。

---

## 4. 清理

```bash
docker compose -f docker/docker-compose.demo.yml down       # 停容器
docker image rm kiss-imu:demo-cpu                           # 删镜像
```

生成的数据和结果在宿主机：`data/synth/`、`results/`，按需删除。

---

## 这套 Demo 都做了什么 / 为什么这么配

| 选择 | 原因 |
| --- | --- |
| `python:3.10-slim` + CPU PyTorch | 不依赖任何预构建的私有镜像，谁都能 `--build` 出来 |
| `USE_GT=true` | 用 GT 位姿监督，**跳过 LiDAR ICP/PGO**，因此不需要那些 C++ 库；它本身就是论文里的 GT 监督消融，会真正跑到 IMU 网络、GMM 运动均衡采样、频率门控、协方差加权损失和 IMU 预积分 |
| `CUDA_VISIBLE_DEVICES="" DEVICE=cpu` | 强制 CPU，无 GPU 也能跑 |
| `LO_MODEL=small_gicp` | `USE_GT=true` 下其实不会被调用，只是让 `LOModule` 构造保持简单 |
| 仓库 bind-mount | 改代码 / 看结果都直接在宿主机，无需重建镜像 |

更多背景（数据契约、原仓库为何开箱跑不起来、原生环境怎么装）见
[`../RUNNING.md`](../RUNNING.md)。
