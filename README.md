# Contrastive Representation Learning for Conservative Offline RL

- **轨迹感知的对比学习**：利用同一轨迹内的时序相关性构建正负样本对
- **表征空间分布建模**：使用高斯分布建模数据集的表征分布，计算马氏距离
- **自适应保守惩罚**：根据状态-动作对的表征距离动态调整 Q 值惩罚强度

## project structure

```
crlc/
├── crlc/                    # 核心算法模块
│   ├── __init__.py
│   ├── encoder.py           # 状态-动作联合编码器
│   ├── contrastive.py       # 对比学习模块 (InfoNCE)
│   ├── distribution.py      # 表征分布建模
│   ├── crlc_sac.py          # CRLC-SAC 算法
│   └── buffer.py            # 离线数据缓冲区
├── envs/                    # 实验环境
│   ├── __init__.py
│   └── point_maze.py        # 2D 点导航环境
├── config/                  # 配置文件
│   └── default.yaml
├── train_crlc.py            # 主训练脚本
├── visualize.py             # 可视化工具
├── requirements.txt         # 依赖列表
└── README.md
```

## quick start

### environment setup

```bash
conda create -n crlc python=3.9
conda activate crlc

pip install -r requirements.txt

export MUJOCO_DIR=/root/.mujoco/mujoco210
export MUJOCO_INCLUDE=$MUJOCO_DIR/include
export MUJOCO_LIBRARY=$MUJOCO_DIR/bin
export MUJOCO_PLUGIN_PATH=$MUJOCO_DIR/bin
export LD_LIBRARY_PATH=$MUJOCO_DIR/bin:$LD_LIBRARY_PATH

pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

### run training

```bash
python train_crlc.py --env antmaze-medium-diverse-v2 --seed 42

python train_crlc.py \
    --env antmaze-large-play-v2 \
    --contrastive-steps 100000 \
    --policy-steps 500000 \
    --beta-min 0.1 \
    --beta-max 5.0
```

### 2D Point environment experiment

```python
from envs.point_maze import PointMaze2D, generate_offline_dataset

env = PointMaze2D()

dataset = generate_offline_dataset(env, num_trajectories=100, policy_type='mixed')
```

## visualization

```python
from visualize import (
    plot_tsne_representation,
    plot_ood_detection_roc,
    plot_distance_q_error_correlation,
    plot_training_curves
)

plot_tsne_representation(encoder, in_data, ood_data)

plot_ood_detection_roc(distribution, in_repr, ood_repr)

plot_distance_q_error_correlation(distances, q_errors)
```

## mini-test

```bash
# -m module
python -m envs.point_maze
```
