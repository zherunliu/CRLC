# CRLC: Contrastive Representation Learning for Conservative Offline RL

- **轨迹感知的对比学习**：利用同一轨迹内的时序相关性构建正负样本对
- **表征空间分布建模**：使用高斯分布建模数据集的表征分布，计算马氏距离
- **自适应保守惩罚**：根据状态-动作对的表征距离动态调整 Q 值惩罚强度

## 项目结构

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

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n crlc python=3.9
conda activate crlc

# 安装依赖
pip install -r requirements.txt

# 安装 D4RL（Antmaze）
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

### 2. 运行训练

```bash
# 在 Antmaze 环境上训练
python train_crlc.py --env antmaze-medium-diverse-v2 --seed 42

# 使用自定义参数
python train_crlc.py \
    --env antmaze-large-play-v2 \
    --contrastive-steps 100000 \
    --policy-steps 500000 \
    --beta-min 0.1 \
    --beta-max 5.0
```

### 3. 2D Point 环境实验

```python
from envs.point_maze import PointMaze2D, generate_offline_dataset

# 创建环境
env = PointMaze2D()

# 生成离线数据
dataset = generate_offline_dataset(env, num_trajectories=100, policy_type='mixed')
```

## 实验说明

### D4RL Antmaze 任务

| 任务                   | 数据集            |
| ---------------------- | ----------------- |
| antmaze-umaze          | 简单U形迷宫       |
| antmaze-medium-diverse | 中等迷宫+多样轨迹 |
| antmaze-large-play     | 大型迷宫+探索数据 |

### 关键超参数

| 参数                 | 默认值 | 说明             |
| -------------------- | ------ | ---------------- |
| `representation_dim` | 128    | 表征向量维度     |
| `temperature`        | 0.1    | InfoNCE 温度参数 |
| `trajectory_window`  | 5      | 正样本轨迹窗口   |
| `beta_min`           | 0.1    | 最小惩罚权重     |
| `beta_max`           | 5.0    | 最大惩罚权重     |
| `percentile`         | 95     | 距离阈值百分位   |

## 算法流程

### 阶段 1: 对比学习预训练 (约 50K 步)

1. 从离线数据集构建轨迹索引
2. 采样锚点、正样本（同轨迹）、负样本（其他轨迹）
3. 使用 InfoNCE 损失训练联合编码器

### 阶段 2: 分布建模

1. 计算所有训练数据的表征
2. 拟合高斯分布（均值 + 协方差）
3. 确定 95% 百分位距离阈值

### 阶段 3: 保守策略优化 (约 1M 步)

1. 采样数据批次
2. 计算自适应惩罚权重: β(s,a) = β_min + (β_max - β_min) · σ(d_M(z) / τ - 1)
3. 使用加权 CQL 损失更新 Critic
4. 更新 Actor 和温度参数

## 可视化工具

```python
from visualize import (
    plot_tsne_representation,
    plot_ood_detection_roc,
    plot_distance_q_error_correlation,
    plot_training_curves
)

# t-SNE 表征可视化
plot_tsne_representation(encoder, in_data, ood_data)

# OOD 检测 ROC 曲线
plot_ood_detection_roc(distribution, in_repr, ood_repr)

# 表征距离-Q误差相关性
plot_distance_q_error_correlation(distances, q_errors)
```

## 模块测试

```bash
# 测试编码器
python -m crlc.encoder

# 测试对比学习
python -m crlc.contrastive

# 测试分布建模
python -m crlc.distribution

# 测试 CRLC-SAC
python -m crlc.crlc_sac

# 测试 2D 环境
python -m envs.point_maze
```
