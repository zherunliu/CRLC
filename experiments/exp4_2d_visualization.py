"""
实验4: 2D Point 环境可视化实验

包含:
- 状态空间与表征空间对比可视化
- 惩罚权重分布可视化
- 策略轨迹可视化

运行方式:
    python experiments/exp4_2d_visualization.py

快速测试（减少训练步数）:
    python experiments/exp4_2d_visualization.py --contrastive_steps 5000
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def run_2d_experiment(
    n_trajectories: int = 100,
    contrastive_steps: int = 10000,
    policy_steps: int = 50000,
    save_dir: str = "./results/2d_visualization",
    device: str = "cuda",
):
    """运行 2D Point 环境的完整实验"""
    from envs.point_maze import PointMaze2D, generate_offline_dataset
    from crlc.encoder import JointEncoder
    from crlc.contrastive import ContrastiveLearner, TrajectoryBuffer
    from crlc.distribution import RepresentationDistribution

    os.makedirs(save_dir, exist_ok=True)

    # 创建环境和数据集
    print("[1/5] 创建环境和数据集...")
    env = PointMaze2D()
    dataset = generate_offline_dataset(
        env, num_trajectories=n_trajectories, policy_type="mixed"
    )

    state_dim = 2
    action_dim = 2

    # 创建编码器
    print("[2/5] 训练对比编码器...")
    encoder = JointEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        state_hidden_dim=64,
        action_hidden_dim=32,
        fusion_hidden_dim=64,
        representation_dim=32,
        projection_dim=16,
    ).to(device)

    # 对比学习预训练
    traj_buffer = TrajectoryBuffer(
        states=dataset["observations"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        next_states=dataset["next_observations"],
        dones=dataset["terminals"],
        trajectory_window=5,
    )

    learner = ContrastiveLearner(
        encoder=encoder,
        trajectory_buffer=traj_buffer,
        temperature=0.1,
        lr=3e-4,
        device=device,
    )

    learner.pretrain(
        num_steps=contrastive_steps, batch_size=128, num_negatives=64, log_freq=2000
    )

    # 构建分布
    print("[3/5] 构建表征分布...")
    distribution = RepresentationDistribution(representation_dim=32, device=device)

    encoder.eval()
    with torch.no_grad():
        all_repr = encoder.encode(
            torch.FloatTensor(dataset["observations"]).to(device),
            torch.FloatTensor(dataset["actions"]).to(device),
        )
    distribution.fit(all_repr)

    # 生成可视化
    print("[4/5] 生成可视化图表...")

    # 图1: 数据集分布
    plot_dataset_distribution(dataset, env, save_dir)

    # 图2: 表征空间可视化 (使用 PCA 降维到 2D)
    plot_representation_space(encoder, dataset, distribution, env, save_dir, device)

    # 图3: 惩罚权重热力图
    plot_penalty_heatmap(encoder, distribution, env, save_dir, device)

    # 图4: 惩罚权重分布对比
    plot_penalty_distribution(encoder, distribution, dataset, save_dir, device)

    print(f"[5/5] 所有可视化图表已保存至 {save_dir}")


def plot_dataset_distribution(dataset, env, save_dir):
    """绘制数据集在状态空间中的分布"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制轨迹点
    ax.scatter(
        dataset["observations"][:, 0],
        dataset["observations"][:, 1],
        c=dataset["rewards"],
        cmap="viridis",
        alpha=0.3,
        s=3,
    )

    # 绘制障碍物
    for obs in env.obstacles:
        x_min, x_max, y_min, y_max = obs
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="gray",
            edgecolor="black",
            alpha=0.7,
        )
        ax.add_patch(rect)

    # 绘制目标
    goal_circle = Circle(
        env.goal_pos, env.goal_radius, facecolor="green", alpha=0.5, label="Goal"
    )
    ax.add_patch(goal_circle)

    # 绘制起点
    ax.plot(env.start_pos[0], env.start_pos[1], "r^", markersize=12, label="Start")

    ax.set_xlim(-env.maze_size - 1, env.maze_size + 1)
    ax.set_ylim(-env.maze_size - 1, env.maze_size + 1)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title("Offline Dataset Distribution in State Space", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "dataset_distribution.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close()


def plot_representation_space(encoder, dataset, distribution, env, save_dir, device):
    """绘制表征空间的 2D 投影"""
    from sklearn.decomposition import PCA

    encoder.eval()
    with torch.no_grad():
        repr_all = (
            encoder.encode(
                torch.FloatTensor(dataset["observations"]).to(device),
                torch.FloatTensor(dataset["actions"]).to(device),
            )
            .cpu()
            .numpy()
        )

    # PCA 降维到 2D
    pca = PCA(n_components=2)
    repr_2d = pca.fit_transform(repr_all)

    # 计算距离
    with torch.no_grad():
        distances = (
            distribution.mahalanobis_distance(torch.FloatTensor(repr_all).to(device))
            .cpu()
            .numpy()
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 按原始位置着色
    ax = axes[0]
    scatter = ax.scatter(
        repr_2d[:, 0],
        repr_2d[:, 1],
        c=np.linalg.norm(dataset["observations"], axis=1),
        cmap="coolwarm",
        alpha=0.5,
        s=5,
    )
    plt.colorbar(scatter, ax=ax, label="Distance from Origin")
    ax.set_xlabel("PCA Dimension 1", fontsize=12)
    ax.set_ylabel("PCA Dimension 2", fontsize=12)
    ax.set_title("Representation Space (colored by state position)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # 右图: 按马氏距离着色
    ax = axes[1]
    scatter = ax.scatter(
        repr_2d[:, 0], repr_2d[:, 1], c=distances, cmap="RdYlGn_r", alpha=0.5, s=5
    )
    plt.colorbar(scatter, ax=ax, label="Mahalanobis Distance")
    ax.set_xlabel("PCA Dimension 1", fontsize=12)
    ax.set_ylabel("PCA Dimension 2", fontsize=12)
    ax.set_title("Representation Space (colored by Mahalanobis distance)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "representation_space.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close()


def plot_penalty_heatmap(encoder, distribution, env, save_dir, device):
    """绘制惩罚权重在状态空间中的热力图"""
    # 创建网格
    n_points = 50
    x = np.linspace(-env.maze_size, env.maze_size, n_points)
    y = np.linspace(-env.maze_size, env.maze_size, n_points)
    xx, yy = np.meshgrid(x, y)
    states = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    # 使用零动作
    actions = np.zeros((len(states), 2), dtype=np.float32)

    # 计算惩罚权重
    encoder.eval()
    with torch.no_grad():
        repr_all = encoder.encode(
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
        )
        weights = (
            distribution.compute_penalty_weight(repr_all, beta_min=0.1, beta_max=5.0)
            .cpu()
            .numpy()
        )

    weights_grid = weights.reshape(n_points, n_points)

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(
        weights_grid,
        extent=[-env.maze_size, env.maze_size, -env.maze_size, env.maze_size],
        origin="lower",
        cmap="RdYlGn_r",
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Penalty Weight β(s,a)")

    # 绘制障碍物
    for obs in env.obstacles:
        x_min, x_max, y_min, y_max = obs
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="gray",
            edgecolor="black",
            alpha=0.9,
        )
        ax.add_patch(rect)

    # 绘制目标
    goal_circle = Circle(
        env.goal_pos, env.goal_radius, facecolor="green", edgecolor="black", alpha=0.7
    )
    ax.add_patch(goal_circle)

    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title("Adaptive Penalty Weight Heatmap", fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "penalty_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close()


def plot_penalty_distribution(encoder, distribution, dataset, save_dir, device):
    """绘制分布内外样本的惩罚权重分布对比"""
    # 分布内样本
    in_states = dataset["observations"]
    in_actions = dataset["actions"]

    # 分布外样本 (随机动作)
    ood_states = in_states.copy()
    ood_actions = np.random.uniform(-1, 1, in_actions.shape).astype(np.float32)

    encoder.eval()
    with torch.no_grad():
        in_repr = encoder.encode(
            torch.FloatTensor(in_states).to(device),
            torch.FloatTensor(in_actions).to(device),
        )
        ood_repr = encoder.encode(
            torch.FloatTensor(ood_states).to(device),
            torch.FloatTensor(ood_actions).to(device),
        )

        weights_in = (
            distribution.compute_penalty_weight(in_repr, beta_min=0.1, beta_max=5.0)
            .cpu()
            .numpy()
        )
        weights_ood = (
            distribution.compute_penalty_weight(ood_repr, beta_min=0.1, beta_max=5.0)
            .cpu()
            .numpy()
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 5.5, 50)

    ax.hist(
        weights_in,
        bins=bins,
        alpha=0.6,
        label="In-distribution",
        color="#2ecc71",
        density=True,
    )
    ax.hist(
        weights_ood,
        bins=bins,
        alpha=0.6,
        label="Out-of-distribution",
        color="#e74c3c",
        density=True,
    )

    ax.axvline(
        weights_in.mean(),
        color="#27ae60",
        linestyle="--",
        lw=2,
        label=f"In-dist mean: {weights_in.mean():.2f}",
    )
    ax.axvline(
        weights_ood.mean(),
        color="#c0392b",
        linestyle="--",
        lw=2,
        label=f"OOD mean: {weights_ood.mean():.2f}",
    )

    ax.set_xlabel("Penalty Weight β", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Penalty Weight Distribution: In-distribution vs. OOD", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "penalty_distribution.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="2D 可视化实验")
    parser.add_argument("--n_trajectories", type=int, default=100, help="轨迹数量")
    parser.add_argument(
        "--contrastive_steps", type=int, default=10000, help="对比学习步数"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results/2d_visualization", help="保存目录"
    )
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    args = parser.parse_args()

    run_2d_experiment(
        n_trajectories=args.n_trajectories,
        contrastive_steps=args.contrastive_steps,
        save_dir=args.save_dir,
        device=args.device if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    main()
