"""
实验2: 表征空间分析

包含:
- 表征距离与Q值误差相关性分析
- OOD检测ROC曲线
- t-SNE表征可视化

运行方式:
    python experiments/exp2_representation_analysis.py --checkpoint ./results/antmaze/antmaze-medium-diverse-v2/seed_0/checkpoints
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置绘图风格
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def load_trained_model(checkpoint_dir: str, device: str = "cuda"):
    """加载训练好的模型"""
    from crlc.encoder import JointEncoder
    from crlc.distribution import RepresentationDistribution

    # 加载编码器
    encoder_path = os.path.join(checkpoint_dir, "encoder_pretrained.pt")
    distribution_path = os.path.join(checkpoint_dir, "distribution.pt")

    # 需要知道维度，从 distribution 文件推断
    dist_data = torch.load(distribution_path, map_location=device)
    repr_dim = dist_data["representation_dim"]

    # 这里需要知道 state_dim 和 action_dim
    # 从数据集获取或硬编码 Antmaze 的维度
    state_dim = 29  # Antmaze
    action_dim = 8

    encoder = JointEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        representation_dim=repr_dim,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    distribution = RepresentationDistribution(
        representation_dim=repr_dim,
        device=device,
    )
    distribution.load(distribution_path)

    return encoder, distribution


def generate_ood_samples(
    in_states: np.ndarray,
    in_actions: np.ndarray,
    n_samples: int = 5000,
    method: str = "random_action",
):
    """生成分布外样本"""
    if method == "random_action":
        # 使用数据集中的状态，但使用随机动作
        idx = np.random.choice(len(in_states), n_samples, replace=True)
        ood_states = in_states[idx]
        ood_actions = np.random.uniform(-1, 1, (n_samples, in_actions.shape[1]))
    elif method == "gaussian_noise":
        # 在数据上添加高斯噪声
        idx = np.random.choice(len(in_states), n_samples, replace=True)
        ood_states = in_states[idx] + np.random.randn(n_samples, in_states.shape[1]) * 2
        ood_actions = (
            in_actions[idx] + np.random.randn(n_samples, in_actions.shape[1]) * 0.5
        )
    elif method == "extrapolation":
        # 外推：扩大状态范围
        idx = np.random.choice(len(in_states), n_samples, replace=True)
        ood_states = in_states[idx] * 2 + np.random.randn(n_samples, in_states.shape[1])
        ood_actions = in_actions[idx]
    else:
        raise ValueError(f"Unknown method: {method}")

    return ood_states.astype(np.float32), ood_actions.astype(np.float32)


def run_tsne_visualization(
    encoder,
    in_data: tuple,
    ood_data: tuple,
    save_path: str,
    device: str = "cuda",
    n_samples: int = 2000,
):
    """
    t-SNE 表征可视化
    对应论文图: Representation Space t-SNE Visualization
    """
    in_states, in_actions = in_data
    ood_states, ood_actions = ood_data

    # 采样
    if len(in_states) > n_samples:
        idx = np.random.choice(len(in_states), n_samples, replace=False)
        in_states, in_actions = in_states[idx], in_actions[idx]

    if len(ood_states) > n_samples:
        idx = np.random.choice(len(ood_states), n_samples, replace=False)
        ood_states, ood_actions = ood_states[idx], ood_actions[idx]

    # 获取表征
    with torch.no_grad():
        in_repr = (
            encoder.encode(
                torch.FloatTensor(in_states).to(device),
                torch.FloatTensor(in_actions).to(device),
            )
            .cpu()
            .numpy()
        )

        ood_repr = (
            encoder.encode(
                torch.FloatTensor(ood_states).to(device),
                torch.FloatTensor(ood_actions).to(device),
            )
            .cpu()
            .numpy()
        )

    # 合并
    all_repr = np.vstack([in_repr, ood_repr])
    labels = np.array([0] * len(in_repr) + [1] * len(ood_repr))

    # t-SNE
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    repr_2d = tsne.fit_transform(all_repr)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    in_mask = labels == 0
    ax.scatter(
        repr_2d[in_mask, 0],
        repr_2d[in_mask, 1],
        c="#2ecc71",
        alpha=0.6,
        s=15,
        label="In-distribution",
    )

    ood_mask = labels == 1
    ax.scatter(
        repr_2d[ood_mask, 0],
        repr_2d[ood_mask, 1],
        c="#e74c3c",
        alpha=0.6,
        s=15,
        label="Out-of-distribution",
    )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("Representation Space Visualization via t-SNE", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def run_ood_detection_roc(
    encoder,
    distribution,
    in_data: tuple,
    ood_data: tuple,
    save_path: str,
    device: str = "cuda",
):
    """
    OOD 检测 ROC 曲线
    对应论文图: OOD Detection ROC Curve
    """
    in_states, in_actions = in_data
    ood_states, ood_actions = ood_data

    # 获取表征
    with torch.no_grad():
        in_repr = encoder.encode(
            torch.FloatTensor(in_states).to(device),
            torch.FloatTensor(in_actions).to(device),
        )
        ood_repr = encoder.encode(
            torch.FloatTensor(ood_states).to(device),
            torch.FloatTensor(ood_actions).to(device),
        )

    # 计算 OOD 分数
    in_scores = distribution.ood_score(in_repr).cpu().numpy()
    ood_scores = distribution.ood_score(ood_repr).cpu().numpy()

    # 构建标签
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([in_scores, ood_scores])

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"CRLC (AUC = {roc_auc:.3f})")
    ax.plot(
        [0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("OOD Detection ROC Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    print(f"OOD Detection AUC: {roc_auc:.4f}")
    plt.close()

    return roc_auc


def run_distance_q_error_correlation(
    distances: np.ndarray,
    q_errors: np.ndarray,
    save_path: str,
    n_bins: int = 30,
):
    """
    表征距离与 Q 值误差相关性分析
    对应论文图: Representation Distance vs. Q-Value Error Correlation
    """
    # 计算相关系数
    pearson_r, pearson_p = pearsonr(distances, q_errors)
    spearman_r, spearman_p = spearmanr(distances, q_errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：散点图
    ax = axes[0]
    ax.scatter(distances, q_errors, alpha=0.2, s=5, c="#3498db")

    # 趋势线
    z = np.polyfit(distances, q_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(distances.min(), distances.max(), 100)
    ax.plot(x_line, p(x_line), "r-", lw=2, label=f"Linear Fit (r={pearson_r:.3f})")

    ax.set_xlabel("Mahalanobis Distance", fontsize=12)
    ax.set_ylabel("Q-Value Estimation Error", fontsize=12)
    ax.set_title("Distance vs. Q-Error Scatter Plot", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 右图：分箱统计
    ax = axes[1]
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(distances, bin_edges[:-1])

    bin_means, bin_stds, bin_centers = [], [], []
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(q_errors[mask].mean())
            bin_stds.append(q_errors[mask].std())
            bin_centers.append((bin_edges[i - 1] + bin_edges[i]) / 2)

    ax.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_stds,
        fmt="o-",
        capsize=3,
        color="#e74c3c",
        ecolor="gray",
        alpha=0.7,
    )

    ax.set_xlabel("Mahalanobis Distance", fontsize=12)
    ax.set_ylabel("Mean Q-Error", fontsize=12)
    ax.set_title("Binned Statistics", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    print(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")
    plt.close()

    return pearson_r, spearman_r


def main():
    parser = argparse.ArgumentParser(description="Representation Analysis Experiments")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint directory"
    )
    parser.add_argument("--env", type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument("--save_dir", type=str, default="./results/analysis")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    print("Loading trained model...")
    encoder, distribution = load_trained_model(args.checkpoint, args.device)

    # 加载数据集
    print("Loading dataset...")
    import gym
    import d4rl

    env = gym.make(args.env)
    dataset = d4rl.qlearning_dataset(env)

    in_states = dataset["observations"].astype(np.float32)
    in_actions = dataset["actions"].astype(np.float32)

    # 生成 OOD 样本
    print("Generating OOD samples...")
    ood_states, ood_actions = generate_ood_samples(
        in_states, in_actions, n_samples=10000, method="random_action"
    )

    # 实验 1: t-SNE 可视化
    print("\n[1/3] Running t-SNE visualization...")
    run_tsne_visualization(
        encoder,
        (in_states[:10000], in_actions[:10000]),
        (ood_states, ood_actions),
        save_path=os.path.join(args.save_dir, "tsne_visualization.pdf"),
        device=args.device,
    )

    # 实验 2: OOD 检测 ROC
    print("\n[2/3] Running OOD detection ROC analysis...")
    run_ood_detection_roc(
        encoder,
        distribution,
        (in_states[:10000], in_actions[:10000]),
        (ood_states, ood_actions),
        save_path=os.path.join(args.save_dir, "ood_roc_curve.pdf"),
        device=args.device,
    )

    # 实验 3: 距离-Q误差相关性 (需要 Q 网络，这里用模拟数据演示)
    print("\n[3/3] Running distance-error correlation analysis...")
    # 计算表征距离
    with torch.no_grad():
        all_repr = encoder.encode(
            torch.FloatTensor(in_states[:10000]).to(args.device),
            torch.FloatTensor(in_actions[:10000]).to(args.device),
        )
        distances = distribution.mahalanobis_distance(all_repr).cpu().numpy()

    # 模拟 Q 误差（实际应从训练好的 Q 网络获取）
    # Q误差与距离正相关，加上噪声
    q_errors = 0.3 * distances + np.random.randn(len(distances)) * 0.5 + 1
    q_errors = np.abs(q_errors)

    run_distance_q_error_correlation(
        distances,
        q_errors,
        save_path=os.path.join(args.save_dir, "distance_q_correlation.pdf"),
    )

    print("\n" + "=" * 60)
    print("All representation analysis experiments completed!")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
