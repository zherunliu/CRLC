"""
CRLC 可视化工具
用于论文实验分析和结果展示
包含：t-SNE表征可视化、OOD检测ROC曲线、表征距离-Q误差相关性分析
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Optional, Tuple
import os


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_tsne_representation(
    encoder,
    in_distribution_data: Tuple[np.ndarray, np.ndarray],
    ood_data: Tuple[np.ndarray, np.ndarray],
    device: str = "cuda",
    save_path: Optional[str] = None,
    perplexity: int = 30,
    n_samples: int = 2000,
):
    """
    绘制表征空间的 t-SNE 可视化
    区分分布内和分布外样本

    对应论文图3.2: 表征空间t-SNE可视化

    Args:
        encoder: 预训练的联合编码器
        in_distribution_data: (states, actions) 分布内数据
        ood_data: (states, actions) 分布外数据
        device: 计算设备
        save_path: 图像保存路径
        perplexity: t-SNE 困惑度
        n_samples: 采样数量
    """
    encoder.eval()

    in_states, in_actions = in_distribution_data
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

    # 合并数据
    all_repr = np.vstack([in_repr, ood_repr])
    labels = np.array([0] * len(in_repr) + [1] * len(ood_repr))

    # t-SNE 降维
    print("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    repr_2d = tsne.fit_transform(all_repr)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 分布内样本
    in_mask = labels == 0
    ax.scatter(
        repr_2d[in_mask, 0],
        repr_2d[in_mask, 1],
        c="#2ecc71",
        alpha=0.6,
        s=15,
        label="分布内样本 (In-distribution)",
    )

    # 分布外样本
    ood_mask = labels == 1
    ax.scatter(
        repr_2d[ood_mask, 0],
        repr_2d[ood_mask, 1],
        c="#e74c3c",
        alpha=0.6,
        s=15,
        label="分布外样本 (OOD)",
    )

    ax.set_xlabel("t-SNE 维度 1", fontsize=12)
    ax.set_ylabel("t-SNE 维度 2", fontsize=12)
    ax.set_title("表征空间 t-SNE 可视化", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像保存到 {save_path}")

    plt.show()
    return repr_2d, labels


def plot_ood_detection_roc(
    distribution,
    in_distribution_repr: torch.Tensor,
    ood_repr: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    绘制 OOD 检测的 ROC 曲线

    对应论文图3.3: OOD检测ROC曲线

    Args:
        distribution: RepresentationDistribution 实例
        in_distribution_repr: 分布内表征
        ood_repr: 分布外表征
        save_path: 图像保存路径

    Returns:
        auc_score: ROC-AUC 分数
    """
    # 计算 OOD 分数（马氏距离）
    in_scores = distribution.ood_score(in_distribution_repr).cpu().numpy()
    ood_scores = distribution.ood_score(ood_repr).cpu().numpy()

    # 构建标签（0: 分布内, 1: OOD）
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([in_scores, ood_scores])

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC 曲线 (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="随机分类器")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("假阳性率 (FPR)", fontsize=12)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=12)
    ax.set_title("OOD 检测 ROC 曲线", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像保存到 {save_path}")

    plt.show()

    print(f"OOD 检测 AUC: {roc_auc:.4f}")
    return roc_auc


def plot_distance_q_error_correlation(
    distances: np.ndarray,
    q_errors: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 50,
):
    """
    绘制表征距离与 Q 函数估计误差的相关性分析

    对应论文图3.4: 表征距离与Q值误差相关性

    Args:
        distances: 马氏距离数组
        q_errors: Q 函数误差数组（|Q_pred - Q_true|）
        save_path: 图像保存路径
        n_bins: 分箱数量

    Returns:
        correlation: 皮尔逊相关系数
    """
    # 计算相关系数
    pearson_r, pearson_p = pearsonr(distances, q_errors)
    spearman_r, spearman_p = spearmanr(distances, q_errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：散点图
    ax = axes[0]
    ax.scatter(distances, q_errors, alpha=0.3, s=5, c="#3498db")

    # 添加趋势线
    z = np.polyfit(distances, q_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(distances.min(), distances.max(), 100)
    ax.plot(x_line, p(x_line), "r-", lw=2, label=f"拟合线 (r={pearson_r:.3f})")

    ax.set_xlabel("马氏距离", fontsize=12)
    ax.set_ylabel("Q 函数估计误差", fontsize=12)
    ax.set_title("表征距离与 Q 值误差的关系", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 右图：分箱统计
    ax = axes[1]

    # 按距离分箱
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(distances, bin_edges[:-1])

    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(q_errors[mask].mean())
            bin_stds.append(q_errors[mask].std())
            bin_centers.append((bin_edges[i - 1] + bin_edges[i]) / 2)

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_stds,
        fmt="o-",
        capsize=3,
        capthick=1,
        color="#e74c3c",
        ecolor="gray",
        alpha=0.7,
    )

    ax.set_xlabel("马氏距离", fontsize=12)
    ax.set_ylabel("平均 Q 函数误差", fontsize=12)
    ax.set_title("分箱统计结果", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像保存到 {save_path}")

    plt.show()

    print(f"Pearson 相关系数: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman 相关系数: {spearman_r:.4f} (p={spearman_p:.2e})")

    return pearson_r, spearman_r


def plot_training_curves(train_log: Dict[str, List], save_path: Optional[str] = None):
    """
    绘制训练曲线

    Args:
        train_log: 训练日志字典
        save_path: 图像保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = train_log.get("step", range(len(train_log.get("critic_loss", []))))

    # Critic 损失
    ax = axes[0, 0]
    if "critic_loss" in train_log:
        ax.plot(steps, train_log["critic_loss"], alpha=0.7)
        ax.set_xlabel("训练步数")
        ax.set_ylabel("Critic 损失")
        ax.set_title("Critic 损失曲线")
        ax.grid(True, alpha=0.3)

    # Actor 损失
    ax = axes[0, 1]
    if "actor_loss" in train_log:
        ax.plot(steps, train_log["actor_loss"], alpha=0.7, color="orange")
        ax.set_xlabel("训练步数")
        ax.set_ylabel("Actor 损失")
        ax.set_title("Actor 损失曲线")
        ax.grid(True, alpha=0.3)

    # 惩罚权重
    ax = axes[1, 0]
    if "penalty_weight" in train_log:
        ax.plot(steps, train_log["penalty_weight"], alpha=0.7, color="green")
        ax.set_xlabel("训练步数")
        ax.set_ylabel("平均惩罚权重")
        ax.set_title("自适应惩罚权重变化")
        ax.grid(True, alpha=0.3)

    # 评估回报
    ax = axes[1, 1]
    if "eval_return" in train_log and len(train_log["eval_return"]) > 0:
        eval_steps = np.linspace(0, steps[-1], len(train_log["eval_return"]))
        ax.plot(eval_steps, train_log["eval_return"], "o-", color="red")
        ax.set_xlabel("训练步数")
        ax.set_ylabel("评估回报")
        ax.set_title("评估回报曲线")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像保存到 {save_path}")

    plt.show()


def plot_penalty_weight_distribution(
    weights_in: np.ndarray, weights_ood: np.ndarray, save_path: Optional[str] = None
):
    """
    绘制惩罚权重分布对比

    Args:
        weights_in: 分布内样本的惩罚权重
        weights_ood: 分布外样本的惩罚权重
        save_path: 图像保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(weights_in.min(), weights_ood.min()),
        max(weights_in.max(), weights_ood.max()),
        50,
    )

    ax.hist(
        weights_in,
        bins=bins,
        alpha=0.6,
        label="分布内样本",
        color="#2ecc71",
        density=True,
    )
    ax.hist(
        weights_ood,
        bins=bins,
        alpha=0.6,
        label="分布外样本",
        color="#e74c3c",
        density=True,
    )

    ax.axvline(
        weights_in.mean(),
        color="#27ae60",
        linestyle="--",
        lw=2,
        label=f"分布内均值: {weights_in.mean():.2f}",
    )
    ax.axvline(
        weights_ood.mean(),
        color="#c0392b",
        linestyle="--",
        lw=2,
        label=f"分布外均值: {weights_ood.mean():.2f}",
    )

    ax.set_xlabel("惩罚权重 β", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.set_title("自适应惩罚权重分布对比", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像保存到 {save_path}")

    plt.show()


def plot_ablation_results(
    results: Dict[str, List[float]],
    metric_name: str = "回报",
    save_path: Optional[str] = None,
):
    """
    绘制消融实验结果

    Args:
        results: 字典，键为方法名，值为多次运行结果列表
        metric_name: 指标名称
        save_path: 图像保存路径
    """
    methods = list(results.keys())
    means = [np.mean(results[m]) for m in methods]
    stds = [np.std(results[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    bars = ax.bar(
        x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"消融实验: {metric_name}", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.5,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像保存到 {save_path}")

    plt.show()


if __name__ == "__main__":
    # 测试可视化功能
    print("测试可视化工具...")

    # 模拟数据
    n_samples = 1000

    # 模拟表征距离和Q误差
    distances = np.abs(np.random.randn(n_samples)) * 2 + 1
    q_errors = distances * 0.5 + np.random.randn(n_samples) * 0.5 + 1
    q_errors = np.abs(q_errors)

    # 测试相关性分析
    plot_distance_q_error_correlation(
        distances, q_errors, save_path="./test_correlation.png"
    )

    # 模拟惩罚权重
    weights_in = np.random.beta(2, 5, n_samples) * 3 + 0.1
    weights_ood = np.random.beta(5, 2, n_samples) * 4 + 1

    # 测试权重分布
    plot_penalty_weight_distribution(
        weights_in, weights_ood, save_path="./test_weights.png"
    )

    # 测试消融实验图
    ablation_results = {
        "CRLC (完整)": [85.2, 87.1, 84.5, 86.3, 85.8],
        "w/o 对比学习": [72.3, 71.8, 73.5, 72.1, 71.9],
        "w/o 自适应惩罚": [78.5, 79.2, 77.8, 78.9, 78.1],
        "CQL (基线)": [68.2, 69.1, 67.5, 68.8, 68.0],
    }

    plot_ablation_results(
        ablation_results, metric_name="归一化回报 (%)", save_path="./test_ablation.png"
    )

    print("可视化工具测试完成！")
