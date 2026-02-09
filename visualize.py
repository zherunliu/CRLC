import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Optional, Tuple


# Set font for English labels
plt.rcParams["font.family"] = "DejaVu Sans"
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
    Plot t-SNE visualization of representation space.
    Distinguish in-distribution and out-of-distribution samples.

    Corresponds to Paper Figure: Representation Space t-SNE Visualization

    Args:
        encoder: Pretrained joint encoder
        in_distribution_data: (states, actions) in-distribution data
        ood_data: (states, actions) out-of-distribution data
        device: Computing device
        save_path: Path to save figure (PDF format)
        perplexity: t-SNE perplexity
        n_samples: Number of samples to use
    """
    encoder.eval()

    in_states, in_actions = in_distribution_data
    ood_states, ood_actions = ood_data

    # Sampling
    if len(in_states) > n_samples:
        idx = np.random.choice(len(in_states), n_samples, replace=False)
        in_states, in_actions = in_states[idx], in_actions[idx]

    if len(ood_states) > n_samples:
        idx = np.random.choice(len(ood_states), n_samples, replace=False)
        ood_states, ood_actions = ood_states[idx], ood_actions[idx]

    # Get representations
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

    # Merge data
    all_repr = np.vstack([in_repr, ood_repr])
    labels = np.array([0] * len(in_repr) + [1] * len(ood_repr))

    # t-SNE dimensionality reduction
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    repr_2d = tsne.fit_transform(all_repr)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # In-distribution samples
    in_mask = labels == 0
    ax.scatter(
        repr_2d[in_mask, 0],
        repr_2d[in_mask, 1],
        c="#2ecc71",
        alpha=0.6,
        s=15,
        label="In-distribution",
    )

    # Out-of-distribution samples
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

    if save_path:
        # Save as PDF (vector) and PNG
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return repr_2d, labels


def plot_ood_detection_roc(
    distribution,
    in_distribution_repr: torch.Tensor,
    ood_repr: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Plot OOD detection ROC curve.

    Corresponds to Paper Figure: OOD Detection ROC Curve

    Args:
        distribution: RepresentationDistribution instance
        in_distribution_repr: In-distribution representations
        ood_repr: Out-of-distribution representations
        save_path: Path to save figure

    Returns:
        auc_score: ROC-AUC score
    """
    # Compute OOD scores (Mahalanobis distance)
    in_scores = distribution.ood_score(in_distribution_repr).cpu().numpy()
    ood_scores = distribution.ood_score(ood_repr).cpu().numpy()

    # Build labels (0: in-distribution, 1: OOD)
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([in_scores, ood_scores])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    print(f"OOD Detection AUC: {roc_auc:.4f}")
    return roc_auc


def plot_distance_q_error_correlation(
    distances: np.ndarray,
    q_errors: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 50,
):
    """
    Plot correlation between representation distance and Q-value estimation error.

    Corresponds to Paper Figure: Representation Distance vs. Q-Value Error

    Args:
        distances: Mahalanobis distance array
        q_errors: Q-function error array (|Q_pred - Q_true|)
        save_path: Path to save figure
        n_bins: Number of bins

    Returns:
        correlation: Pearson correlation coefficient
    """
    # Compute correlation coefficients
    pearson_r, pearson_p = pearsonr(distances, q_errors)
    spearman_r, spearman_p = spearmanr(distances, q_errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scatter plot
    ax = axes[0]
    ax.scatter(distances, q_errors, alpha=0.3, s=5, c="#3498db")

    # Add trend line
    z = np.polyfit(distances, q_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(distances.min(), distances.max(), 100)
    ax.plot(x_line, p(x_line), "r-", lw=2, label=f"Linear Fit (r={pearson_r:.3f})")

    ax.set_xlabel("Mahalanobis Distance", fontsize=12)
    ax.set_ylabel("Q-Value Estimation Error", fontsize=12)
    ax.set_title("Distance vs. Q-Error Scatter Plot", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: Binned statistics
    ax = axes[1]

    # Bin by distance
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

    ax.set_xlabel("Mahalanobis Distance", fontsize=12)
    ax.set_ylabel("Mean Q-Value Error", fontsize=12)
    ax.set_title("Binned Statistics", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    print(f"Pearson Correlation: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman Correlation: {spearman_r:.4f} (p={spearman_p:.2e})")

    return pearson_r, spearman_r


def plot_training_curves(train_log: Dict[str, List], save_path: Optional[str] = None):
    """
    Plot training curves.

    Args:
        train_log: Training log dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = train_log.get("step", range(len(train_log.get("critic_loss", []))))

    # Critic loss
    ax = axes[0, 0]
    if "critic_loss" in train_log:
        ax.plot(steps, train_log["critic_loss"], alpha=0.7)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Critic Loss")
        ax.set_title("Critic Loss Curve")
        ax.grid(True, alpha=0.3)

    # Actor loss
    ax = axes[0, 1]
    if "actor_loss" in train_log:
        ax.plot(steps, train_log["actor_loss"], alpha=0.7, color="orange")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Actor Loss")
        ax.set_title("Actor Loss Curve")
        ax.grid(True, alpha=0.3)

    # Penalty weight
    ax = axes[1, 0]
    if "penalty_weight" in train_log:
        ax.plot(steps, train_log["penalty_weight"], alpha=0.7, color="green")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Mean Penalty Weight")
        ax.set_title("Adaptive Penalty Weight")
        ax.grid(True, alpha=0.3)

    # Evaluation return
    ax = axes[1, 1]
    if "eval_return" in train_log and len(train_log["eval_return"]) > 0:
        eval_steps = np.linspace(0, steps[-1], len(train_log["eval_return"]))
        ax.plot(eval_steps, train_log["eval_return"], "o-", color="red")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Evaluation Return")
        ax.set_title("Evaluation Return Curve")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_penalty_weight_distribution(
    weights_in: np.ndarray, weights_ood: np.ndarray, save_path: Optional[str] = None
):
    """
    Plot penalty weight distribution comparison.

    Args:
        weights_in: Penalty weights for in-distribution samples
        weights_ood: Penalty weights for out-of-distribution samples
        save_path: Path to save figure
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
        label=f"In-dist Mean: {weights_in.mean():.2f}",
    )
    ax.axvline(
        weights_ood.mean(),
        color="#c0392b",
        linestyle="--",
        lw=2,
        label=f"OOD Mean: {weights_ood.mean():.2f}",
    )

    ax.set_xlabel("Penalty Weight β", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Adaptive Penalty Weight Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_ablation_results(
    results: Dict[str, List[float]],
    metric_name: str = "Normalized Return",
    save_path: Optional[str] = None,
):
    """
    Plot ablation study results.

    Args:
        results: Dictionary with method names as keys and result lists as values
        metric_name: Name of the metric
        save_path: Path to save figure
    """
    methods = list(results.keys())
    means = [np.mean(results[m]) for m in methods]
    stds = [np.std(results[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(methods)],
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Ablation Study: {metric_name}", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_baseline_comparison(
    crlc_results: Dict[str, Tuple[float, float]],
    baseline_results: Dict[str, Dict[str, Tuple[float, float]]],
    task_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Plot comparison with baseline methods across multiple tasks.

    Args:
        crlc_results: CRLC results {task: (mean, std)}
        baseline_results: Baseline results {method: {task: (mean, std)}}
        task_names: List of task names to compare
        save_path: Path to save figure
    """
    methods = ["CRLC"] + list(baseline_results.keys())
    n_tasks = len(task_names)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_tasks)
    width = 0.8 / n_methods
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, method in enumerate(methods):
        means = []
        stds = []
        for task in task_names:
            if method == "CRLC":
                if task in crlc_results:
                    means.append(crlc_results[task][0])
                    stds.append(crlc_results[task][1])
                else:
                    means.append(0)
                    stds.append(0)
            else:
                if task in baseline_results.get(method, {}):
                    means.append(baseline_results[method][task][0])
                    stds.append(baseline_results[method][task][1])
                else:
                    means.append(0)
                    stds.append(0)

        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=method,
            color=colors[i % len(colors)],
            alpha=0.8,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("antmaze-", "").replace("-v2", "") for t in task_names],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Normalized Return (%)", fontsize=12)
    ax.set_title("Comparison with Baseline Methods on D4RL Antmaze", fontsize=14)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_2d_trajectory_analysis(
    states: np.ndarray,
    actions: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray,
    goal_position: Tuple[float, float] = None,
    save_path: Optional[str] = None,
):
    """
    Plot 2D trajectory analysis for Point environment visualization.

    Args:
        states: 2D state positions (N, 2)
        actions: Actions taken (N, 2)
        distances: Mahalanobis distances (N,)
        weights: Penalty weights (N,)
        goal_position: Optional goal position (x, y)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Data Distribution
    ax = axes[0]
    ax.scatter(states[:, 0], states[:, 1], c="#3498db", alpha=0.3, s=5)
    if goal_position:
        ax.scatter(*goal_position, c="red", s=100, marker="*", label="Goal")
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title("Offline Dataset Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    if goal_position:
        ax.legend()

    # Plot 2: Penalty Weight Heatmap
    ax = axes[1]
    sc = ax.scatter(
        states[:, 0],
        states[:, 1],
        c=weights,
        cmap="RdYlGn_r",
        alpha=0.5,
        s=10,
        vmin=0,
        vmax=weights.max(),
    )
    plt.colorbar(sc, ax=ax, label="Penalty Weight β")
    if goal_position:
        ax.scatter(*goal_position, c="blue", s=100, marker="*")
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title("Penalty Weight Spatial Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Plot 3: Distance Distribution
    ax = axes[2]
    ax.hist(distances, bins=50, color="#3498db", alpha=0.7, edgecolor="black")
    ax.axvline(
        np.median(distances),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(distances):.2f}",
    )
    ax.set_xlabel("Mahalanobis Distance", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distance Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization tools...")

    # Simulated data
    n_samples = 1000

    # Simulated representation distances and Q-errors
    distances = np.abs(np.random.randn(n_samples)) * 2 + 1
    q_errors = distances * 0.5 + np.random.randn(n_samples) * 0.5 + 1
    q_errors = np.abs(q_errors)

    # Test correlation analysis
    plot_distance_q_error_correlation(
        distances, q_errors, save_path="./test_correlation.pdf"
    )

    # Simulated penalty weights
    weights_in = np.random.beta(2, 5, n_samples) * 3 + 0.1
    weights_ood = np.random.beta(5, 2, n_samples) * 4 + 1

    # Test weight distribution
    plot_penalty_weight_distribution(
        weights_in, weights_ood, save_path="./test_weights.pdf"
    )

    # Test ablation results
    ablation_results = {
        "CRLC (Full)": [85.2, 87.1, 84.5, 86.3, 85.8],
        "w/o Contrastive": [72.3, 71.8, 73.5, 72.1, 71.9],
        "w/o Adaptive": [78.5, 79.2, 77.8, 78.9, 78.1],
        "CQL": [68.2, 69.1, 67.5, 68.8, 68.0],
    }

    plot_ablation_results(
        ablation_results,
        metric_name="Normalized Return (%)",
        save_path="./test_ablation.pdf",
    )

    print("Visualization tools test completed!")
