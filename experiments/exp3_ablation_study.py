"""
实验3: 消融实验

消融设置:
1. CRLC (Full): 完整算法
2. w/o Contrastive: 不使用对比预训练
3. w/o Adaptive: 使用固定惩罚权重
4. Fixed β_max: 固定使用最大惩罚
5. Fixed β_min: 固定使用最小惩罚

运行方式:
    python experiments/exp3_ablation_study.py --env antmaze-medium-diverse-v2

仅绘图（使用已有结果）:
    python experiments/exp3_ablation_study.py --plot_only
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


ABLATION_CONFIGS = {
    "CRLC (Full)": {
        "use_contrastive": True,
        "use_adaptive": True,
        "fixed_beta": None,
    },
    "w/o Contrastive": {
        "use_contrastive": False,
        "use_adaptive": True,
        "fixed_beta": None,
    },
    "w/o Adaptive": {
        "use_contrastive": True,
        "use_adaptive": False,
        "fixed_beta": 2.5,  # (β_min + β_max) / 2
    },
    "Fixed β=β_max": {
        "use_contrastive": True,
        "use_adaptive": False,
        "fixed_beta": 5.0,
    },
    "Fixed β=β_min": {
        "use_contrastive": True,
        "use_adaptive": False,
        "fixed_beta": 0.1,
    },
}


def run_ablation_variant(
    env_name: str,
    seed: int,
    config_name: str,
    config: dict,
    policy_steps: int = 500000,
    save_dir: str = "./results/ablation",
):
    """运行单个消融变体"""
    from train_crlc import train_crlc

    exp_dir = os.path.join(
        save_dir, env_name, config_name.replace(" ", "_"), f"seed_{seed}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Ablation: {config_name}")
    print(f"Env: {env_name}, Seed: {seed}")
    print(f"Config: {config}")
    print(f"{'=' * 60}")

    # 根据配置调整训练参数
    contrastive_steps = 50000 if config["use_contrastive"] else 0

    # 运行训练
    agent, train_log = train_crlc(
        env_name=env_name,
        seed=seed,
        contrastive_steps=contrastive_steps,
        policy_steps=policy_steps,
        beta_min=config["fixed_beta"] if config["fixed_beta"] else 0.1,
        beta_max=config["fixed_beta"] if config["fixed_beta"] else 5.0,
        log_dir=os.path.join(exp_dir, "logs"),
        save_dir=os.path.join(exp_dir, "checkpoints"),
        eval_freq=10000,
    )

    # 保存结果
    results = {
        "env_name": env_name,
        "seed": seed,
        "config_name": config_name,
        "config": config,
        "final_return": train_log["eval_return"][-1] if train_log["eval_return"] else 0,
        "best_return": max(train_log["eval_return"]) if train_log["eval_return"] else 0,
    }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_and_plot(
    env_name: str,
    save_dir: str = "./results/ablation",
):
    """汇总并绘制消融实验结果"""
    results = {}

    for config_name in ABLATION_CONFIGS.keys():
        config_dir = os.path.join(save_dir, env_name, config_name.replace(" ", "_"))
        if not os.path.exists(config_dir):
            continue

        returns = []
        for seed_dir in os.listdir(config_dir):
            if seed_dir.startswith("seed_"):
                result_file = os.path.join(config_dir, seed_dir, "results.json")
                if os.path.exists(result_file):
                    with open(result_file) as f:
                        result = json.load(f)
                    returns.append(result["best_return"])

        if returns:
            results[config_name] = {
                "mean": np.mean(returns),
                "std": np.std(returns),
                "values": returns,
            }

    if not results:
        print("No results found!")
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(results.keys())
    means = [results[m]["mean"] for m in methods]
    stds = [results[m]["std"] for m in methods]

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
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Normalized Return", fontsize=12)
    ax.set_title(f"Ablation Study on {env_name}", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 1,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"ablation_{env_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # 打印表格
    print("\n" + "=" * 60)
    print(f"Ablation Results on {env_name}")
    print("=" * 60)
    print(f"{'Method':<25} {'Return':<20}")
    print("-" * 60)
    for method, res in results.items():
        print(f"{method:<25} {res['mean']:.1f} ± {res['std']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Experiments")
    parser.add_argument("--env", type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--policy_steps", type=int, default=500000)
    parser.add_argument("--save_dir", type=str, default="./results/ablation")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        aggregate_and_plot(args.env, args.save_dir)
        return

    # 运行所有消融变体
    for config_name, config in ABLATION_CONFIGS.items():
        for seed in args.seeds:
            run_ablation_variant(
                env_name=args.env,
                seed=seed,
                config_name=config_name,
                config=config,
                policy_steps=args.policy_steps,
                save_dir=args.save_dir,
            )

    # 汇总结果
    aggregate_and_plot(args.env, args.save_dir)


if __name__ == "__main__":
    main()
