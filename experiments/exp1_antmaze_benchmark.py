"""
实验1: D4RL Antmaze 基准测试

运行方式:
    python experiments/exp1_antmaze_benchmark.py --env antmaze-medium-diverse-v2 --seed 0

批量运行:
    python experiments/exp1_antmaze_benchmark.py --run_all --seeds 0 1 2
"""

import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Antmaze 任务列表
ANTMAZE_TASKS = [
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
]


def run_single_experiment(
    env_name: str,
    seed: int,
    contrastive_steps: int = 50000,
    policy_steps: int = 1000000,
    save_dir: str = "./results/antmaze",
):
    """运行单个实验"""
    import torch
    from train_crlc import train_crlc

    # 创建保存目录
    exp_dir = os.path.join(save_dir, env_name, f"seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running: {env_name}, Seed: {seed}")
    print(f"{'=' * 60}")

    # 运行训练
    agent, train_log = train_crlc(
        env_name=env_name,
        seed=seed,
        contrastive_steps=contrastive_steps,
        policy_steps=policy_steps,
        log_dir=os.path.join(exp_dir, "logs"),
        save_dir=os.path.join(exp_dir, "checkpoints"),
        eval_freq=10000,
        eval_episodes=20,
    )

    # 保存结果
    results = {
        "env_name": env_name,
        "seed": seed,
        "final_return": train_log["eval_return"][-1] if train_log["eval_return"] else 0,
        "best_return": max(train_log["eval_return"]) if train_log["eval_return"] else 0,
        "final_success": train_log["eval_success"][-1]
        if train_log.get("eval_success")
        else 0,
        "best_success": max(train_log.get("eval_success", [0])),
    }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_results(save_dir: str = "./results/antmaze"):
    """汇总所有实验结果"""
    all_results = {}

    for env_name in ANTMAZE_TASKS:
        env_dir = os.path.join(save_dir, env_name)
        if not os.path.exists(env_dir):
            continue

        returns = []
        successes = []

        for seed_dir in os.listdir(env_dir):
            if seed_dir.startswith("seed_"):
                result_file = os.path.join(env_dir, seed_dir, "results.json")
                if os.path.exists(result_file):
                    with open(result_file) as f:
                        result = json.load(f)
                    returns.append(result["best_return"])
                    successes.append(result.get("best_success", 0))

        if returns:
            all_results[env_name] = {
                "return_mean": np.mean(returns),
                "return_std": np.std(returns),
                "success_mean": np.mean(successes) * 100,  # 转换为百分比
                "success_std": np.std(successes) * 100,
                "n_seeds": len(returns),
            }

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("CRLC Results on D4RL Antmaze Benchmark")
    print("=" * 80)
    print(f"{'Task':<35} {'Return':<20} {'Success Rate':<20}")
    print("-" * 80)

    for env_name, res in all_results.items():
        print(
            f"{env_name:<35} "
            f"{res['return_mean']:.1f} ± {res['return_std']:.1f}    "
            f"{res['success_mean']:.1f}% ± {res['success_std']:.1f}%"
        )

    # 保存汇总结果
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Antmaze Benchmark Experiments")
    parser.add_argument("--env", type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--run_all", action="store_true", help="Run all tasks")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results")
    parser.add_argument("--contrastive_steps", type=int, default=50000)
    parser.add_argument("--policy_steps", type=int, default=1000000)
    parser.add_argument("--save_dir", type=str, default="./results/antmaze")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results(args.save_dir)
        return

    if args.run_all:
        # 运行所有任务的所有种子
        for env_name in ANTMAZE_TASKS:
            for seed in args.seeds:
                run_single_experiment(
                    env_name=env_name,
                    seed=seed,
                    contrastive_steps=args.contrastive_steps,
                    policy_steps=args.policy_steps,
                    save_dir=args.save_dir,
                )
        # 汇总结果
        aggregate_results(args.save_dir)
    else:
        # 运行单个实验
        run_single_experiment(
            env_name=args.env,
            seed=args.seed,
            contrastive_steps=args.contrastive_steps,
            policy_steps=args.policy_steps,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
