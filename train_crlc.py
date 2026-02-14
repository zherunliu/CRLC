"""
CRLC 完整训练流程
包含对比学习预训练和策略优化两个阶段
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crlc.encoder import JointEncoder
from crlc.contrastive import ContrastiveLearner, TrajectoryBuffer
from crlc.distribution import RepresentationDistribution
from crlc.crlc_sac import CRLC_SAC
from crlc.buffer import OfflineReplayBuffer


def load_d4rl_dataset(env_name: str):
    """加载 D4RL 数据集"""
    import gym
    import d4rl

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    return env, dataset


def evaluate_policy(env, agent: CRLC_SAC, num_episodes: int = 10) -> Dict[str, float]:
    """评估策略"""
    returns = []
    successes = []

    for _ in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_return = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            result = env.step(action)

            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result

            episode_return += reward
            state = next_state

        returns.append(episode_return)

        # Antmaze 任务的成功判定
        if "antmaze" in env.spec.id.lower():
            successes.append(float(episode_return > 0))
        else:
            successes.append(0.0)

    return {
        "return_mean": np.mean(returns),
        "return_std": np.std(returns),
        "success_rate": np.mean(successes) if successes[0] > 0 else 0.0,
    }


def train_crlc(
    env_name: str = "antmaze-medium-diverse-v2",
    seed: int = 42,
    # 编码器参数
    representation_dim: int = 128,
    projection_dim: int = 64,
    # 对比学习参数
    contrastive_steps: int = 50000,
    contrastive_batch_size: int = 256,
    contrastive_lr: float = 3e-4,
    temperature: float = 0.1,
    trajectory_window: int = 5,
    num_negatives: int = 256,
    # 策略优化参数
    policy_steps: int = 1000000,
    policy_batch_size: int = 256,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    # 保守惩罚参数
    beta_min: float = 0.1,
    beta_max: float = 5.0,
    distance_scale: float = 1.0,
    # 日志参数
    log_freq: int = 1000,
    eval_freq: int = 5000,
    eval_episodes: int = 10,
    save_freq: int = 50000,
    log_dir: str = "./logs",
    save_dir: str = "./checkpoints",
    # 其他
    device: str = "cuda",
):
    """
    完整的 CRLC 训练流程

    阶段 1: 对比学习预训练编码器
    阶段 2: 基于表征的保守策略优化
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{env_name}_{timestamp}"
    log_path = os.path.join(log_dir, run_name)
    save_path = os.path.join(save_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # 保存配置
    config = {
        "env_name": env_name,
        "seed": seed,
        "representation_dim": representation_dim,
        "contrastive_steps": contrastive_steps,
        "policy_steps": policy_steps,
        "beta_min": beta_min,
        "beta_max": beta_max,
    }
    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"CRLC 训练开始")
    print(f"环境: {env_name}")
    print(f"日志目录: {log_path}")
    print("=" * 60)

    # ========== 加载数据 ==========
    print("\n[1/5] 加载 D4RL 数据集...")
    env, dataset = load_d4rl_dataset(env_name)

    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"数据集大小: {len(dataset['observations'])}")

    # 创建离线缓冲区
    buffer = OfflineReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=len(dataset["observations"]),
        device=device,
    )
    buffer.add_batch(
        states=dataset["observations"].astype(np.float32),
        actions=dataset["actions"].astype(np.float32),
        rewards=dataset["rewards"].astype(np.float32),
        next_states=dataset["next_observations"].astype(np.float32),
        dones=dataset["terminals"].astype(np.float32),
    )

    # 创建轨迹缓冲区
    traj_buffer = TrajectoryBuffer(
        states=dataset["observations"].astype(np.float32),
        actions=dataset["actions"].astype(np.float32),
        rewards=dataset["rewards"].astype(np.float32),
        next_states=dataset["next_observations"].astype(np.float32),
        dones=dataset["terminals"].astype(np.float32),
        trajectory_window=trajectory_window,
    )

    # ========== 创建编码器 ==========
    print("\n[2/5] 创建联合编码器...")
    encoder = JointEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        representation_dim=representation_dim,
        projection_dim=projection_dim,
    ).to(device)

    # ========== 对比学习预训练 ==========
    print("\n[3/5] 对比学习预训练...")
    contrastive_learner = ContrastiveLearner(
        encoder=encoder,
        trajectory_buffer=traj_buffer,
        temperature=temperature,
        lr=contrastive_lr,
        device=device,
    )

    contrastive_losses = contrastive_learner.pretrain(
        num_steps=contrastive_steps,
        batch_size=contrastive_batch_size,
        num_negatives=num_negatives,
        log_freq=log_freq,
    )

    # 保存预训练编码器
    torch.save(encoder.state_dict(), os.path.join(save_path, "encoder_pretrained.pt"))
    np.save(os.path.join(log_path, "contrastive_losses.npy"), contrastive_losses)

    # ========== 构建表征分布 ==========
    print("\n[4/5] 构建表征空间分布...")
    distribution = RepresentationDistribution(
        representation_dim=representation_dim, percentile=95.0, device=device
    )

    # 计算所有训练数据的表征
    encoder.eval()
    all_repr = []
    batch_size = 1024

    with torch.no_grad():
        for i in range(0, buffer.size, batch_size):
            end_idx = min(i + batch_size, buffer.size)
            states = torch.FloatTensor(buffer.states[i:end_idx]).to(device)
            actions = torch.FloatTensor(buffer.actions[i:end_idx]).to(device)
            repr_batch = encoder.encode(states, actions)
            all_repr.append(repr_batch)

    all_repr = torch.cat(all_repr, dim=0)
    distribution.fit(all_repr)

    # 保存分布参数
    distribution.save(os.path.join(save_path, "distribution.pt"))

    # ========== 策略优化 ==========
    print("\n[5/5] 自适应保守策略优化...")
    agent = CRLC_SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        encoder=encoder,
        distribution=distribution,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        beta_min=beta_min,
        beta_max=beta_max,
        distance_scale=distance_scale,
        device=device,
    )

    # 训练日志
    train_log = {
        "step": [],
        "critic_loss": [],
        "actor_loss": [],
        "cql_loss": [],
        "penalty_weight": [],
        "eval_return": [],
        "eval_success": [],
    }

    best_return = -float("inf")

    for step in range(1, policy_steps + 1):
        # 训练一步
        info = agent.train_step(buffer, batch_size=policy_batch_size)

        # 记录日志
        if step % log_freq == 0:
            print(
                f"Step {step}/{policy_steps} | "
                f"Critic: {info['critic_loss']:.3f} | "
                f"Actor: {info['actor_loss']:.3f} | "
                f"CQL: {info['cql_loss']:.3f} | "
                f"Penalty: {info['penalty_weight_mean']:.3f} | "
                f"Alpha: {info['alpha']:.3f}"
            )

            train_log["step"].append(step)
            train_log["critic_loss"].append(info["critic_loss"])
            train_log["actor_loss"].append(info["actor_loss"])
            train_log["cql_loss"].append(info["cql_loss"])
            train_log["penalty_weight"].append(info["penalty_weight_mean"])

        # 评估
        if step % eval_freq == 0:
            eval_info = evaluate_policy(env, agent, num_episodes=eval_episodes)
            print(
                f">>> Evaluation | Return: {eval_info['return_mean']:.2f} ± {eval_info['return_std']:.2f} | "
                f"Success: {eval_info['success_rate']:.2%}"
            )

            train_log["eval_return"].append(eval_info["return_mean"])
            train_log["eval_success"].append(eval_info["success_rate"])

            # 保存最佳模型
            if eval_info["return_mean"] > best_return:
                best_return = eval_info["return_mean"]
                agent.save(os.path.join(save_path, "best_model.pt"))
                print(f">>> 保存最佳模型 (Return: {best_return:.2f})")

        # 定期保存
        if step % save_freq == 0:
            agent.save(os.path.join(save_path, f"model_{step}.pt"))

    # 保存最终模型和日志
    agent.save(os.path.join(save_path, "final_model.pt"))

    with open(os.path.join(log_path, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳回报: {best_return:.2f}")
    print(f"模型保存: {save_path}")
    print(f"日志保存: {log_path}")
    print("=" * 60)

    return agent, train_log


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CRLC 训练脚本")

    # 环境
    parser.add_argument(
        "--env", type=str, default="antmaze-medium-diverse-v2", help="D4RL 环境名称"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 编码器
    parser.add_argument("--repr-dim", type=int, default=128, help="表征维度")

    # 对比学习
    parser.add_argument(
        "--contrastive-steps", type=int, default=50000, help="对比学习预训练步数"
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE 温度")

    # 策略优化
    parser.add_argument(
        "--policy-steps", type=int, default=1000000, help="策略优化步数"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="批次大小")

    # 保守惩罚
    parser.add_argument("--beta-min", type=float, default=0.1, help="最小惩罚权重")
    parser.add_argument("--beta-max", type=float, default=5.0, help="最大惩罚权重")

    # 设备
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备"
    )

    # 日志
    parser.add_argument("--log-dir", type=str, default="./logs", help="日志目录")
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints", help="模型保存目录"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_crlc(
        env_name=args.env,
        seed=args.seed,
        representation_dim=args.repr_dim,
        contrastive_steps=args.contrastive_steps,
        temperature=args.temperature,
        policy_steps=args.policy_steps,
        policy_batch_size=args.batch_size,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        device=args.device,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
    )
