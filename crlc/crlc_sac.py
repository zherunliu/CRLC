"""
CRLC-SAC 算法实现
基于表征距离的自适应保守策略优化
对应论文第3.3节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import copy

from .encoder import JointEncoder
from .distribution import RepresentationDistribution
from .buffer import OfflineReplayBuffer


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
    """构建 MLP 网络"""
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianPolicy(nn.Module):
    """
    Squashed Gaussian 策略网络
    输出经过 tanh 压缩的动作
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        self.trunk = build_mlp(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: 状态 [batch, state_dim]
            deterministic: 是否使用确定性动作

        Returns:
            action: 动作 [batch, action_dim]
            log_prob: 对数概率 [batch, 1]
        """
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(state.shape[0], 1, device=state.device)
        else:
            # 重参数化采样
            noise = torch.randn_like(mean)
            raw_action = mean + std * noise
            action = torch.tanh(raw_action)

            # 计算对数概率（包括 tanh 的雅可比修正）
            log_prob = -0.5 * (
                ((raw_action - mean) / (std + 1e-8)) ** 2
                + 2 * log_std
                + np.log(2 * np.pi)
            ).sum(dim=-1, keepdim=True)

            # tanh 变换的修正项
            log_prob -= torch.log(1 - action**2 + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob

    def sample_multiple(
        self, state: torch.Tensor, num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样多个动作（用于 CQL 中的 Q 值估计）

        Args:
            state: 状态 [batch, state_dim]
            num_samples: 采样数量

        Returns:
            actions: [batch, num_samples, action_dim]
            log_probs: [batch, num_samples, 1]
        """
        batch_size = state.shape[0]

        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # 扩展维度
        mean = mean.unsqueeze(1).expand(-1, num_samples, -1)
        std = std.unsqueeze(1).expand(-1, num_samples, -1)

        noise = torch.randn_like(mean)
        raw_actions = mean + std * noise
        actions = torch.tanh(raw_actions)

        log_probs = -0.5 * (
            ((raw_actions - mean) / (std + 1e-8)) ** 2
            + 2 * log_std.unsqueeze(1)
            + np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)
        log_probs -= torch.log(1 - actions**2 + 1e-6).sum(dim=-1, keepdim=True)

        return actions, log_probs


class TwinQNetwork(nn.Module):
    """
    双 Q 网络
    使用两个独立的 Q 网络减少过估计
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        self.q1 = build_mlp(
            input_dim=state_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
        )

        self.q2 = build_mlp(
            input_dim=state_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算两个 Q 值

        Returns:
            q1: [batch, 1]
            q2: [batch, 1]
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只计算 Q1"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class CRLC_SAC:
    """
    CRLC-SAC 算法
    基于对比表征学习的保守离线策略优化

    核心创新：
    1. 使用预训练的联合编码器计算表征距离
    2. 根据表征距离自适应调整保守惩罚权重
    3. 对 OOD 动作施加更强的 Q 值惩罚
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        encoder: JointEncoder,
        distribution: RepresentationDistribution,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_alpha: bool = True,
        init_alpha: float = 0.2,
        beta_min: float = 0.1,
        beta_max: float = 5.0,
        distance_scale: float = 1.0,
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.device = device

        # 保守惩罚参数
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.distance_scale = distance_scale

        # 编码器和分布（预训练，冻结参数）
        self.encoder = encoder.to(device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.distribution = distribution

        # 策略网络
        self.actor = SquashedGaussianPolicy(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(device)

        # Q 网络
        self.critic = TwinQNetwork(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(device)

        # 目标 Q 网络
        self.critic_target = copy.deepcopy(self.critic)

        # 温度参数 alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = init_alpha

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 训练统计
        self.train_step_count = 0

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            action, _ = self.actor(state, deterministic=deterministic)
            return action.cpu().numpy().flatten()

    def _compute_adaptive_penalty(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        计算自适应惩罚权重

        对应公式:
        β(s,a) = β_min + (β_max - β_min) · σ(scale · (d_M(z) / τ - 1))
        """
        with torch.no_grad():
            # 获取表征
            z = self.encoder.encode(states, actions)

            # 计算惩罚权重
            weights = self.distribution.compute_penalty_weight(
                z,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                scale=self.distance_scale,
            )

        return weights.unsqueeze(-1)  # [batch, 1]

    def train_step(
        self,
        buffer: OfflineReplayBuffer,
        batch_size: int = 256,
        num_policy_samples: int = 10,
    ) -> Dict[str, float]:
        """
        执行一步训练

        Args:
            buffer: 离线数据缓冲区
            batch_size: 批次大小
            num_policy_samples: 用于 CQL 损失的策略采样数量

        Returns:
            info: 训练信息字典
        """
        self.train_step_count += 1

        # 采样数据
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # ========== 更新 Critic ==========
        with torch.no_grad():
            # 采样下一状态的动作
            next_actions, next_log_probs = self.actor(next_states)

            # 计算目标 Q 值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # 当前 Q 值
        current_q1, current_q2 = self.critic(states, actions)

        # TD 损失
        td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # ========== 自适应保守惩罚 ==========
        # 采样随机动作
        random_actions = (
            torch.FloatTensor(batch_size, self.action_dim)
            .uniform_(-1, 1)
            .to(self.device)
        )

        # 采样策略动作
        with torch.no_grad():
            policy_actions, policy_log_probs = self.actor.sample_multiple(
                states, num_policy_samples
            )

        # 计算不同动作的 Q 值
        # 随机动作 Q 值
        random_q1, random_q2 = self.critic(states, random_actions)

        # 策略动作 Q 值 (需要重塑维度)
        states_expanded = states.unsqueeze(1).expand(-1, num_policy_samples, -1)
        states_flat = states_expanded.reshape(-1, self.state_dim)
        policy_actions_flat = policy_actions.reshape(-1, self.action_dim)

        policy_q1, policy_q2 = self.critic(states_flat, policy_actions_flat)
        policy_q1 = policy_q1.reshape(batch_size, num_policy_samples)
        policy_q2 = policy_q2.reshape(batch_size, num_policy_samples)

        # 计算自适应惩罚权重
        # 对于随机动作
        random_penalty_weights = self._compute_adaptive_penalty(states, random_actions)

        # 对于策略动作 (使用平均权重)
        policy_penalty_weights_list = []
        for i in range(num_policy_samples):
            w = self._compute_adaptive_penalty(states, policy_actions[:, i])
            policy_penalty_weights_list.append(w)
        policy_penalty_weights = torch.stack(
            policy_penalty_weights_list, dim=1
        ).squeeze(-1)  # [batch, num_samples]

        # CQL 损失：对 OOD 动作施加惩罚
        # logsumexp 计算 soft maximum
        cat_q1 = torch.cat(
            [
                random_q1 * random_penalty_weights,  # 随机动作
                policy_q1 * policy_penalty_weights,  # 策略动作
            ],
            dim=-1,
        )
        cat_q2 = torch.cat(
            [
                random_q2 * random_penalty_weights,
                policy_q2 * policy_penalty_weights,
            ],
            dim=-1,
        )

        # 保守损失
        cql_loss_q1 = torch.logsumexp(cat_q1, dim=-1).mean() - current_q1.mean()
        cql_loss_q2 = torch.logsumexp(cat_q2, dim=-1).mean() - current_q2.mean()
        cql_loss = cql_loss_q1 + cql_loss_q2

        # 总 Critic 损失
        critic_loss = td_loss + 0.5 * cql_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== 更新 Actor ==========
        actions_new, log_probs_new = self.actor(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs_new - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== 更新 Alpha ==========
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs_new + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)

        # ========== 软更新目标网络 ==========
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # 返回训练信息
        return {
            "critic_loss": critic_loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q_mean": current_q1.mean().item(),
            "penalty_weight_mean": random_penalty_weights.mean().item(),
        }

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha if self.auto_alpha else None,
                "train_step_count": self.train_step_count,
            },
            path,
        )
        print(f"模型已保存到 {path}")

    def load(self, path: str):
        """加载模型"""
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        if self.auto_alpha and data["log_alpha"] is not None:
            self.log_alpha.data = data["log_alpha"].data
            self.alpha = self.log_alpha.exp().item()
        self.train_step_count = data["train_step_count"]
        print(f"模型已从 {path} 加载")


if __name__ == "__main__":
    # 简单测试（不需要真实环境）
    state_dim = 29  # Antmaze
    action_dim = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建编码器
    encoder = JointEncoder(
        state_dim=state_dim, action_dim=action_dim, representation_dim=128
    ).to(device)

    # 创建模拟的分布模型
    distribution = RepresentationDistribution(representation_dim=128, device=device)

    # 使用随机表征初始化分布
    with torch.no_grad():
        fake_states = torch.randn(1000, state_dim).to(device)
        fake_actions = torch.randn(1000, action_dim).to(device)
        fake_repr = encoder.encode(fake_states, fake_actions)
    distribution.fit(fake_repr)

    # 创建 CRLC-SAC
    agent = CRLC_SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        encoder=encoder,
        distribution=distribution,
        device=device,
    )

    # 创建模拟缓冲区
    buffer = OfflineReplayBuffer(
        state_dim=state_dim, action_dim=action_dim, max_size=1000, device=device
    )

    # 添加模拟数据
    for _ in range(500):
        buffer.add(
            state=np.random.randn(state_dim),
            action=np.random.randn(action_dim),
            reward=np.random.randn(),
            next_state=np.random.randn(state_dim),
            done=np.random.rand() < 0.01,
        )

    # 执行几步训练
    for i in range(5):
        info = agent.train_step(buffer, batch_size=64)
        print(
            f"训练步 {i + 1}: critic_loss={info['critic_loss']:.4f}, "
            f"actor_loss={info['actor_loss']:.4f}, "
            f"penalty_weight={info['penalty_weight_mean']:.4f}"
        )

    # 测试动作选择
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print(f"动作维度: {action.shape}")

    print("CRLC-SAC 测试通过！")
