"""
CRLC 对比学习模块
实现 InfoNCE 损失和轨迹感知的对比学习
对应论文第3.2.2节：基于轨迹感知的对比损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
# from encoder import JointEncoder # for mini test
from .encoder import JointEncoder


class InfoNCELoss(nn.Module):
    """
    InfoNCE 损失函数

    对应公式:
    L_contrast = -E[log(exp(sim(z_i, z_i^+)/τ) / Σ_j exp(sim(z_i, z_j)/τ))]

    其中 sim(z_i, z_j) = z_i^T z_j (使用L2归一化后的余弦相似度)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失

        Args:
            anchor: 锚点表征 [batch_size, dim]
            positive: 正样本表征 [batch_size, dim]
            negatives: 负样本表征 [batch_size, num_negatives, dim]

        Returns:
            loss: 标量损失值
        """
        batch_size = anchor.shape[0]

        # L2 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # 计算正样本相似度 [batch_size, 1]
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature

        # 计算负样本相似度 [batch_size, num_negatives]
        neg_sim = (
            torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
        )

        # 拼接所有相似度 [batch_size, 1 + num_negatives]
        logits = torch.cat([pos_sim, neg_sim], dim=-1)

        # 标签为0（正样本在第一个位置）
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss


class TrajectoryBuffer:
    """
    轨迹缓冲区，用于管理离线数据集中的轨迹信息
    支持轨迹感知的正负样本采样
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        trajectory_window: int = 5,
    ):
        """
        初始化轨迹缓冲区

        Args:
            states: 状态数组 [N, state_dim]
            actions: 动作数组 [N, action_dim]
            rewards: 奖励数组 [N]
            next_states: 下一状态数组 [N, state_dim]
            dones: 终止标记数组 [N]
            trajectory_window: 正样本采样的轨迹窗口大小
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.size = len(states)
        self.trajectory_window = trajectory_window

        # 构建轨迹索引
        self._build_trajectory_indices()

    def _build_trajectory_indices(self):
        """构建轨迹起止索引"""
        self.trajectory_starts = [0]
        self.trajectory_ends = []
        self.step_to_trajectory = np.zeros(self.size, dtype=np.int64)

        current_traj = 0
        for i in range(self.size):
            self.step_to_trajectory[i] = current_traj
            if self.dones[i] or i == self.size - 1:
                self.trajectory_ends.append(i)
                if i < self.size - 1:
                    self.trajectory_starts.append(i + 1)
                    current_traj += 1

        self.num_trajectories = len(self.trajectory_ends)
        print(f"构建了 {self.num_trajectories} 条轨迹")

    def sample_batch(
        self, batch_size: int, num_negatives: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        采样一个批次，包含锚点、正样本和负样本

        正样本：同一轨迹中距离在 trajectory_window 内的样本
        负样本：其他轨迹中的随机样本

        Args:
            batch_size: 批次大小
            num_negatives: 每个锚点的负样本数量

        Returns:
            anchor_states: [batch_size, state_dim]
            anchor_actions: [batch_size, action_dim]
            positive_states: [batch_size, state_dim]
            positive_actions: [batch_size, action_dim]
            negative_states: [batch_size, num_negatives, state_dim]
            negative_actions: [batch_size, num_negatives, action_dim]
        """
        # 随机采样锚点索引
        anchor_indices = np.random.randint(0, self.size, batch_size)

        # 初始化数组
        state_dim = self.states.shape[1]
        action_dim = self.actions.shape[1]

        positive_indices = np.zeros(batch_size, dtype=np.int64)
        negative_indices = np.zeros((batch_size, num_negatives), dtype=np.int64)

        for i, anchor_idx in enumerate(anchor_indices):
            # 获取锚点所在轨迹
            traj_id = self.step_to_trajectory[anchor_idx]
            traj_start = self.trajectory_starts[traj_id]
            traj_end = self.trajectory_ends[traj_id]

            # 采样正样本（同一轨迹内，窗口范围内）
            window_start = max(traj_start, anchor_idx - self.trajectory_window)
            window_end = min(traj_end, anchor_idx + self.trajectory_window)

            # 排除锚点自身
            valid_positive = list(range(window_start, anchor_idx)) + list(
                range(anchor_idx + 1, window_end + 1)
            )
            if len(valid_positive) == 0:
                valid_positive = [anchor_idx]  # 如果没有有效正样本，使用自身

            positive_indices[i] = np.random.choice(valid_positive)

            # 采样负样本（其他轨迹）
            # 获取当前轨迹外的所有索引
            current_traj_indices = set(range(traj_start, traj_end + 1))
            all_indices = set(range(self.size))
            other_indices = list(all_indices - current_traj_indices)

            if len(other_indices) >= num_negatives:
                negative_indices[i] = np.random.choice(
                    other_indices, num_negatives, replace=False
                )
            else:
                # 如果其他轨迹样本不够，允许重复采样
                negative_indices[i] = np.random.choice(
                    other_indices, num_negatives, replace=True
                )

        # 提取数据
        anchor_states = self.states[anchor_indices]
        anchor_actions = self.actions[anchor_indices]
        positive_states = self.states[positive_indices]
        positive_actions = self.actions[positive_indices]
        negative_states = self.states[negative_indices]  # [batch, num_neg, state_dim]
        negative_actions = self.actions[
            negative_indices
        ]  # [batch, num_neg, action_dim]

        return (
            anchor_states,
            anchor_actions,
            positive_states,
            positive_actions,
            negative_states,
            negative_actions,
        )


class ContrastiveLearner:
    """
    对比学习训练器
    负责预训练编码器
    """

    def __init__(
        self,
        encoder: JointEncoder,
        trajectory_buffer: TrajectoryBuffer,
        temperature: float = 0.1,
        lr: float = 3e-4,
        device: str = "cuda",
    ):
        self.encoder = encoder.to(device)
        self.trajectory_buffer = trajectory_buffer
        self.device = device

        self.loss_fn = InfoNCELoss(temperature=temperature)
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    def train_step(self, batch_size: int = 256, num_negatives: int = 256) -> float:
        """
        执行一步对比学习训练

        Returns:
            loss: 当前步的损失值
        """
        # 采样批次
        (anchor_s, anchor_a, pos_s, pos_a, neg_s, neg_a) = (
            self.trajectory_buffer.sample_batch(batch_size, num_negatives)
        )

        # 转换为张量
        anchor_s = torch.FloatTensor(anchor_s).to(self.device)
        anchor_a = torch.FloatTensor(anchor_a).to(self.device)
        pos_s = torch.FloatTensor(pos_s).to(self.device)
        pos_a = torch.FloatTensor(pos_a).to(self.device)
        neg_s = torch.FloatTensor(neg_s).to(self.device)
        neg_a = torch.FloatTensor(neg_a).to(self.device)

        # 编码
        _, anchor_proj = self.encoder(anchor_s, anchor_a, use_projection=True)
        _, pos_proj = self.encoder(pos_s, pos_a, use_projection=True)

        # 编码负样本 [batch, num_neg, dim]
        batch_size = neg_s.shape[0]
        num_neg = neg_s.shape[1]
        neg_s_flat = neg_s.view(-1, neg_s.shape[-1])
        neg_a_flat = neg_a.view(-1, neg_a.shape[-1])
        _, neg_proj_flat = self.encoder(neg_s_flat, neg_a_flat, use_projection=True)
        neg_proj = neg_proj_flat.view(batch_size, num_neg, -1)

        # 计算损失
        loss = self.loss_fn(anchor_proj, pos_proj, neg_proj)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def pretrain(
        self,
        num_steps: int = 50000,
        batch_size: int = 256,
        num_negatives: int = 256,
        log_freq: int = 1000,
    ) -> List[float]:
        """
        预训练编码器

        Args:
            num_steps: 总训练步数
            batch_size: 批次大小
            num_negatives: 负样本数量
            log_freq: 日志频率

        Returns:
            losses: 训练过程中的损失列表
        """
        losses = []

        print(f"开始对比学习预训练，共 {num_steps} 步...")

        for step in range(num_steps):
            loss = self.train_step(batch_size, num_negatives)
            losses.append(loss)

            if (step + 1) % log_freq == 0:
                avg_loss = np.mean(losses[-log_freq:])
                print(f"Step {step + 1}/{num_steps}, Loss: {avg_loss:.4f}")

        print("预训练完成！")
        return losses


if __name__ == "__main__":
    # 简单测试
    batch_size = 32
    state_dim = 17
    action_dim = 6
    num_negatives = 64

    # 创建模拟数据
    num_samples = 1000
    states = np.random.randn(num_samples, state_dim).astype(np.float32)
    actions = np.random.randn(num_samples, action_dim).astype(np.float32)
    rewards = np.random.randn(num_samples).astype(np.float32)
    next_states = np.random.randn(num_samples, state_dim).astype(np.float32)
    dones = np.zeros(num_samples, dtype=np.float32)
    # 模拟5条轨迹
    dones[199] = 1.0
    dones[399] = 1.0
    dones[599] = 1.0
    dones[799] = 1.0
    dones[999] = 1.0

    # 创建轨迹缓冲区
    traj_buffer = TrajectoryBuffer(
        states, actions, rewards, next_states, dones, trajectory_window=5
    )

    # 创建编码器
    encoder = JointEncoder(state_dim=state_dim, action_dim=action_dim)

    # 测试 InfoNCE 损失
    loss_fn = InfoNCELoss(temperature=0.1)

    anchor = torch.randn(batch_size, 64)
    positive = torch.randn(batch_size, 64)
    negatives = torch.randn(batch_size, num_negatives, 64)

    loss = loss_fn(anchor, positive, negatives)
    print(f"InfoNCE 损失: {loss.item():.4f}")

    # 测试对比学习器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learner = ContrastiveLearner(
        encoder=encoder, trajectory_buffer=traj_buffer, device=device
    )

    # 执行几步训练
    for i in range(5):
        loss = learner.train_step(batch_size=32, num_negatives=32)
        print(f"训练步 {i + 1}, 损失: {loss:.4f}")

    print("对比学习模块测试通过！")
