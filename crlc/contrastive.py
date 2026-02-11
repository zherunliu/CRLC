import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from .encoder import JointEncoder


class InfoNCELoss(nn.Module):
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
        self.trajectory_starts = np.array(self.trajectory_starts)
        self.trajectory_ends = np.array(self.trajectory_ends)
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
            anchor_states, anchor_actions, positive_states, positive_actions,
            negative_states, negative_actions
        """
        # 随机采样锚点索引
        anchor_indices = np.random.randint(0, self.size, batch_size)

        # 获取轨迹信息（向量化）
        traj_ids = self.step_to_trajectory[anchor_indices]
        traj_starts = self.trajectory_starts[traj_ids]
        traj_ends = self.trajectory_ends[traj_ids]

        # 计算正样本窗口（向量化）
        window_starts = np.maximum(traj_starts, anchor_indices - self.trajectory_window)
        window_ends = np.minimum(traj_ends, anchor_indices + self.trajectory_window)

        # 采样正样本
        positive_indices = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            valid_range = list(range(window_starts[i], anchor_indices[i])) + list(
                range(anchor_indices[i] + 1, window_ends[i] + 1)
            )
            if len(valid_range) == 0:
                positive_indices[i] = anchor_indices[i]
            else:
                positive_indices[i] = valid_range[np.random.randint(len(valid_range))]

        # 采样负样本（快速向量化）
        # 多采样一些，然后过滤掉同轨迹的
        negative_indices = np.random.randint(
            0, self.size, (batch_size, num_negatives * 2)
        )
        neg_traj_ids = self.step_to_trajectory[negative_indices]
        valid_mask = neg_traj_ids != traj_ids[:, None]

        # 取每行前 num_negatives 个有效负样本
        final_neg_indices = np.zeros((batch_size, num_negatives), dtype=np.int64)
        for i in range(batch_size):
            valid_negs = negative_indices[i][valid_mask[i]]
            if len(valid_negs) >= num_negatives:
                final_neg_indices[i] = valid_negs[:num_negatives]
            else:
                # 不够的话用有效的填充
                final_neg_indices[i, : len(valid_negs)] = valid_negs
                if len(valid_negs) > 0:
                    final_neg_indices[i, len(valid_negs) :] = np.random.choice(
                        valid_negs, num_negatives - len(valid_negs), replace=True
                    )
                else:
                    final_neg_indices[i] = np.random.randint(
                        0, self.size, num_negatives
                    )

        # 提取数据
        anchor_states = self.states[anchor_indices]
        anchor_actions = self.actions[anchor_indices]
        positive_states = self.states[positive_indices]
        positive_actions = self.actions[positive_indices]
        negative_states = self.states[final_neg_indices]
        negative_actions = self.actions[final_neg_indices]

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
