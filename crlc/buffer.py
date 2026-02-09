import numpy as np
import torch
from typing import Tuple


class OfflineReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 2000000,
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = device

        # 预分配内存
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        batch_size = len(states)

        if self.ptr + batch_size <= self.max_size:
            self.states[self.ptr : self.ptr + batch_size] = states
            self.actions[self.ptr : self.ptr + batch_size] = actions
            self.rewards[self.ptr : self.ptr + batch_size] = rewards
            self.next_states[self.ptr : self.ptr + batch_size] = next_states
            self.dones[self.ptr : self.ptr + batch_size] = dones
        else:
            # 处理环绕情况
            remaining = self.max_size - self.ptr
            self.states[self.ptr :] = states[:remaining]
            self.actions[self.ptr :] = actions[:remaining]
            self.rewards[self.ptr :] = rewards[:remaining]
            self.next_states[self.ptr :] = next_states[:remaining]
            self.dones[self.ptr :] = dones[:remaining]

            overflow = batch_size - remaining
            self.states[:overflow] = states[remaining:]
            self.actions[:overflow] = actions[remaining:]
            self.rewards[:overflow] = rewards[remaining:]
            self.next_states[:overflow] = next_states[remaining:]
            self.dones[:overflow] = dones[remaining:]

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1]
        """
        indices = np.random.randint(0, self.size, batch_size)

        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).reshape(-1, 1).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).reshape(-1, 1).to(self.device),
        )

    def get_all_data(self) -> Tuple[np.ndarray, ...]:
        return (
            self.states[: self.size].copy(),
            self.actions[: self.size].copy(),
            self.rewards[: self.size].copy(),
            self.next_states[: self.size].copy(),
            self.dones[: self.size].copy(),
        )

    @classmethod
    def from_d4rl(cls, env_name: str, device: str = "cuda") -> "OfflineReplayBuffer":
        import gym
        import d4rl

        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)

        state_dim = dataset["observations"].shape[1]
        action_dim = dataset["actions"].shape[1]

        buffer = cls(
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

        print(f"从 D4RL 加载了 {buffer.size} 个样本")
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

        return buffer


if __name__ == "__main__":
    # 简单测试（without D4RL）
    buffer = OfflineReplayBuffer(
        state_dim=10, action_dim=4, max_size=1000, device="cpu"
    )

    # 添加测试数据
    for _ in range(100):
        buffer.add(
            state=np.random.randn(10),
            action=np.random.randn(4),
            reward=np.random.randn(),
            next_state=np.random.randn(10),
            done=False,
        )

    # 采样测试
    batch = buffer.sample(32)
    print(f"采样批次大小: {[b.shape for b in batch]}")

    print("缓冲区测试通过！")
