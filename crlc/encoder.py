"""
CRLC 编码器网络模块
实现状态编码器、动作编码器和联合编码器
对应论文第3.2.1节：状态-动作联合编码器设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    activation: str = "relu",
) -> nn.Sequential:
    """构建多层感知机"""
    activation_fn = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }.get(activation, nn.ReLU)

    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation_fn())

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_fn())

    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """
    状态编码器 f_s: S -> R^d_s
    将高维状态映射到低维表征空间
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.encoder = build_mlp(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            activation=activation,
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态张量 [batch_size, state_dim]
        Returns:
            状态表征 [batch_size, output_dim]
        """
        return self.encoder(state)


class ActionEncoder(nn.Module):
    """
    动作编码器 f_a: A -> R^d_a
    将动作映射到表征空间
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.output_dim = output_dim

        self.encoder = build_mlp(
            input_dim=action_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            activation=activation,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            action: 动作张量 [batch_size, action_dim]
        Returns:
            动作表征 [batch_size, output_dim]
        """
        return self.encoder(action)


class FusionNetwork(nn.Module):
    """
    融合网络 g: R^d_s × R^d_a -> R^d
    将状态表征和动作表征融合为联合表征
    """

    def __init__(
        self,
        state_repr_dim: int,
        action_repr_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.output_dim = output_dim

        # 拼接后通过MLP融合
        self.fusion = build_mlp(
            input_dim=state_repr_dim + action_repr_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            activation=activation,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, state_repr: torch.Tensor, action_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        Args:
            state_repr: 状态表征 [batch_size, state_repr_dim]
            action_repr: 动作表征 [batch_size, action_repr_dim]
        Returns:
            联合表征 [batch_size, output_dim]
        """
        # 拼接状态和动作表征
        combined = torch.cat([state_repr, action_repr], dim=-1)
        return self.fusion(combined)


class ProjectionHead(nn.Module):
    """
    投影头 h: R^d -> R^d'
    用于对比学习的投影层（仅在训练时使用）
    """

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 64
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            z: 联合表征 [batch_size, input_dim]
        Returns:
            投影表征 [batch_size, output_dim]
        """
        return self.projection(z)


class JointEncoder(nn.Module):
    """
    联合编码器 φ: S × A -> R^d
    完整的状态-动作联合编码器，包含投影头

    对应公式: z_φ(s,a) = g(f_s(s), f_a(a))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_hidden_dim: int = 256,
        action_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        representation_dim: int = 128,
        projection_dim: int = 64,
        activation: str = "relu",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.representation_dim = representation_dim
        self.projection_dim = projection_dim

        # 状态编码器
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            hidden_dim=state_hidden_dim,
            output_dim=state_hidden_dim // 2,  # d_s
            activation=activation,
        )

        # 动作编码器
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=action_hidden_dim,
            output_dim=action_hidden_dim // 2,  # d_a
            activation=activation,
        )

        # 融合网络
        self.fusion = FusionNetwork(
            state_repr_dim=state_hidden_dim // 2,
            action_repr_dim=action_hidden_dim // 2,
            hidden_dim=fusion_hidden_dim,
            output_dim=representation_dim,
            activation=activation,
        )

        # 投影头（仅用于对比学习）
        self.projection_head = ProjectionHead(
            input_dim=representation_dim,
            hidden_dim=representation_dim,
            output_dim=projection_dim,
        )

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        编码状态-动作对为联合表征（不使用投影头）
        这是用于策略优化时的表征

        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
        Returns:
            联合表征 z_φ [batch_size, representation_dim]
        """
        state_repr = self.state_encoder(state)
        action_repr = self.action_encoder(action)
        z = self.fusion(state_repr, action_repr)
        return z

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, use_projection: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            use_projection: 是否使用投影头
        Returns:
            z: 联合表征 [batch_size, representation_dim]
            z_proj: 投影表征 [batch_size, projection_dim] (如果use_projection=True)
        """
        # 编码
        z = self.encode(state, action)

        # 投影（用于对比学习）
        if use_projection:
            z_proj = self.projection_head(z)
            # L2 归一化
            z_proj = F.normalize(z_proj, dim=-1)
            return z, z_proj

        return z, None

    def get_state_representation(self, state: torch.Tensor) -> torch.Tensor:
        """
        获取仅状态的表征（用于某些分析）

        Args:
            state: 状态张量 [batch_size, state_dim]
        Returns:
            状态表征 [batch_size, state_hidden_dim // 2]
        """
        return self.state_encoder(state)


if __name__ == "__main__":
    # 简单测试
    batch_size = 32
    state_dim = 17  # Ant 环境
    action_dim = 6

    encoder = JointEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        representation_dim=128,
        projection_dim=64,
    )

    # 随机输入
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)

    # 测试编码
    z, z_proj = encoder(states, actions, use_projection=True)
    print(f"表征维度: {z.shape}")  # [32, 128]
    print(f"投影维度: {z_proj.shape}")  # [32, 64]

    # 测试仅编码（不使用投影头）
    z_only = encoder.encode(states, actions)
    print(f"仅表征维度: {z_only.shape}")  # [32, 128]

    print("编码器测试通过！")
