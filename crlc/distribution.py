import torch
import numpy as np


class RepresentationDistribution:
    def __init__(
        self,
        representation_dim: int,
        percentile: float = 95.0,
        regularization: float = 1e-6,
        device: str = "cuda",
    ):
        """
        初始化分布模型

        Args:
            representation_dim: 表征维度
            percentile: 用于确定距离阈值的百分位数
            regularization: 协方差矩阵的正则化项
            device: 计算设备
        """
        self.representation_dim = representation_dim
        self.percentile = percentile
        self.regularization = regularization
        self.device = device

        # 分布参数
        self.mean = None  # [dim]
        self.covariance = None  # [dim, dim]
        self.precision = None  # 协方差逆矩阵 [dim, dim]
        self.distance_threshold = None  # 距离阈值

        # 缓存的训练集距离
        self.train_distances = None

    def fit(self, representations: torch.Tensor, batch_size: int = 1024):
        """
        使用训练数据拟合高斯分布

        Args:
            representations: 训练集表征 [N, dim]
            batch_size: 批处理大小
        """
        representations = representations.to(self.device)
        N = representations.shape[0]

        print(f"拟合分布，共 {N} 个样本...")

        # 计算均值
        self.mean = representations.mean(dim=0)  # [dim]

        # 计算协方差矩阵（分批处理以节省内存）
        centered = representations - self.mean.unsqueeze(0)  # [N, dim]

        # 使用增量更新计算协方差
        cov = torch.zeros(
            self.representation_dim, self.representation_dim, device=self.device
        )

        for i in range(0, N, batch_size):
            batch = centered[i : min(i + batch_size, N)]
            cov += batch.T @ batch

        self.covariance = cov / N

        # 添加正则化项以保证数值稳定性
        self.covariance += self.regularization * torch.eye(
            self.representation_dim, device=self.device
        )

        # 计算精度矩阵（协方差逆矩阵）
        self.precision = torch.linalg.inv(self.covariance)

        # 计算训练集上的马氏距离并确定阈值
        self._compute_threshold(representations, batch_size)

        print(f"分布拟合完成，距离阈值: {self.distance_threshold:.4f}")

    def _compute_threshold(self, representations: torch.Tensor, batch_size: int = 1024):
        """计算距离阈值"""
        distances = []

        for i in range(0, len(representations), batch_size):
            batch = representations[i : min(i + batch_size, len(representations))]
            batch_distances = self.mahalanobis_distance(batch)
            distances.append(batch_distances.cpu().numpy())

        self.train_distances = np.concatenate(distances)
        self.distance_threshold = np.percentile(self.train_distances, self.percentile)

    def mahalanobis_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算马氏距离

        Args:
            z: 表征向量 [batch_size, dim]

        Returns:
            distances: 马氏距离 [batch_size]
        """
        if self.mean is None or self.precision is None:
            raise RuntimeError("请先调用 fit() 拟合分布")

        z = z.to(self.device)

        # 中心化
        centered = z - self.mean.unsqueeze(0)  # [batch, dim]

        # 计算马氏距离: sqrt(x^T Σ^{-1} x)
        # (batch, dim) @ (dim, dim) -> (batch, dim)
        temp = centered @ self.precision
        # (batch, dim) * (batch, dim) -> sum -> (batch,)
        squared_distance = (temp * centered).sum(dim=-1)

        # 取平方根，添加小的 epsilon 防止数值问题
        distance = torch.sqrt(squared_distance + 1e-8)

        return distance

    def normalized_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算归一化的马氏距离
        使用阈值进行归一化，使得训练集内 95% 样本的距离 < 1

        Args:
            z: 表征向量 [batch_size, dim]

        Returns:
            normalized_distances: 归一化距离 [batch_size]
        """
        distance = self.mahalanobis_distance(z)
        return distance / self.distance_threshold

    def compute_penalty_weight(
        self,
        z: torch.Tensor,
        beta_min: float = 0.1,
        beta_max: float = 5.0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        计算自适应惩罚权重

        对应公式:
        β(s,a) = β_min + (β_max - β_min) · σ(scale · (d_M(z) / τ - 1))

        Args:
            z: 表征向量 [batch_size, dim]
            beta_min: 最小惩罚权重
            beta_max: 最大惩罚权重
            scale: 距离缩放因子

        Returns:
            weights: 惩罚权重 [batch_size]
        """
        # 计算归一化距离
        norm_dist = self.normalized_distance(z)

        # 使用 sigmoid 函数平滑过渡
        # 当 norm_dist = 1 (阈值处) 时，sigmoid(0) = 0.5
        # 当 norm_dist > 1 (分布外) 时，sigmoid > 0.5，惩罚增大
        # 当 norm_dist < 1 (分布内) 时，sigmoid < 0.5，惩罚减小
        sigmoid_input = scale * (norm_dist - 1.0)
        weight = beta_min + (beta_max - beta_min) * torch.sigmoid(sigmoid_input)

        return weight

    def is_ood(self, z: torch.Tensor) -> torch.Tensor:
        """
        判断样本是否为分布外 (OOD)

        Args:
            z: 表征向量 [batch_size, dim]

        Returns:
            is_ood: 布尔张量 [batch_size]
        """
        distance = self.mahalanobis_distance(z)
        return distance > self.distance_threshold

    def ood_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算 OOD 分数（用于 ROC 分析）

        Args:
            z: 表征向量 [batch_size, dim]

        Returns:
            scores: OOD 分数 [batch_size]，越大越可能是 OOD
        """
        return self.mahalanobis_distance(z)

    def save(self, path: str):
        """保存分布参数"""
        torch.save(
            {
                "mean": self.mean,
                "covariance": self.covariance,
                "precision": self.precision,
                "distance_threshold": self.distance_threshold,
                "train_distances": self.train_distances,
                "representation_dim": self.representation_dim,
                "percentile": self.percentile,
                "regularization": self.regularization,
            },
            path,
        )
        print(f"分布参数已保存到 {path}")

    def load(self, path: str):
        """加载分布参数"""
        data = torch.load(path, map_location=self.device)
        self.mean = data["mean"]
        self.covariance = data["covariance"]
        self.precision = data["precision"]
        self.distance_threshold = data["distance_threshold"]
        self.train_distances = data["train_distances"]
        self.representation_dim = data["representation_dim"]
        self.percentile = data["percentile"]
        self.regularization = data["regularization"]
        print(f"分布参数已从 {path} 加载")


if __name__ == "__main__":
    # 简单测试
    dim = 128
    N_train = 1000
    N_test = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟训练数据（服从某个分布）
    train_repr = torch.randn(N_train, dim).to(device)

    # 创建分布模型
    dist_model = RepresentationDistribution(
        representation_dim=dim, percentile=95.0, device=device
    )

    # 拟合分布
    dist_model.fit(train_repr)

    # 测试分布内样本
    test_in = torch.randn(N_test, dim).to(device) * 0.5  # 较小方差
    distances_in = dist_model.mahalanobis_distance(test_in)
    print(f"分布内样本平均距离: {distances_in.mean().item():.4f}")

    # 测试分布外样本
    test_out = torch.randn(N_test, dim).to(device) * 3.0 + 5.0  # 较大方差和偏移
    distances_out = dist_model.mahalanobis_distance(test_out)
    print(f"分布外样本平均距离: {distances_out.mean().item():.4f}")

    # 测试惩罚权重
    weights_in = dist_model.compute_penalty_weight(test_in)
    weights_out = dist_model.compute_penalty_weight(test_out)
    print(f"分布内样本平均惩罚权重: {weights_in.mean().item():.4f}")
    print(f"分布外样本平均惩罚权重: {weights_out.mean().item():.4f}")

    # 测试 OOD 检测
    ood_in = dist_model.is_ood(test_in)
    ood_out = dist_model.is_ood(test_out)
    print(f"分布内样本 OOD 比例: {ood_in.float().mean().item():.2%}")
    print(f"分布外样本 OOD 比例: {ood_out.float().mean().item():.2%}")

    print("分布建模模块测试通过！")
