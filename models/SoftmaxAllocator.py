import torch
import torch.nn as nn


class SoftmaxAllocator(nn.Module):
    def __init__(self, num_assets, input_dim, hidden_layers=[], dropout_rate=0.0):
        """
        Args:
            num_assets: 资产数量 (8)
            input_dim: 每个资产的特征数量 (7)
            hidden_layers: list, 例如 [64] 或 [64, 32]。如果是 [] 则为纯线性.
            dropout_rate: 防止过拟合，通常 0.2 ~ 0.5
        """
        super().__init__()

        layers = []
        # 输入维度 = 资产数量 * 特征数 (Flatten之后)
        current_dim = num_assets * input_dim

        # ============================================================
        # 【关键修复】第一层必须加 BatchNorm
        # 金融特征(如收益率)数值很小，不加这个 DNN 根本训练不动
        # ============================================================
        layers.append(nn.BatchNorm1d(current_dim))

        # 构建隐藏层
        if hidden_layers:
            for h_dim in hidden_layers:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())  # 激活函数
                layers.append(nn.Dropout(p=dropout_rate))  # Dropout
                current_dim = h_dim

        # 输出层 (Output Layer)
        layers.append(nn.Linear(current_dim, num_assets))
        layers.append(nn.Softmax(dim=1))  # 保证 sum=1, >0

        self.net = nn.Sequential(*layers)
        self.num_assets = num_assets
        self.input_dim = input_dim

    def forward(self, x):
        """
        x shape: (batch_size, num_assets, input_dim)
        """
        batch_size = x.size(0)

        # Flatten: (Batch, 8, 7) -> (Batch, 56)
        x_flat = x.reshape(batch_size, -1)

        weights = self.net(x_flat)

        # ============================================================
        # 【数值保护】防止极其罕见的 NaN 或纯 0 导致后续 Log 报错
        # ============================================================
        weights = torch.nan_to_num(weights, nan=1.0 / self.num_assets)
        weights = torch.clamp(weights, min=1e-8, max=1.0)  # 加上极小值防止 log(0)

        return weights
