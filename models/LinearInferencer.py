import torch.nn as nn

class LinearInferencer(nn.Module):
    def __init__(self, num_assets, input_dim):
        super().__init__()
        self.num_assets = num_assets
        self.input_dim = input_dim
        
        # 输入维度 = 所有资产特征 flatten = num_assets × input_dim
        self.linear = nn.Linear(num_assets * input_dim, num_assets)

    def forward(self, x):
        """
        x: shape = (batch_size, num_assets, input_dim)
        output: (batch_size, num_assets)
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_assets * self.input_dim)
        return self.linear(x)

