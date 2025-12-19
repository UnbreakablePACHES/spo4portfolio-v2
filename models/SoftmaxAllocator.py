import torch
import torch.nn as nn

class SoftmaxAllocator(nn.Module):
    def __init__(self, num_assets, input_dim, hidden_dim=None):
        super().__init__()
        
        # 如果你想做 DNN，可以在这里加 hidden layers
        # 这里演示的是 LR (Linear) + Softmax
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(num_assets * input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_assets),
                nn.Softmax(dim=1) # 保证输出是权重 (sum=1, >0)
            )
        else:
            # 纯 LR + Softmax
            self.net = nn.Sequential(
                nn.Linear(num_assets * input_dim, num_assets),
                nn.Softmax(dim=1)
            )
            
        self.num_assets = num_assets
        self.input_dim = input_dim

    def forward(self, x):
        """
        x shape: (batch_size, num_assets, input_dim)
        output: (batch_size, num_assets) -> Weights
        """
        batch_size = x.size(0)
        # Flatten input
        x_flat = x.reshape(batch_size, -1)
        weights = self.net(x_flat)
        return weights