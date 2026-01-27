import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 定义残差块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.identity = nn.Linear(in_channels, out_channels)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        residual = self.identity(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # 跳跃连接
        out = self.relu(out)
        return out

# 定义ResNet模型
class ResNetAntenna(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(ResNetAntenna, self).__init__()
        self.block1 = ResidualBlock(input_dim, 128)
        self.block2 = ResidualBlock(128, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

