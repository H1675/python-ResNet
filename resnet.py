import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 定义残差块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.identity = nn.Identity()  # 用于跳跃连接

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # 跳跃连接
        out = self.relu(out)
        return out

# 2. 定义ResNet模型
class ResNetAntenna(nn.Module):
    def __init__(self, input_dim):
        super(ResNetAntenna, self).__init__()
        self.block1 = ResidualBlock(input_dim, 128)
        self.block2 = ResidualBlock(128, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)  # 输出一个S11幅值
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 生成示例数据
num_features = 10
num_samples = 1000

# 输入：随机天线参数
X = np.random.rand(num_samples, num_features).astype(np.float32)

# 输出：对应的S11幅值（这里模拟数据，真实场景应根据仿真结果生成）
Y = np.random.rand(num_samples, 1).astype(np.float32)

# 4. 创建DataLoader
train_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 5. 初始化模型，损失函数和优化器
model = ResNetAntenna(input_dim=num_features)
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 打印训练过程中的损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 7. 保存模型
torch.save(model.state_dict(), 'resnet_antenna_model.pth')