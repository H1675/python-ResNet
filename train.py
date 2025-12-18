import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset
from resnet import ResNetAntenna
import torch.nn as nn
import torch.optim as optim

# 变量数量
para_dim = 6

# 获取当前路径
path = os.getcwd()

# 读取sample数据
data_path = path + "\\data" + "\\sample.mat"
data = scio.loadmat(data_path)
sample = data['sample']
sample = sample.astype('float32')

# 读取S11数据
data_path = path + "\\data" + "\\target_S11.mat"
data = scio.loadmat(data_path)
target_S11 = data['target_S11']
target_S11 = target_S11.astype('float32')
target_S11 = target_S11.reshape(-1,1)


# 读取方向图数据
# data_path = path + "\\data" + "\\Dir.mat"
# data = scio.loadmat(data_path)
# Dir = data['Dir']

# 训练集
sample_train = sample[0:139,:]
target_S11_train = target_S11[0:139,:]

# 测试集
sample_test = sample[140:,:]

# 创建DataLoader
train_data = TensorDataset(torch.from_numpy(sample_train), torch.from_numpy(target_S11_train))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 初始化模型，损失函数和优化器
model = ResNetAntenna(input_dim=para_dim)
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10 # 训练轮数
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
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    # if (epoch + 1) % 10 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), 'resnet_antenna_model.pth')