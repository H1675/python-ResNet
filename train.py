import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset
from resnet import ResNetAntenna
import torch.nn as nn
import torch.optim as optim

def train_model(sample_train,target_S11_train,para_dim,output_dim,num_epochs):
    # 创建DataLoader
    train_data = TensorDataset(torch.from_numpy(sample_train), torch.from_numpy(target_S11_train))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # 初始化模型，损失函数和优化器
    model = ResNetAntenna(input_dim=para_dim,output_dim=output_dim)
    # model.load_state_dict(torch.load('weights/weights.pth'))
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # num_epochs = 40 # 训练轮数

    # 训练模型
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
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    return model