import numpy as np
import torch
import torch.optim as optim
from scipy.special import expit

# 1. 模拟退火算法实现
def simulated_annealing(model, initial_state, temp_initial=100, temp_final=0.1, alpha=0.95, max_iter=1000):
    # 初始化
    current_state = initial_state
    current_temp = temp_initial
    best_state = current_state
    best_s11 = float('inf')  # S11 初始为极大值

    # 初始化记录
    s11_history = []
    state_history = []

    for iteration in range(max_iter):
        # 计算当前状态的 S11 幅值（目标函数）
        current_s11 = predict_s11(model, current_state)

        # 如果当前的 S11 更好，则更新最优解
        if current_s11 < best_s11:
            best_s11 = current_s11
            best_state = current_state

        # 记录历史
        s11_history.append(current_s11)
        state_history.append(current_state)

        # 温度衰减
        current_temp *= alpha

        # 生成新解（扰动）
        new_state = current_state + np.random.uniform(-0.1, 0.1, size=current_state.shape)

        # 计算新解的 S11 幅值
        new_s11 = predict_s11(model, new_state)

        # 接受新解的准则：如果新解更好，或者根据概率接受更差的解
        if new_s11 < current_s11 or expit((current_s11 - new_s11) / current_temp) > np.random.rand():
            current_state = new_state

        # 逐步降温
        if current_temp < temp_final:
            break

    return best_state, best_s11, s11_history, state_history

# 2. 目标函数：通过 ResNet 模型预测 S11 幅值
def predict_s11(model, input_params):
    # 假设模型输入是一个长度为 10 的参数向量（天线设计参数）
    input_tensor = torch.tensor(input_params, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():  # 不需要梯度
        s11_pred = model(input_tensor)
    return s11_pred.item()

# 3. 定义目标参数的初始范围（例如，10个设计参数）
initial_state = np.random.rand(10)  # 10个天线设计参数

# 4. 假设已经有一个训练好的 ResNet 模型
# 在实际中，你应该先训练好模型
model = torch.load('resnet_antenna_model.pth')  # 加载已训练的模型

# 5. 使用模拟退火进行优化
best_state, best_s11, s11_history, state_history = simulated_annealing(
    model, initial_state, temp_initial=100, temp_final=0.1, alpha=0.95, max_iter=1000
)

print(f'最优天线设计参数: {best_state}')
print(f'对应的最小S11幅值: {best_s11}')

# 6. 绘制 S11 历史曲线
import matplotlib.pyplot as plt

plt.plot(s11_history)
plt.xlabel('Iteration')
plt.ylabel('S11 Magnitude')
plt.title('S11 Magnitude vs Iteration')
plt.show()