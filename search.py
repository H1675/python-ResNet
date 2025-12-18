import torch
from torchvision.models import ResNet
from resnet import ResNetAntenna
from sa import simulated_annealing,predict_s11
import numpy as np
import matplotlib.pyplot as plt

def search_sa(para_dim, para_bounds, model):
    # 变量数量
    # para_dim = 6

    # 变量范围
    # para_bounds = np.array([[20,30],[20,30],[20,25],[50,60],[30,40],[0.3,0.7]])

    # 导入模型
    # model = ResNetAntenna(input_dim=para_dim)
    # model.load_state_dict(torch.load('resnet_antenna_model.pth'))
    model.eval()

    # 搜索初始点
    initial_state = np.zeros(para_dim)
    for i in range(para_dim):
        initial_state[i] = np.random.random(1)*(para_bounds[i][1]-para_bounds[i][0])+para_bounds[i][0]

    # 使用模拟退火进行优化
    best_state, best_s11, s11_history, state_history = simulated_annealing(
        model, initial_state, para_bounds, temp_initial=200, temp_final=0.1, alpha=0.99, max_iter=1000)

    return best_state, best_s11

    # print(f'最优天线设计参数: {best_state}')
    # print(f'对应的最小S11幅值: {best_s11}')
    #
    # # 绘制 S11 历史曲线
    # plt.plot(s11_history)
    # plt.xlabel('Iteration')
    # plt.ylabel('S11 Magnitude')
    # plt.title('S11 Magnitude vs Iteration')
    # plt.show()



