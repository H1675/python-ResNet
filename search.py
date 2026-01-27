import torch
from torchvision.models import ResNet
from resnet import ResNetAntenna
import numpy as np
import matplotlib.pyplot as plt
from simanneal import Annealer


class Myproblem(Annealer):
    def __init__(self, initial_point, model_eva, para_bounds):
        super().__init__(initial_point)
        self.model_eva = model_eva
        self.para_bounds = para_bounds

    def energy(self):
        self.model_eva.eval()

        input_tensor = torch.from_numpy(self.state).float()
        with torch.no_grad():
            eva = self.model_eva(input_tensor)

        return eva.item()

    def move(self):
        noise = np.random.uniform(-0.1, 0.1, size=self.state.shape)
        self.state += noise
        self.state  = np.clip(self.state, self.para_bounds[:, 0], self.para_bounds[:, 1])

    def update(self, *args, **kwargs):
        pass


def search_sa(para_dim, model_eva):
    # 搜索初始点
    initial_point = np.random.rand(para_dim)

    # 变量范围
    para_bounds = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
    # para_bounds = np.array([[-10,10],[-10,10]])
    # para_bounds = np.array([[-10,10],[-10,10],[-10,10],[-10,10],[-10,10],[-10,10],[-10,10],[-10,10]])

    # 使用模拟退火进行优化
    problem = Myproblem(initial_point, model_eva, para_bounds)

    schedule = {'tmax':100, 'tmin':0.1, 'updates':100, 'steps':1000}
    problem.set_schedule(schedule)

    best_x, best_y = problem.anneal()

    return best_x, best_y

    # best_state, best_value, s11_history, state_history, s11_best, ftbr_best = simulated_annealing(
    #     model_s11, model_ftbr, ftbr_bounds, initial_state, para_bounds, ref_s11,
    #     temp_initial=100, temp_final=0.1, alpha=0.95, max_iter=1000)

    # print(f'最优天线设计参数（归一化）: {best_state}')
    # print(f'对应的最小评估值: {best_value}')
    # print(f'对应的S11值: {s11_best}')
    # print(f'对应的前后比: {ftbr_best}')

    # 绘制 S11 历史曲线
    # plt.plot(s11_history)
    # plt.xlabel('Iteration')
    # plt.ylabel('S11 Magnitude')
    # plt.title('S11 Magnitude vs Iteration')
    # plt.show()
    #
    # return best_state, best_value

