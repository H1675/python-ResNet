from search import search_sa
from resnet import  ResNetAntenna
import torch
import numpy as np

# 变量数量
para_dim = 6

# 变量范围
para_bounds = np.array([[20,30],[20,30],[20,25],[50,60],[30,40],[0.3,0.7]])

# 导入模型
model = ResNetAntenna(input_dim=para_dim)
model.load_state_dict(torch.load('resnet_antenna_model.pth'))

# 模拟退火
best_state, best_s11 = search_sa(para_dim, para_bounds, model)