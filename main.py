from search import search_sa
import torch
import numpy as np
from cst_simulation import cst_simulation
import scipy.io as scio
from train import train_model
from standardized import normalization
import time
from global_var import _init,get_value

print(time.ctime())

# 初始化变量
_init()

# 变量数量
para_dim = get_value("para_dim")

# 变量范围
para_bounds = get_value("para_bounds")

# 权重
weight = get_value("weight")

# 目标值
tar_s11 = get_value("tar_s11")
tar_ftbr = get_value("tar_ftbr")
tar_gain = get_value("tar_gain")

# 过程记录
best_sample = np.zeros((12,1)).reshape(1,-1)
best_record = np.zeros((12,1)).reshape(1,-1)
all_record  = np.zeros((12,1)).reshape(1,-1)
temp_record = np.zeros((4,1)).reshape(1,-1)

# 读取sample数据
data_path = get_value("path") + "\\data" + "\\sample.mat"
data = scio.loadmat(data_path)
sample = data['sample']

sample = sample[0:100,:]
sample_norm = normalization(sample,para_bounds,para_dim)
sample_norm = sample_norm.astype('float32')

# 读取评估值数据
data_path = get_value("path") + "\\data" + "\\eva.mat"
data = scio.loadmat(data_path)
eva = data['eva']

eva = eva[0:100,:]
eva = eva.astype('float32')

# 训练模型
model_eva = train_model(sample_norm,eva,para_dim,1,30)

for j in range(50):

    print(f"第{j+1}次迭代")

    # 模拟退火
    best_x, best_y = search_sa(para_dim, model_eva)

    # 添加到训练集中
    best_x = best_x.reshape(1,-1)
    best_x = best_x.astype('float32')
    sample_norm = np.vstack((sample_norm, best_x))

    # 仿真
    for i in range(para_dim):
        best_x[0][i] = best_x[0][i] * (para_bounds[i][1] - para_bounds[i][0]) + para_bounds[i][0]

    sample = np.vstack((sample,best_x))
    s11, ftbr, gain = cst_simulation(best_x, 1, get_value("para_name"), para_dim, get_value("filename"), get_value("temp_name"))

    # 处理s11
    temp_record[0][0] = s11
    s11 = (s11 - tar_s11) / tar_s11
    s11 = s11.astype('float32')

    # 处理ftbr
    temp_record[0][1] = ftbr
    ftbr = (-1) * (ftbr - tar_ftbr) / tar_ftbr
    ftbr = ftbr.astype('float32')

    # 处理gain
    temp_record[0][2] = gain
    gain = (-1) * (gain - tar_gain) / tar_gain
    gain = gain.astype('float32')

    # 计算eva_new
    eva_new = s11 * weight[0] + ftbr * weight[1] + gain * weight[2]
    temp_record[0][3] = eva_new

    # 添加到数据集中
    eva = np.vstack((eva,eva_new))

    # 更新模型
    model_eva = train_model(sample_norm, eva, para_dim, 1, 30)

    # 记录结果
    temp = np.vstack((best_x.reshape(-1,1),temp_record.reshape(-1,1))).reshape(1,-1)

    if j == 0:
        best_sample = temp
    elif temp_record[0,-1] < best_sample[0,-1]:
        best_sample = temp

    if j == 0:
        best_record = temp
    else:
        best_record = np.vstack((best_record,best_sample))

    if j == 0:
        all_record = temp
    else:
        all_record = np.vstack((all_record,temp))


# print(best_sample)
# print(best_record)
# print(all_record)

# 保存文件
data_path = get_value("path") + "\\data" + "\\best_record.mat"
scio.savemat(data_path,{'best_record':best_record})

data_path = get_value("path") + "\\data" + "\\all_record.mat"
scio.savemat(data_path,{'all_record':all_record})

data_path = get_value("path") + "\\data" + "\\best_sample.mat"
scio.savemat(data_path,{'best_sample':best_sample})

data_path = get_value("path") + "\\data" + "\\sample_after.mat"
scio.savemat(data_path,{'sample_after':sample})

data_path = get_value("path") + "\\data" + "\\eva.mat"
scio.savemat(data_path,{'eva':eva})

# 保存模型
torch.save(model_eva.state_dict(), 'model_eva.pth')

print(time.ctime())