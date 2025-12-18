import numpy as np
import os
from LHS import LHSample
import scipy.io as scio
from cst_simulation import cst_simulation

# 获取当前路径
path = os.getcwd()

# 临时文件路径
temp = path + "\\temp"
temp_name = temp + "\\temp.cst"

# 工程文件路径
filename = path + "\model\yagi_antenna.cst"

# 变量设置
para_name = ["D1","D2","L0","L1","L2","R1"]

# 变量范围
para_bounds = np.array([[20,30],[20,30],[20,25],[50,60],[30,40],[0.3,0.7]])

# 变量数量
para_dim = 6

# 采样数量
sample_num = 200

# 目标频率点 3GHz
target_freq_index =  667

# 拉丁超立方采样
# 每一行代表一个样本
sample = LHSample(para_dim,para_bounds,sample_num)
# print(sample)

# 样本仿真
# S11: 每一列是一个样本的仿真结果
# freq: 1-4GHz 步进0.003Ghz
S11,Dir,freq,angle = cst_simulation(sample,sample_num,para_name,para_dim,filename,temp_name)

target_S11 = S11[target_freq_index,:]

# 保存文件
data_path = path + "\\data" + "\\sample.mat"
scio.savemat(data_path,{'sample':sample})

data_path = path + "\\data" + "\\S11.mat"
scio.savemat(data_path,{'S11':S11})

data_path = path + "\\data" + "\\freq.mat"
scio.savemat(data_path,{'freq':freq})

data_path = path + "\\data" + "\\target_S11.mat"
scio.savemat(data_path,{'target_S11':target_S11})

data_path = path + "\\data" + "\\Dir.mat"
scio.savemat(data_path,{'Dir':Dir})

data_path = path + "\\data" + "\\angle.mat"
scio.savemat(data_path,{'angle':angle})
