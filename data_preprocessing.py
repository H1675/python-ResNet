import scipy.io as scio
import os
import numpy as np
from standardized import normalization,min_max_normalization
from global_var import _init,get_value
import time
from global_var import _init,get_value

print(time.ctime())

# 初始化化变量
_init()

# 获取当前路径
path = os.getcwd()

# 权重
weight = get_value("weight")

# 目标值
tar_s11 = get_value("tar_s11")
tar_ftbr = get_value("tar_ftbr")
tar_gain = get_value("tar_gain")

# 读取S11数据
data_path = get_value("path") + "\\data" + "\\S11.mat"
data = scio.loadmat(data_path)
S11 = data['S11']

s11_norm = (S11 - tar_s11) / tar_s11
s11_norm = s11_norm.astype('float32')

# 读取FTBR数据
data_path = get_value("path") + "\\data" + "\\FTBR.mat"
data = scio.loadmat(data_path)
FTBR = data['FTBR']

ftbr_norm = (-1) * (FTBR - tar_ftbr) / tar_ftbr
ftbr_norm = ftbr_norm.astype('float32')

# 读取Gain数据
data_path = get_value("path") + "\\data" + "\\Gain.mat"
data = scio.loadmat(data_path)
Gain = data['Gain']

gain_norm = (-1) * (Gain - tar_gain) / tar_gain
gain_norm = gain_norm.astype('float32')

# 计算评估值
eva = s11_norm * weight[0] + (-1) * ftbr_norm * weight[1] + (-1) * gain_norm * weight[2]
eva = eva.astype('float32')

data_path = get_value("path") + "\\data" + "\\eva.mat"
scio.savemat(data_path,{'eva':eva})

print(time.ctime())
