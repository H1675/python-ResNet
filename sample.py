import numpy as np
import os
from lhs import LHSample
import scipy.io as scio
from cst_simulation import cst_simulation
import time
from global_var import _init,get_value

print(time.ctime())

# 初始化变量
_init()

# 拉丁超立方采样
# 每一行代表一个样本
sample = LHSample(get_value("para_dim"),get_value("para_bounds"),get_value("sample_num"))

# 样本仿真
# freq: 1-4GHz 步进0.003Ghz
S11, FTBR, Gain = cst_simulation(sample,get_value("sample_num"),
                                get_value("para_name"),get_value("para_dim"),
                                get_value("filename"),get_value("temp_name"))



# 保存文件
data_path = get_value("path") + "\\data\\sample.mat"
scio.savemat(data_path,{'sample':sample})

data_path = get_value("path") + "\\data\\S11.mat"
scio.savemat(data_path,{'S11':S11})

data_path = get_value("path") + "\\data\\FTBR.mat"
scio.savemat(data_path,{'FTBR':FTBR})

data_path = get_value("path") + "\\data\Gain.mat"
scio.savemat(data_path,{'Gain':Gain})

print(time.ctime())