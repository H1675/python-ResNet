import numpy as np
import os

def _init():
    global _global_dict
    _global_dict = {}

    path = os.getcwd()
    # 获取当前路径
    _global_dict["path"] = path

    # 临时文件路径
    _global_dict["temp"] = path + "//temp"
    _global_dict["temp_name"] = path + "//temp//temp.cst"

    # 工程文件路径
    _global_dict["filename"] = path + "//cst_model//test.cst"

    # 变量设置
    _global_dict["para_name"] = ["D1","D2","D3","L0","L1","L2","L3","R1"]

    # 变量范围
    _global_dict["para_bounds"] = np.array([[10,30],[10,30],[10,30],[15,30],[40,70],[30,55],[30,55],[2,3]])
    
    # 变量数量
    _global_dict["para_dim"] = 8
    
    # 采样数量
    _global_dict["sample_num"] = 100

    # 权重
    _global_dict["weight"] = np.array([0.5,0,0.5])

    # 目标值
    _global_dict["tar_s11"] = 0.3
    _global_dict["tar_ftbr"] = 20
    _global_dict["tar_gain"] = 8


def get_value(key):
    try:
        return _global_dict[key]
    except KeyError:
        print('读取' + key + '失败\r\n')