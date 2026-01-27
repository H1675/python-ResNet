import numpy as np

def normalization(data,para_bounds,size):
    temp = np.zeros((data.shape[0],data.shape[1]))
    for i in range(size):
        min = para_bounds[i][0]
        max = para_bounds[i][1]
        temp[:,i] = (data[:,i] - min) / (max - min)

    return temp

def min_max_normalization(data,size):
    data_proc = np.zeros((data.shape[0],data.shape[1]))
    for i in range(size):
        temp = data[:,i]
        min = np.min(temp)
        max = np.max(temp)
        data_proc[:,i] = (temp-min)/(max-min)
    return data_proc
