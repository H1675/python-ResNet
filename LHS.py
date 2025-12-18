import numpy as np

# param D:参数个数
# param bounds:参数对应范围（list）
# param N:拉丁超立方层数
# return:样本数据
def LHSample( D,bounds,N):
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size = 1)[0]

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    #对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    return result