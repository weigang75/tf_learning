import numpy as np


def unknow(x):
    W = 3.0
    b = 2.0
    noise = np.random.normal()/10000.0
    return x * W + b + noise


