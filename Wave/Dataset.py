import random
import numpy as np
import math
import torch

def wave1_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """

    return np.sin(3*x1+3*t)

def wave2_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """

    return np.sin(x1+3*t)





if __name__ == '__main__':
    x1 = np.random.uniform(0, 1, 10)
    x2 = np.random.uniform(0, 1, 10)
    x3 = np.random.uniform(0, 1, 10)
    t = np.random.uniform(0, 1, 10)
    advection1_1D_exact_solution(x1,t)