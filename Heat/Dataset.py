import random
import numpy as np
import math
import torch

def heat1_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """
    k = 1.0
    return 1 / (np.sqrt(np.pi * 4 * k* t) ) * np.exp(- (x1 ** 2) / (4 * k * t))

def heat2_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """
    k = 0.4
    return 1 / (np.sqrt(np.pi * 4 * k * t) ) * np.exp(- (x1 ** 2) / (4 * k * t))





if __name__ == '__main__':
    x1 = np.random.uniform(0, 1, 10)
    x2 = np.random.uniform(0, 1, 10)
    x3 = np.random.uniform(0, 1, 10)
    t = np.random.uniform(0, 1, 10)
    advection1_1D_exact_solution(x1,t)