# coding:UTF-8
# @Time: 2024/10/1 21:49
# @Author: Lulu Cao
# @File: PoissonDataset.py
# @Software: PyCharm
import random
import numpy as np
import math
import torch

def advection1_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """
    return x1-t

def advection2_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """
    return np.sin(x1-t)




if __name__ == '__main__':
    x1 = np.random.uniform(0, 1, 10)
    x2 = np.random.uniform(0, 1, 10)
    x3 = np.random.uniform(0, 1, 10)
    t = np.random.uniform(0, 1, 10)
    advection1_1D_exact_solution(x1,t)