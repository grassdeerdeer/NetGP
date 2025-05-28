import random
import numpy as np
import math
import torch

def diffusion_reaction1_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """
   
    return np.exp(-2 * t) * np.sin(x1)

def diffusion_reaction2_1D_exact_solution(x1,t):
    """
    :param x1: np.ndarray
    :return:
    """
  
    return np.exp(-3 * t) * np.sin(x1)





if __name__ == '__main__':
    x1 = np.random.uniform(0, 1, 10)
    x2 = np.random.uniform(0, 1, 10)
    x3 = np.random.uniform(0, 1, 10)
    t = np.random.uniform(0, 1, 10)
    advection1_1D_exact_solution(x1,t)