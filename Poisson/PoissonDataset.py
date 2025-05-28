# coding:UTF-8
# @Time: 2024/10/1 21:49
# @Author: Lulu Cao
# @File: PoissonDataset.py
# @Software: PyCharm
import random
import numpy as np
import math
import torch

def poissionsin_1D_exact_solution(x1):
    """
    :param x1: np.ndarray
    :return:
    """
    return np.sin(np.pi*x1)
def poissionsin_2D_exact_solution(x1, x2,):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :return:
    """
    return np.sin(np.pi*x1)*np.sin(np.pi*x2)


def poissionsin_3D_exact_solution(x1, x2, x3):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :return:
    """
    return np.sin(np.pi*x1)*np.sin(np.pi*x2)*np.sin(np.pi*x3)

def poission1_1D_exact_solution(x1):
    """
    :param x1: np.ndarray
    :return:
    """
    return 0.5-0.5*x1*x1
def poission1_2D_exact_solution(x1, x2):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :return:
    """
    return 0.25-0.25*x1*x1-0.25*x2*x2


def poission1_3D_exact_solution(x1, x2, x3):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :return:
    """
    return 1/6-1/6*x1*x1-1/6*x2*x2-1/6*x3*x3



def Dataset():
    # -----------------------------------
    # ------------Dataset----------------
    # -----------------------------------

    # Poisson1_3D
    x1 = np.random.uniform(0, 1, 50)
    x2 = np.random.uniform(0, 1, 50)
    x3 = np.random.uniform(0, 1, 50)

    # X1D = np.stack((x1,), axis=0)
    # X2D = np.stack((x1, x2), axis=0)
    X = np.stack((x1, x2, x3), axis=0)
    X = torch.tensor(X)  # .to(DEVICE)

    y1D = poissionsin_1D_exact_solution(x1)
    y2D = poissionsin_2D_exact_solution(x1, x2)
    y3D = poissionsin_3D_exact_solution(x1, x2, x3)
    y1 = torch.tensor(y1D)  # .to(DEVICE)
    y2 = torch.tensor(y2D)  # .to(DEVICE)
    y3 = torch.tensor(y3D)  # .to(DEVICE)

    X_names = ['x1', 'x2', 'x3']
    return X,y1,y2,y3,X_names

if __name__ == '__main__':
    x1 = np.random.uniform(0, 1, 10)
    x2 = np.random.uniform(0, 1, 10)
    x3 = np.random.uniform(0, 1, 10)
    poission1_3D_exact_solution(x1, x2, x3)