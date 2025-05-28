# coding:UTF-8
# @Time: 2024/9/3 21:13
# @Author: Lulu Cao
# @File: evaluate.py
# @Software: PyCharm

import math
from sympy import symbols, diff, lambdify,Derivative,Function,simplify,expand
import warnings
import numpy as np
import sympy
import torch
import torch.nn.functional as F
from sympy import Symbol
from sympy.utilities.lambdify import lambdify
import sympy, torch, sympytorch
import inspect

def torch_Poisson1d(func,X,y):
    try:
        du2_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)

        else:
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True)

        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]
    except:
        du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)

    # Define the source term
    source_term = -torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X[0])
    du2_dx1 = du2_dx1 - source_term

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(du2_dx1 , torch.zeros_like(du2_dx1))
    mse2 = F.mse_loss(func,y)
    #print(mse1.item()+mse2.item())
    return mse1.item()+mse2.item()

def torch_Poisson2d(func,X,y):
    try:
        # 一阶导
        du2_dx1 = torch.autograd.grad(func.sum(), X[0], create_graph=True,)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            # 二阶导
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True)

        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1= du2_dx1[0]

        # 一阶导
        du2_dx2 = torch.autograd.grad(func.sum(), X[1], create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            # 二阶导
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx2= du2_dx2[0]

        du_x1_x2 = du2_dx1+du2_dx2
    except:
        du_x1_x2 = torch.zeros(X[0].size(), dtype=X[0].dtype)



    # Define the source term
    source_term = -2*torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X[0]) * torch.sin(torch.tensor(np.pi) * X[1])
    du_x1_x2 = du_x1_x2 - source_term

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(du_x1_x2 , torch.zeros_like(du_x1_x2))
    #print(mse.item())

    mse2 = F.mse_loss(func, y)
    # print(mse1.item()+mse2.item())
    return mse1.item() + mse2.item()

def torch_Poisson3d(func,X,y):
    try:
        # 一阶导
        du2_dx1 = torch.autograd.grad(func.sum(), X[0], create_graph=True, )
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            # 二阶导
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True)

        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]

        # 一阶导
        du2_dx2 = torch.autograd.grad(func.sum(), X[1], create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            # 二阶导
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx2 = du2_dx2[0]

        # 一阶导
        du2_dx3 = torch.autograd.grad(func.sum(), X[2], create_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            # 二阶导
            du2_dx3 = torch.autograd.grad(du2_dx3[0].sum(), X[2], allow_unused=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx3 = du2_dx3[0]


        du_x1_x2_x3 = du2_dx1 + du2_dx2 + du2_dx3
    except:
        du_x1_x2_x3 = torch.zeros(X[0].size(), dtype=X[0].dtype)

    # Define the source term
    source_term = -3*torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X[0]) * torch.sin(
        torch.tensor(np.pi) * X[1])* torch.sin(
        torch.tensor(np.pi) * X[2])
    du_x1_x2_x3 = du_x1_x2_x3 - source_term

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(du_x1_x2_x3, torch.zeros_like(du_x1_x2_x3))
    # print(mse.item())

    mse2 = F.mse_loss(func, y)
    # print(mse1.item()+mse2.item())
    return mse1.item() + mse2.item()



