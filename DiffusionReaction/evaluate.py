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

def diffusion_reaction1_torch_1d(func,X,y):
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

    try:
        # 计算函数的导数
        du_dt = torch.autograd.grad(func.sum(), X[1], create_graph=True)
        if du_dt[0] is None:
            du_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du_dt = du_dt[0]
    except:
        du_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)

    source_term = 3.0
    du2_dx1 = du_dt - source_term * du2_dx1 -func

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(du2_dx1 , torch.zeros_like(du2_dx1))
    mse2 = F.mse_loss(func,y)
    #print(mse1.item()+mse2.item())
    return mse1.item()+mse2.item()
    return mse1.item()

def diffusion_reaction2_torch_1d(func,X,y):
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

    try:
        # 计算函数的导数
        du_dt = torch.autograd.grad(func.sum(), X[1], create_graph=True)
        if du_dt[0] is None:
            du_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du_dt = du_dt[0]
    except:
        du_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)

    source_term = 2
    du2_dx1 = du_dt - source_term * du2_dx1 +func

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(du2_dx1 , torch.zeros_like(du2_dx1))
    mse2 = F.mse_loss(func,y)
    #print(mse1.item()+mse2.item())
    return mse1.item()+mse2.item()