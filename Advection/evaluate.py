import math
import torch
import torch.nn.functional as F
import numpy as np

def advection_torch_1d(func, X, y):
    
    try:
        # 计算函数的导数
        du_dx = torch.autograd.grad(func.sum(), X[0], create_graph=True)
        if du_dx[0] is None:
            du_dx = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du_dx = du_dx[0]
    except:
        du_dx = torch.zeros(X[0].size(), dtype=X[0].dtype)
    
    try:
        # 计算函数的导数
        du_dt = torch.autograd.grad(func.sum(), X[1], create_graph=True)
        if du_dt[0] is None:
            du_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du_dt = du_dt[0]
    except:
        du_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)

    

    # 计算残差
    du_dx = du_dx + du_dt

    # 计算对流方程的MSE
    mse_advection = F.mse_loss(du_dx, torch.zeros_like(du_dx))
    mse_func = F.mse_loss(func, y)

    #return mse_advection.item() + mse_func.item()
    return mse_advection.item()