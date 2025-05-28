# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gzip
import torch
import torch.nn.functional as F


#===========================Heat 1-1D===========================
def Heat1_1D_sol(X,eva_torch,save_path_data):
    expr_str = "1/(4*pi*alpha*t)**(1/2)*exp(-x1^2/(4*alpha*t))"
    alpha = 0.4
    d = 1
    y = 1/(4*torch.pi*alpha*X[1])**(d/2)*torch.exp(-X[0]**2/(4*alpha*X[1]))

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'t':X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Heat1_1D_eva(func, X):
    #==================du2_dx1==================
    try:
        du2_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)

        else:
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True, retain_graph=True)

        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]
    except Exception as error:
        print("Error in computing du2_dx1:", error)
        
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True,retain_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        # 处理错误，例如设置默认值或记录错误
        du1_dt = torch.full(X[1].size(), 100, dtype=X[1].dtype)


    pde_residual = du1_dt - 0.4*du2_dx1
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1

#==============================Heat 1-2D===========================
def Heat1_2D_sol(X,eva_torch,save_path_data):
    expr_str = "1/(4*pi*alpha*t)*exp(-x1^2/(4*alpha*t))*exp(-x2^2/(4*alpha*t))"
    alpha = 0.4
    d = 2
    y = 1/(4*torch.pi*alpha*X[2])**(d/2)*torch.exp(-X[0]**2/(4*alpha*X[2]))*torch.exp(-X[1]**2/(4*alpha*X[2]))

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'t':X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Heat1_2D_eva(func, X):
    #==================du2_dx1==================
    try:
        du2_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)
    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True, retain_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[2].size(), 100, dtype=X[2].dtype)

    pde_residual = du1_dt - 0.4*(du2_dx1 + du2_dx2)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1
#==============================Heat 1-3D===========================
def Heat1_3D_sol(X,eva_torch,save_path_data):
    expr_str = "1/(4*pi*alpha*t)**(3/2)*exp(-x1^2/(4*alpha*t))*exp(-x2^2/(4*alpha*t))*exp(-x3^2/(4*alpha*t))"
    alpha = 0.4
    d = 3
    y = 1/(4*torch.pi*alpha*X[3])**(d/2)*torch.exp(-X[0]**2/(4*alpha*X[3]))*torch.exp(-X[1]**2/(4*alpha*X[3]))*torch.exp(-X[2]**2/(4*alpha*X[3]))

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'x3': X[2].detach().numpy(),'t':X[3].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Heat1_3D_eva(func, X):
    #==================du2_dx1==================
    try:
        du2_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)
    #==================du2_dx3==================
    try:
        du2_dx3 = torch.autograd.grad(func.sum(),X[2], create_graph=True, retain_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = torch.autograd.grad(du2_dx3[0].sum(), X[2], allow_unused=True, retain_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = du2_dx3[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx3 = torch.full(X[2].size(), 100, dtype=X[2].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True, retain_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[3].size(), 100, dtype=X[3].dtype)

    pde_residual = du1_dt - 0.4*(du2_dx1 + du2_dx2 + du2_dx3)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1
















































#===========================Heat 2-1D===========================
def Heat2_1D_sol(X,eva_torch,save_path_data):
    expr_str = "1/(4*pi*alpha*t)**(1/2)*exp(-x1^2/(4*alpha*t))"
    alpha = 1
    d = 1
    y = 1/(4*torch.pi*alpha*X[1])**(d/2)*torch.exp(-X[0]**2/(4*alpha*X[1]))

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'t':X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Heat2_1D_eva(func, X):
    #==================du2_dx1==================
    try:
        du2_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)

        else:
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True, retain_graph=True)

        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]
    except Exception as error:
        print("Error in computing du2_dx1:", error)
        
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True,retain_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        # 处理错误，例如设置默认值或记录错误
        du1_dt = torch.full(X[1].size(), 100, dtype=X[1].dtype)


    pde_residual = du1_dt - du2_dx1
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1

#==============================Heat 2-2D===========================
def Heat2_2D_sol(X,eva_torch,save_path_data):
    expr_str = "1/(4*pi*alpha*t)*exp(-x1^2/(4*alpha*t))*exp(-x2^2/(4*alpha*t))"
    alpha = 1
    d = 2
    y = 1/(4*torch.pi*alpha*X[2])**(d/2)*torch.exp(-X[0]**2/(4*alpha*X[2]))*torch.exp(-X[1]**2/(4*alpha*X[2]))

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'t':X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Heat2_2D_eva(func, X):
    #==================du2_dx1==================
    try:
        du2_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)
    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True, retain_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[2].size(), 100, dtype=X[2].dtype)

    pde_residual = du1_dt - (du2_dx1 + du2_dx2)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1
#==============================Heat 2-3D===========================
def Heat2_3D_sol(X,eva_torch,save_path_data):
    expr_str = "1/(4*pi*alpha*t)**(3/2)*exp(-x1^2/(4*alpha*t))*exp(-x2^2/(4*alpha*t))*exp(-x3^2/(4*alpha*t))"
    alpha = 1
    d = 3
    y = 1/(4*torch.pi*alpha*X[3])**(d/2)*torch.exp(-X[0]**2/(4*alpha*X[3]))*torch.exp(-X[1]**2/(4*alpha*X[3]))*torch.exp(-X[2]**2/(4*alpha*X[3]))

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'x3': X[2].detach().numpy(),'t':X[3].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Heat2_3D_eva(func, X):
    #==================du2_dx1==================
    try:
        du2_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X[0], allow_unused=True, retain_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du2_dx1 = du2_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True, retain_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)
    #==================du2_dx3==================
    try:
        du2_dx3 = torch.autograd.grad(func.sum(),X[2], create_graph=True, retain_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = torch.autograd.grad(du2_dx3[0].sum(), X[2], allow_unused=True, retain_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = du2_dx3[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx3 = torch.full(X[2].size(), 100, dtype=X[2].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True, retain_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[3].size(), 100, dtype=X[3].dtype)

    pde_residual = du1_dt - (du2_dx1 + du2_dx2 + du2_dx3)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1