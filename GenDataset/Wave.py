# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gzip
import torch
import torch.nn.functional as F

#============================Wave 1-1D===========================
def Wave1_1D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(3*x1+3*t)"

    y = torch.sin(3*X[0]+3*X[1])
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'t':X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Wave1_1D_eva(func, X):
    #==================du2_dx1==================
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
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)
    #==================du2_dt==================
    try:
        du2_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dt = torch.autograd.grad(du2_dt[0].sum(), X[-1], allow_unused=True)

        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dt = du2_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dt = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    pde_residual = du2_dt - du2_dx1
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1

#============================Wave 1-2D===========================
def Wave1_2D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(3*x1+3*t)+sin(3*x2+3*t)"
    y = torch.sin(3*X[0]+3*X[2])+torch.sin(3*X[1]+3*X[2])

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'t':X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Wave1_2D_eva(func, X):
    #==================du2_dx1==================
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
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)
    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True)

        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    
    #==================du2_dt==================
    try:
        du2_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dt = torch.autograd.grad(du2_dt[0].sum(), X[-1], allow_unused=True)

        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dt = du2_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dt = torch.full(X[2].size(), 100, dtype=X[2].dtype)
    
    pde_residual = du2_dt - (du2_dx1 + du2_dx2)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1



#============================Wave 1-3D===========================
def Wave1_3D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(3*x1+3*t)+sin(3*x2+3*t)+sin(3*x3+3*t)"
    y = torch.sin(3*X[0]+3*X[3])+torch.sin(3*X[1]+3*X[3])+torch.sin(3*X[2]+3*X[3])

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'x3': X[2].detach().numpy(),'t':X[3].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Wave1_3D_eva(func, X):
    #==================du2_dx1==================
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
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)
    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True)

        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    #==================du2_dx3==================
    try:
        du2_dx3 = torch.autograd.grad(func.sum(),X[2], create_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = torch.autograd.grad(du2_dx3[0].sum(), X[2], allow_unused=True)

        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = du2_dx3[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx3 = torch.full(X[2].size(), 100, dtype=X[2].dtype)
    
    #==================du2_dt==================
    try:
        du2_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du2_dt = torch.autograd.grad(du2_dt[0].sum(), X[-1], allow_unused=True)

        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du2_dt = du2_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dt = torch.full(X[3].size(), 100, dtype=X[3].dtype)
    pde_residual = du2_dt - (du2_dx1 + du2_dx2 + du2_dx3)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1






























#============================Wave 2-1D===========================
def Wave2_1D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(x1+3*t)"
    y = torch.sin(X[0]+3*X[1])

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'t':X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Wave2_1D_eva(func, X):
    #==================du2_dx1==================
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
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)
    #==================du2_dt==================
    try:
        du2_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dt = torch.autograd.grad(du2_dt[0].sum(), X[-1], allow_unused=True)

        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dt = du2_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dt = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    pde_residual = du2_dt - 9*du2_dx1
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1

#============================Wave 2-2D===========================
def Wave2_2D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(x1+3*t)+sin(x2+3*t)"
    y = torch.sin(X[0]+3*X[2])+torch.sin(X[1]+3*X[2])

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'t':X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Wave2_2D_eva(func, X):
    #==================du2_dx1==================
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
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)
    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True)

        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    
    #==================du2_dt==================
    try:
        du2_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dt = torch.autograd.grad(du2_dt[0].sum(), X[-1], allow_unused=True)

        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dt = du2_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dt = torch.full(X[2].size(), 100, dtype=X[2].dtype)
    
    pde_residual = du2_dt - 9*(du2_dx1 + du2_dx2)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1



#============================Wave 2-3D===========================
def Wave2_3D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(x1+3*t)+sin(x2+3*t)+sin(x3+3*t)"
    y = torch.sin(X[0]+3*X[3])+torch.sin(X[1]+3*X[3])+torch.sin(X[2]+3*X[3])

    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'x3': X[2].detach().numpy(),'t':X[3].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str


def Wave2_3D_eva(func, X):
    #==================du2_dx1==================
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
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)
    #==================du2_dx2==================
    try:
        du2_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X[1], allow_unused=True)

        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    #==================du2_dx3==================
    try:
        du2_dx3 = torch.autograd.grad(func.sum(),X[2], create_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = torch.autograd.grad(du2_dx3[0].sum(), X[2], allow_unused=True)

        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du2_dx3 = du2_dx3[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dx3 = torch.full(X[2].size(), 100, dtype=X[2].dtype)
    
    #==================du2_dt==================
    try:
        du2_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du2_dt = torch.autograd.grad(du2_dt[0].sum(), X[-1], allow_unused=True)

        if du2_dt[0] is None:
            du2_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du2_dt = du2_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du2_dt = torch.full(X[3].size(), 100, dtype=X[3].dtype)
    
    
    pde_residual = du2_dt - 9*(du2_dx1 + du2_dx2 + du2_dx3)
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1






