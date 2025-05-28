# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gzip
import torch
import torch.nn.functional as F
# 只需一组评估函数：Advection1_1D_eva,
# Advection1_2D_eva, Advection1_3D_eva

#===========================Advection 1-1D===========================
def Advection1_1D_sol(X,eva_torch,save_path_data):
    expr_str = "x1-t"

    y = X[0]-X[1]
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'t':X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Advection1_1D_eva(func, X):
    #==================du1_dx1==================
    try:
        du1_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True)
        if du1_dx1[0] is None:
            du1_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du1_dx1 = du1_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[1].size(), 100, dtype=X[1].dtype)
    
    pde_residual = du1_dt + du1_dx1
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1


#==============================Advection 1-2D===========================
def Advection1_2D_sol(X,eva_torch,save_path_data):
    expr_str = "x1+x2-2*t"

    y = X[0]+X[1]-2*X[2]
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'t':X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Advection1_2D_eva(func, X):
    #==================du1_dx1==================
    try:
        du1_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True)
        if du1_dx1[0] is None:
            du1_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du1_dx1 = du1_dx1[0]

    # 打印错误信息
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx1 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

 

    #==================du1_dx2==================
    try:
        du1_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du1_dx2[0] is None:
            du1_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dx2 = du1_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[2].size(), 100, dtype=X[2].dtype)
    
    pde_residual =  du1_dt +  du1_dx1 +  du1_dx2
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1

#==============================Advection 1-3D===========================
def Advection1_3D_sol(X,eva_torch,save_path_data):
    expr_str = "x1+x2+x3-3*t"

    y = X[0]+X[1]+X[2]-3*X[3]
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'x3': X[2].detach().numpy(),'t':X[3].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Advection1_3D_eva(func, X):
    #==================du1_dx1==================
    try:
        du1_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True)
        if du1_dx1[0] is None:
            du1_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du1_dx1 = du1_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du1_dx2==================
    try:
        du1_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du1_dx2[0] is None:
            du1_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dx2 = du1_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    #==================du1_dx3==================
    try:
        du1_dx3 = torch.autograd.grad(func.sum(),X[2], create_graph=True)
        if du1_dx3[0] is None:
            du1_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du1_dx3 = du1_dx3[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx3 = torch.full(X[2].size(), 100, dtype=X[2].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[3].size(), 100, dtype=X[3].dtype)
    
    pde_residual =  du1_dt +  du1_dx1 +  du1_dx2 +  du1_dx3
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1


#==============================Advection 2-1D===========================
def Advection2_1D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(x1-t)"

    y = torch.sin(X[0]-X[1])
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'t':X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Advection2_1D_eva(func, X):
    #==================du1_dx1==================
    try:
        du1_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True)
        if du1_dx1[0] is None:
            du1_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du1_dx1 = du1_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[1].size(), 100, dtype=X[1].dtype)
    
    pde_residual = du1_dt + du1_dx1
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1


#==============================Advection 2-2D===========================
def Advection2_2D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(x1-t)+sin(x2-t)"

    y = torch.sin(X[0]-X[2])+torch.sin(X[1]-X[2])
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'t':X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Advection2_2D_eva(func, X):
    #==================du1_dx1==================
    try:
        du1_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True)
        if du1_dx1[0] is None:
            du1_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du1_dx1 = du1_dx1[0]

    # 打印错误信息
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx1 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

 

    #==================du1_dx2==================
    try:
        du1_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du1_dx2[0] is None:
            du1_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dx2 = du1_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[2].size(), 100, dtype=X[2].dtype)
    
    pde_residual =  du1_dt +  du1_dx1 +  du1_dx2
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1

#==============================Advection 2-3D===========================
def Advection2_3D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(x1-t)+sin(x2-t)+sin(x3-t)"

    y = torch.sin(X[0]-X[3])+torch.sin(X[1]-X[3])+torch.sin(X[2]-X[3])
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(),'x2': X[1].detach().numpy(),'x3': X[2].detach().numpy(),'t':X[3].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Advection2_3D_eva(func, X):
    #==================du1_dx1==================
    try:
        du1_dx1 = torch.autograd.grad(func.sum(),X[0], create_graph=True)
        if du1_dx1[0] is None:
            du1_dx1 = torch.zeros(X[0].size(), dtype=X[0].dtype)
        else:
            du1_dx1 = du1_dx1[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx1 = torch.full(X[0].size(), 100, dtype=X[0].dtype)

    #==================du1_dx2==================
    try:
        du1_dx2 = torch.autograd.grad(func.sum(),X[1], create_graph=True)
        if du1_dx2[0] is None:
            du1_dx2 = torch.zeros(X[1].size(), dtype=X[1].dtype)
        else:
            du1_dx2 = du1_dx2[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx2 = torch.full(X[1].size(), 100, dtype=X[1].dtype)

    #==================du1_dx3==================
    try:
        du1_dx3 = torch.autograd.grad(func.sum(),X[2], create_graph=True)
        if du1_dx3[0] is None:
            du1_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du1_dx3 = du1_dx3[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dx3 = torch.full(X[2].size(), 100, dtype=X[2].dtype)

    #==================du1_dt==================
    try:
        du1_dt = torch.autograd.grad(func.sum(),X[-1], create_graph=True)
        if du1_dt[0] is None:
            du1_dt = torch.zeros(X[3].size(), dtype=X[3].dtype)
        else:
            du1_dt = du1_dt[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du1_dt = torch.full(X[3].size(), 100, dtype=X[3].dtype)
    
    pde_residual =  du1_dt +  du1_dx1 +  du1_dx2 +  du1_dx3
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(pde_residual))
    return mse1

