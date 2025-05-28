# coding:utf-8
import pandas as pd
import numpy as np
import gzip
import torch
import torch.nn.functional as F

#####################Poisson1_1D#####################

def Poisson1_1D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(3.14*x1)"
    
    y = torch.sin(torch.pi*X[0])
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Poisson1_1D_eva(func,X):
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

    # Define the source term
    source_term = -torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X[0])
    pde_residual = du2_dx1 - source_term

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(du2_dx1))
    # mse2 = F.mse_loss(func,y)
    # print(mse1.item()+mse2.item())
    # return mse1.item()+mse2.item()
    return mse1


###################Poisson1_2D#####################

def Poisson1_2D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(3.14*x1)*sin(3.14*x2)"
    
    y = torch.sin(torch.pi*X[0])*torch.sin(torch.pi*X[1])
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(), 'x2': X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    
    return expr_str

def Poisson1_2D_eva(func,X):
    #====================du2_dx1====================
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

    #====================du2_dx2====================
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

    # Define the source term
    source_term = -2*torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X[0])*torch.sin(torch.tensor(np.pi) * X[1])
    pde_residual = du2_dx1 + du2_dx2 - source_term

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(du2_dx1))
    # mse2 = F.mse_loss(func,y)
    # print(mse1.item()+mse2.item())
    # return mse1.item()+mse2.item()
    return mse1


###################Poisson1_3D#####################
def Poisson1_3D_sol(X,eva_torch,save_path_data):
    expr_str = "sin(3.14*x1)*sin(3.14*x2)*sin(3.14*x3)"
    
    y = torch.sin(torch.pi*X[0])*torch.sin(torch.pi*X[1])*torch.sin(torch.pi*X[2])
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(), 'x2': X[1].detach().numpy(), 'x3': X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    
    return expr_str

def Poisson1_3D_eva(func,X):

    #====================du2_dx1====================
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

    #====================du2_dx2====================
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

    #====================du3_dx3====================
    try:
        du3_dx3 = torch.autograd.grad(func.sum(),X[2], create_graph=True)
        if du3_dx3[0] is None:
            du3_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)

        else:
            du3_dx3 = torch.autograd.grad(du3_dx3[0].sum(), X[2], allow_unused=True)

        if du3_dx3[0] is None:
            du3_dx3 = torch.zeros(X[2].size(), dtype=X[2].dtype)
        else:
            du3_dx3 = du3_dx3[0]
    except Exception as error:
        print("Error in computing grad:", error)
        du3_dx3 = torch.full(X[2].size(), 100, dtype=X[2].dtype)
    # Define the source term
    source_term = -3*torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X[0])*torch.sin(torch.tensor(np.pi) * X[1])*torch.sin(torch.tensor(np.pi) * X[2])
    pde_residual = du2_dx1 + du2_dx2 + du3_dx3 - source_term
    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(du2_dx1))
    # mse2 = F.mse_loss(func,y)
    # print(mse1.item()+mse2.item())
    # return mse1.item()+mse2.item()
    return mse1







#####################Poisson2_1D#####################
def Poisson2_1D_sol(X,eva_torch,save_path_data):
    expr_str = "(1-x1**2)/2"
    
    y = (1-X[0]**2)/2
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    

    return expr_str

def Poisson2_1D_eva(func,X):
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

    # Define the source term
    source_term = -torch.tensor(1.0)
    pde_residual = du2_dx1 - source_term

    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(du2_dx1))
    # mse2 = F.mse_loss(func,y)
    # print(mse1.item()+mse2.item())
    # return mse1.item()+mse2.item()
    return mse1


###################Poisson2_2D#####################
def Poisson2_2D_sol(X,eva_torch,save_path_data):
    expr_str = "(1-x1**2-x2**2)/4"
    
    y = (1-X[0]**2-X[1]**2)/4
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(), 'x2': X[1].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    
    return expr_str

def Poisson2_2D_eva(func,X):
    #====================du2_dx1====================
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

    #====================du2_dx2====================
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
    
    # Define the source term
    source_term = -torch.tensor(1.0)
    pde_residual = du2_dx1 + du2_dx2 - source_term
    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(du2_dx1))
    # mse2 = F.mse_loss(func,y)
    # print(mse1.item()+mse2.item())
    # return mse1.item()+mse2.item()
    return mse1

###################Poisson2_3D#####################
def Poisson2_3D_sol(X,eva_torch,save_path_data):
    expr_str = "(1-x1**2-x2**2-x3**2)/6"
    
    y = (1-X[0]**2-X[1]**2-X[2]**2)/6
    pde_error = eva_torch(y,X)
    print("pde_error:",pde_error.item())
    
    df = pd.DataFrame({'x1': X[0].detach().numpy(), 'x2': X[1].detach().numpy(), 'x3': X[2].detach().numpy(), 'target': y.detach().numpy()})
    # 将DataFrame存储成.tsv.gz数据压缩包
    with gzip.open(save_path_data, 'wt') as f:
        df.to_csv(f, sep='\t', index=False)
    
    return expr_str

def Poisson2_3D_eva(func,X):
    #====================du2_dx1====================
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

    #====================du2_dx2====================
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
    #====================du2_dx3====================
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

    # Define the source term
    source_term = -torch.tensor(1.0)
    pde_residual = du2_dx1 + du2_dx2 + du2_dx3 - source_term
    # Compute the MSE of the Poisson equation
    mse1 = F.mse_loss(pde_residual , torch.zeros_like(du2_dx1))
    # mse2 = F.mse_loss(func,y)
    # print(mse1.item()+mse2.item())
    # return mse1.item()+mse2.item()
    return mse1
