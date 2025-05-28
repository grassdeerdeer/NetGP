import numpy as np
import torch
import importlib
import os
import yaml
import re
import sympy as sp
import matplotlib.pyplot as plt
from sympy import E


def visualize_expressionbymeta(problem_name, problem_index):
    expr_str = problem_name+problem_index+"_sol"
    eva_torch_str = problem_name+problem_index+"_eva"
    
    eva_module = importlib.import_module(problem_name)
    if hasattr(eva_module, expr_str):
        expr = getattr(eva_module, expr_str)
    if hasattr(eva_module, eva_torch_str):
        eva_torch = getattr(eva_module, eva_torch_str)



    save_root = "NewGPSR\PDEdataset"
    save_path_problem=os.path.join(problem_name,problem_index)
    save_path_dir = os.path.join(save_root, save_path_problem )
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path_image = os.path.join(save_path_dir,  "visualization.png")
    save_path_meatadata = os.path.join(save_path_dir,  "metadata.yaml")
    output_image = save_path_image

    # 读取 JSON 文件
    with open(save_path_meatadata , 'r') as f:
        data = yaml.load(f)

    # 提取 SymPy 表达式字符串
    sympy_expr_str = data.get('sol', None)
    sympy_expr_str = sympy_expr_str.replace('−', '-')
    sympy_expr_str = sympy_expr_str.replace('e', 'E')
    if not sympy_expr_str:
        raise ValueError("JSON 文件中未找到 'symbolic_sympy' 字段")

    # 将字符串转换为 SymPy 表达式
    sympy_expr = sp.sympify(sympy_expr_str)
    print(f"SymPy 表达式: {sympy_expr}")

    # 定义变量（假设变量为 x 和 y）
    if 'x2' in sympy_expr.free_symbols:
        x, y = sp.symbols('x1 x2')
    else:
        x, y = sp.symbols('x1 t')

    # 可视化表达式
    if len(sympy_expr.free_symbols) == 1:  # 单变量表达式
        x_vals = np.linspace(-10, 10, 500)
        y_vals = [float(sympy_expr.subs(x, val)) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f"${sp.latex(sympy_expr)}$")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Visualization of SymPy Expression')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_image)
        print(f"图像已保存到: {output_image}")

    elif len(sympy_expr.free_symbols) == 2:  # 双变量表达式
        x_vals = np.linspace(0.01, 0.99, 100)
        y_vals = np.linspace(0.01, 1.99, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        Z = np.array([[float(sympy_expr.subs({x: xv, y: yv})) for xv in x_vals] for yv in y_vals])
        #Z = np.array([[float(sympy_expr.subs({x: xv, y: yv})) for xv in x_vals] for yv in y_vals])

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Value')
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.title(f'Visualization of SymPy Expression: ${sp.latex(sympy_expr)}$')
        plt.savefig(output_image)
        print(f"图像已保存到: {output_image}")
        plt.close()

    else:
        raise ValueError("无法可视化超过两个变量的表达式")


def visualize_expressionbyfunc(problem_name, problem_index, func):
    expr_str = problem_name+problem_index+"_sol"
    eva_torch_str = problem_name+problem_index+"_eva"
    
    eva_module = importlib.import_module(problem_name)
    if hasattr(eva_module, expr_str):
        expr = getattr(eva_module, expr_str)
    if hasattr(eva_module, eva_torch_str):
        eva_torch = getattr(eva_module, eva_torch_str)



    save_root = "NewGPSR\PDEdataset"
    save_path_problem=os.path.join(problem_name,problem_index)
    save_path_dir = os.path.join(save_root, save_path_problem )
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path_image = os.path.join(save_path_dir,  "visualization.png")
    save_path_meatadata = os.path.join(save_path_dir,  "metadata.yaml")
    output_image = save_path_image

    

    

    
    x_vals = np.linspace(0.01, 0.99, 100)
    y_vals = np.linspace(0.01, 0.99, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.array([[float(func(xv,yv)) for xv in x_vals] for yv in y_vals])
    #Z = np.array([[float(sympy_expr.subs({x: xv, y: yv})) for xv in x_vals] for yv in y_vals])

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title(f'Visualization of SymPy Expression: ${sp.latex(sympy_expr)}$')
    plt.savefig(output_image)
    print(f"图像已保存到: {output_image}")
    plt.close()

    

if __name__ == '__main__':
    name_group = ["Diffusion_reaction", "Wave", "Heat", "Advection", "Poisson"]
    name_group = ["Poisson"]
    index_group = ["2_2D"]
    # sin(3.14*x1)*sin(3.14*x2)
    func = lambda x1, x2: (1-x1**2-x2**2)/4
    x1 = np.random.uniform(0.00001, 1, 1000)
    x2 = np.random.uniform(0.00001, 1, 1000)
    x3 = np.random.uniform(0.00001, 1, 1000)
    t = np.random.uniform(0.00001, 2, 1000)
    for problem_name in name_group:
        for problem_index in index_group:
            match = re.search(r'_(\d+)D', problem_index)
            dim = int(match.group(1))
            # 随机保存1000个点
            if dim == 1:
                X = np.stack((x1,t), axis=0)
            elif dim == 2:
                X = np.stack((x1,x2,t), axis=0)
            elif dim == 3:
                X = np.stack((x1,x2,x3,t), axis=0)
            X = torch.tensor(X).tolist() 
            visualize_expressionbyfunc(problem_name, problem_index, func)
   