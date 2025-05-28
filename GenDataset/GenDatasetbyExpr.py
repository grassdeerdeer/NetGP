# -*- coding: utf-8 -*-
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import os
import yaml

# from Poisson import *
# from Advection import *
# from Heat import *
# from Wave import *
# from Diffusion_reaction import *
import importlib
import re

def gen_dataset(problem_name, problem_index):
    
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
    save_path_meatadata = os.path.join(save_path_dir,  "metadata.yaml")
    save_path_data = os.path.join(save_path_dir,  "data.tsv.gz")
     
    for i in range(len(X)):
        X[i]=torch.tensor(X[i],requires_grad=True)
    sol_str = expr(X,eva_torch,save_path_data)

    

    subdata = {"problem_name": problem_name,
        'problem_index': problem_index,
        'sol': sol_str,
        'eva': eva_torch.__name__,}
    # 将字典写入.yaml文件
    with open(save_path_meatadata, 'w') as file:
        yaml.dump(subdata, file)


    
    pass

if __name__ == '__main__':

    name_group = ["Diffusion_reaction", "Wave", "Heat", "Advection", "Poisson"]
    #name_group = ["Advection"]
    index_group = ["1_1D", "1_2D", "1_3D", "2_1D", "2_2D", "2_3D", ]
    
    # problem_name = "Heat"#"Diffusion_reaction/1-1D" #"Wave/1-1D" #"Heat/1-1D" #"Advection/1-1D" # "Poisson/1-1D"
    # problem_index = "1_1D"

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
            gen_dataset(problem_name, problem_index)
    
    

    