# coding:UTF-8
# @Time: 2024/8/22 12:16
# @Author: Lulu Cao
# @File: config.py
# @Software: PyCharm


import torch
import numpy as np

# Maximum length of expressions
MAX_LENGTH = 40

# ---------- LEARNING CONFIG ----------
# Number of trial expressions to try at each epoch
BATCH_SIZE = int(1e2)

default_op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"]


# ---------- PRIORS CONFIG 先验----------
priors_config  = [
                #("UniformArityPrior", None),
                # LENGTH RELATED
                ("HardLengthPrior"  , {"min_length": 2, "max_length": MAX_LENGTH, }),
                ("SoftLengthPrior"  , {"length_loc": 3, "scale": 3, }),
                # RELATIONSHIPS RELATED
                ("NoUselessInversePrior"  , None),
                ("NestedTrigonometryPrior", {"max_nesting" : 1}),
                ("BanVariablePrior",{"variables" : ["x2","x3",],}),
                 ]

# ---------- FREE CONSTANT OPTIMIZATION CONFIG 常数优化----------
free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 5,
                        'tol'     : 1e-8,
                        'lbfgs_func_args' : {
                            'max_iter'       : 15,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
}

# Learning config
learning_config = {
    # Batch related
    'batch_size'       : BATCH_SIZE,
    'max_time_step'    : MAX_LENGTH,
}

# Learning config

# Function returning the torch optimizer given a model
GET_OPTIMIZER = lambda model : torch.optim.Adam(
                                    model.parameters(),
                                    lr=0.0025,
                                                )

learning_config = {
    # Batch related
    'batch_size'       : BATCH_SIZE,
    'max_time_step'    : MAX_LENGTH,
    'n_epochs'         : int(1),
    # Loss related
    'gamma_decay'      : 0.7,
    'entropy_weight'   : 0.005,
    # Reward related
    'risk_factor'      : 0.05,
    #'rewards_computer' : physo.physym.reward.make_RewardsComputer (**reward_config),
    # Optimizer
    'get_optimizer'    : GET_OPTIMIZER,
    'observe_units'    : True,
}

# ---------- RNN CELL CONFIG ----------
cell_config = {
    "hidden_size" : 128,
    "n_layers"    : 3,
    "is_lobotomized" : False,
}

# ---------- RUN CONFIG ----------
config0 = {
    "free_const_opti_args" : free_const_opti_args,
    "priors_config"        : priors_config,
    "learning_config"      : learning_config,
    "cell_config"          : cell_config,
}
