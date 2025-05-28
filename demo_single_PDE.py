# coding:utf-8
# solve one PDE

# External packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
import random
import copy
import time
import pickle

# Internal code import
from physym import batch as Batch
from physym import reward
from learn import rnn
from learn import loss


# Seed
# seed = 0
# np.random.seed(seed)
# torch.manual_seed(seed)


def single_point_crossover(parent1, parent2,length):
    # 随机选择一个交叉点
    point = random.randint(1, length - 1)
    # 生成子代
    child1 = np.concatenate((parent1[:point] , parent2[point:]))
    child2 = np.concatenate((parent2[:point] ,parent1[point:]))
    return child1, child2




def create_model(X,y,run_config):
    def batch_reseter():
        return  Batch.Batch(library_args=run_config["library_config"],
                            priors_config=run_config["priors_config"],
                            batch_size=run_config["learning_config"]["batch_size"],
                            max_time_step=run_config["learning_config"]["max_time_step"],
                            free_const_opti_args=run_config["free_const_opti_args"],
                            X=X,
                            y_target=y,
                            )
    batch = batch_reseter()

    def cell_reseter ():
        input_size  = batch.obs_size
        output_size = batch.n_choices
        cell = rnn.Cell (input_size  = input_size,
                         output_size = output_size,
                         **run_config["cell_config"],
                        )

        return cell

    cell = cell_reseter()
    optimizer = run_config["learning_config"]["get_optimizer"](cell)
    return cell, optimizer
def generated_population(X,y,run_config, evaluate_function):
    generation = 10
    n_keep = int(0.1 * config.BATCH_SIZE)
    cxpb = 0.6
    mutpb = 0.6
    model,optimizer = create_model(X,y,run_config)

    nextgen = None


    fitness_generation = []
    for g in range(generation):
        #sum_len = 0
        fitness_temp = 0
        fitness_count = 0
        # -----------------新种群---------------------
        batch = Batch.Batch(library_args=run_config["library_config"],
                            priors_config=run_config["priors_config"],
                            batch_size=run_config["learning_config"]["batch_size"],
                            max_time_step=run_config["learning_config"]["max_time_step"],
                            free_const_opti_args=run_config["free_const_opti_args"],
                            X=X,
                            y_target=y,
                            )
        batch_size = batch.batch_size

        # Initial RNN cell input
        states = model.get_zeros_initial_state(batch_size)
        # Optimizer reset
        optimizer.zero_grad()


        # Candidates
        logits = []
        actions = []



        for i in range(config.MAX_LENGTH):
            # ------------ OBSERVATIONS ------------
            # (embedding output)
            observations = torch.tensor(batch.get_obs().astype(np.float32),
                                        requires_grad=False, )  # (batch_size, obs_size)

            # ------------ MODEL ------------

            # Giving up-to-date observations
            output, states = model(input_tensor=observations,
                                   # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)
                                   states=states)

            # Getting raw prob distribution for action n°i
            outlogit = output

            # ------------ PRIOR -----------
            # (embedding output)
            prior_array = batch.prior().astype(np.float32)  # (batch_size, output_size)
            # 0 protection so there is always something to sample
            epsilon = 0  # 1e-14 #1e0*np.finfo(np.float32).eps
            prior_array[prior_array == 0] = epsilon

            # To log
            prior = torch.tensor(prior_array, requires_grad=False)  # (batch_size, output_size)
            logprior = torch.log(prior)  # (batch_size, output_size)

            # ------------ SAMPLING ------------

            logit = outlogit + logprior  # (batch_size, output_size)
            action = torch.multinomial(torch.exp(logit),  # (batch_size,)
                                       num_samples=1)[:, 0]

            # ------------ ACTION ------------

            # Saving action n°i
            logits.append(logit)
            actions.append(action)

            # Informing embedding of new action
            # (embedding input)
            batch.programs.append(action.detach().cpu().numpy())

        # -------------------------------------------------
        # ------------------ CANDIDATES  ------------------
        # -------------------------------------------------

        # Keeping prob distribution history for backpropagation
        logits = torch.stack(logits, dim=0)  # (max_time_step, batch_size, n_choices, )
        actions = torch.stack(actions, dim=0)  # (max_time_step, batch_size,)

        # Programs as numpy array for black box reward computation
        actions_array = actions.detach().cpu().numpy()

        #sum_len += batch.programs.n_lengths.sum()
        R = reward.RewardsComputer(programs=batch.programs,
                               X=X,
                               y_target=y,
                               evaluate_function=evaluate_function,
                               free_const_opti_args=run_config["free_const_opti_args"],
                               )
        

        R = np.array(R)
        R = np.nan_to_num(R)
        fitness_temp+=(1/R-1).sum()
        fitness_count+=len(R)

        keep = R.argsort()[::-1][0:n_keep].copy()

        # ----------------- Train batch : black box part (NUMPY) -----------------
        # Elite candidates
        actions_array_train = copy.deepcopy(actions_array[:, keep])

        # Elite candidates as one-hot target probs
        ideal_probs_array_train = np.eye(batch.n_choices)[actions_array_train]
        R_train = torch.tensor(R[keep], requires_grad=False)  # (n_keep,)

        # Elite candidates pred logprobs
        logits_train = logits[:, keep]
        # Lengths of programs
        lengths = batch.programs.n_lengths[keep]

        if nextgen is not None and g % 2:
            ideal_probs_array_train = np.eye(batch.n_choices)[nextgen]
            R_train = R_train_next
            lengths = lengths_next
        # Elite candidates rewards

        R_lim = R_train.min()

        # Elite candidates as one-hot in torch
        # (non-differentiable tensors)
        ideal_probs_train = torch.tensor(  # (max_time_step, n_keep, n_choices,)
            ideal_probs_array_train.astype(np.float32),
            requires_grad=False, )

        baseline = R_lim

        # Loss
        loss_val = loss.loss_func(logits_train=logits_train,
                                  ideal_probs_train=ideal_probs_train,
                                  R_train=R_train,
                                  baseline=baseline,
                                  lengths=lengths,
                                  gamma_decay=run_config["learning_config"]["gamma_decay"],
                                  entropy_weight=run_config["learning_config"]["entropy_weight"], )


        # BACKPROPAGATION
        # -------------------------------------------------
        if model.is_lobotomized:
            pass
        else:
            loss_val.backward()
            optimizer.step()


        # -----------------交叉、变异算子---------------------
        offspring = copy.deepcopy(actions_array[:,keep])
        keep_length = batch.programs.n_lengths

        if nextgen is not None:
            offspring = np.concatenate((offspring,copy.deepcopy(nextgen)),axis=1)

        new_offspring = []
        # 对选出的个体进行交叉和变异
        for child1_i, child2_i in zip(range(0,offspring.shape[1],2),range(1,offspring.shape[1],2)):
            if random.random() < cxpb:
                child1,child2 = offspring[:, child1_i].T, offspring[:, child2_i].T
                templength = min(keep_length[child1_i],keep_length[child2_i])
                temp_child1,temp_child2 = single_point_crossover(child1,child2,templength)
                new_offspring.append(temp_child1.copy())
                new_offspring.append(temp_child2.copy())

        split_token = len(run_config['library_config']["args_make_tokens"]["op_names"])
        for i in range(len(offspring[0])):
            if random.random() < mutpb:
                mutant=offspring[:,i].copy()
                # 随机选择mutant的一个点，变异成另外一个
                mutation_point = random.randint(0, keep_length[i]-1)
                if mutant[mutation_point]<split_token:
                    mutant[mutation_point] = random.randint(0,split_token-1)
                else:
                    mutant[mutation_point] = random.randint(split_token, batch.library.n_choices-1)
                new_offspring.append(mutant)

        # -----------------修正交叉、变异算子---------------------
        new_offspring = np.array(new_offspring)
        #mask_need_action = np.full(shape=(new_offspring.shape), fill_value=True, dtype=bool)
        batch_new = Batch.Batch(library_args=run_config["library_config"],
                            priors_config=run_config["priors_config"],
                            batch_size=len(new_offspring),
                            max_time_step=run_config["learning_config"]["max_time_step"],
                            free_const_opti_args=run_config["free_const_opti_args"],
                            X=X,
                            y_target=y,
                            )
        # Candidates
        logits_new = []
        actions_new = []

        for i in range(config.MAX_LENGTH):
            # ------------ PRIOR -----------
            # (embedding output)
            prior_array = batch_new.prior().astype(np.float32)  # (batch_size, output_size)
            action2 = new_offspring[:, i]
            prior_action = [prior_array[idx][act] for idx, act in enumerate(action2)]
            # prior_action变为bool

            #mask_need_action = np.logical_and(prior_action, action2)


            # 0 protection so there is always something to sample
            epsilon = 0  # 1e-14 #1e0*np.finfo(np.float32).eps
            prior_array[prior_array == 0] = epsilon
            # To log
            prior = torch.tensor(prior_array, requires_grad=False)  # (batch_size, output_size)
            logprior = torch.log(prior)  # (batch_size, output_size)
            action1 = torch.multinomial(torch.exp(logprior),  # (batch_size,)
                                       num_samples=1)[:, 0]




            # ------------ SAMPLING ------------

            action = np.where(prior_action, action1, action2)

            # ------------ ACTION ------------

            # Saving action n°i
            logits_new.append(logit)
            actions_new.append(action)

            # Informing embedding of new action
            # (embedding input)
            batch_new.programs.append(action)


        # -----------------合并、选择下一代---------------------
        if nextgen is not None:
            actions_new = np.concatenate((copy.deepcopy(actions_new),copy.deepcopy(nextgen)),axis=1)
        actions_new = torch.tensor(actions_new)
        actions_new = torch.cat((copy.deepcopy(actions), copy.deepcopy(actions_new)), dim=1)
        actions_new = actions_new.detach().cpu().numpy()
        batch_combine = Batch.Batch(library_args=run_config["library_config"],
                                priors_config=run_config["priors_config"],
                                batch_size=actions_new.shape[1],
                                max_time_step=run_config["learning_config"]["max_time_step"],
                                free_const_opti_args=run_config["free_const_opti_args"],
                                X=X,
                                y_target=y,
                                )

        for i in range(config.MAX_LENGTH):
            action = actions_new[i,:]
            batch_combine.programs.append(action)

        R = reward.RewardsComputer(programs=batch_combine.programs,
                               X=X,
                               y_target=y,
                               evaluate_function=evaluate_function,
                               free_const_opti_args=run_config["free_const_opti_args"],
                               )
        R = np.array(R)
        R = np.nan_to_num(R)
        fitness_temp+=(1/R-1).sum()
        fitness_count+=len(R)

        keep = R.argsort()[::-1][0:n_keep].copy()
        nextgen = copy.deepcopy(actions_new[:, keep])
        R_train_next = torch.tensor(R[keep], requires_grad=False)  # (n_keep,)
        lengths_next = batch_combine.programs.n_lengths[keep]
        #sum_len += batch_combine.programs.n_lengths.sum()



        # -------------------------


        hall_of_fame = [batch_combine.programs.get_prog(keep[0])]

        expr_str = hall_of_fame[0].get_infix_pretty(do_simplify=True)
        print(expr_str,R[keep][0])
        fitness_generation.append(1/R[keep][0]-1)
        if R[keep][0]>0.999:
            return 
    with open('NetGP_heat5.pkl', 'wb') as f:
        pickle.dump({5:fitness_generation}, f)  # 保存
        




    pass

if __name__ == '__main__':

    
    x1 = np.random.uniform(0, 1, 10)
    t = np.random.uniform(0, 2, 10)
    #t = np.zeros(10)

    

    """
    Advection
    """
    # from Advection.Dataset import *
    # import Advection.evaluate as eva
    # y = advection2_1D_exact_solution(x1,t)
    # X = np.stack((x1,t), axis=0)
    # X_names = ['x1', 't']
    # op_names = ["mul", "add", "sub", "sin"]
    # evaluate_function = eva.advection_torch_1d
    

    
    """
    Poisson 
    """
    # from Poisson.PoissonDataset import *
    # import Poisson.evaluate as eva
    # y = poission1_1D_exact_solution(x1)
    # X = np.stack((x1,), axis=0)
    # X_names = ['x1', ]
    # op_names = ["mul", "add", "sub", "sin"]
    # evaluate_function = eva.poisson2_torch_1d

    # """
    # Heat
    # """
    from Heat.Dataset import *
    import Heat.evaluate as eva
    y = heat1_1D_exact_solution(x1,t)
    X = np.stack((x1,t), axis=0)
    X_names = ['x1', 't']
    op_names = ["mul", "add", "sub", "exp", "div"]
    print("heat1")
    #op_names = ["add", "sub","mul","exp","sin"]
    evaluate_function = eva.heat1_torch_1d

    """
    Wave
    """
    # from Wave.Dataset import *
    # import Wave.evaluate as eva
    # y = wave2_1D_exact_solution(x1,t)
    # X = np.stack((x1,t), axis=0)
    # X_names = ['x1', 't']
    # op_names = ["mul", "add", "sub", "sin"]
    # evaluate_function = eva.wave2_torch_1d

    """
    Diffusion-reaction
    """
    # from DiffusionReaction.Dataset import *
    # import DiffusionReaction.evaluate as eva
    # y = diffusion_reaction1_1D_exact_solution(x1,t)
    # X = np.stack((x1,t), axis=0)
    # X_names = ['x1', 't']
    # op_names = ["mul", "add", "sub", "sin","exp"]
    # evaluate_function = eva.diffusion_reaction1_torch_1d



    # --- DATA ---
    X = np.array(X)
    y = np.array(y)
    

    # Converting dataset to torch and sending to device
    X = torch.tensor(X)  # .to(DEVICE)
    y = torch.tensor(y)  # .to(DEVICE)

    
    n_dim, data_size = X.shape
    args_make_tokens = {
        # operations
        "op_names": op_names,
        # input variables
        "input_var_ids": {X_names[i]: i for i in range(n_dim)},
        # free constants
        "free_constants"       : {"c",},
    }

    library_config = {"args_make_tokens": args_make_tokens, }
    run_config = config.config0
    run_config.update({"library_config": library_config})
    generated_population(X,y,run_config=run_config,evaluate_function=evaluate_function)
