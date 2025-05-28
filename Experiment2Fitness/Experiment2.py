# coding:utf-8
import pickle
import matplotlib.pyplot as plt
import glob
import numpy as np

def load_fitness_history(file_path):
    """读取 .pkl 文件并返回适应度历史数据"""
    with open(file_path, 'rb') as f:
        fitness_history = pickle.load(f)
    return fitness_history

def whole_plot(file_path,save_path):
    fig = plt.figure(figsize=(10, 4))
    generation = np.linspace(1, 10, 10)
    for i in range(1,6):
        fitness_history = load_fitness_history(file_path+str(i)+'.pkl')
        plt.plot(generation,fitness_history[i], label="heat"+str(i),linewidth=2)  # 显示文件名作为图例  
    plt.xlabel('Generation')
    plt.ylabel('Minimum Fitness')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def subplot(file_path,save_path):

    # 创建图形
    fig, axes = plt.subplots(1, 5, figsize=(10*5, 4))
    axes = axes.flatten()  # 将 axes 转换为一维数组以便于索引
    #file_path=r"E:\code\Python\NetGP\NewGPSR\Experiment2Fitness\SPGPSR_heat"
    # 循环读取每个文件并绘制曲线
    generation = np.linspace(1, 10, 10)
    for i in range(1,6):
        fitness_history = load_fitness_history(file_path+str(i)+'.pkl')
        axes[i-1].plot(generation,fitness_history[i], label="heat"+str(i))  # 显示文件名作为图例
        #axes[i-1].set_title(f'{i} heat')
        axes[i-1].set_xlabel('Generation')
        axes[i-1].set_ylabel('Minimum Fitness')
        axes[i-1].legend()
        axes[i-1].grid()
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# file_path=r"E:\code\Python\NetGP\NewGPSR\Experiment2Fitness\SPGPSR_heat"
# for i in range(1,6):
#     fitness_history = load_fitness_history(file_path+str(i)+'.pkl')
#     plt.plot(fitness_history[i], label=file_path.split('/')[-1])  # 显示文件名作为图例

if __name__ == '__main__':
    path=r"E:\code\Python\NetGP\NewGPSR\Experiment2Fitness\SPGPSR_heat"
    save_path=r"E:\code\Python\NetGP\NewGPSR\Experiment2Fitness\SPGPSR_heat.png"
    whole_plot(path,save_path) 