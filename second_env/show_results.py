import numpy as np
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
results_dir = cwd + "/second_env/training_results/run1/"


rewards = np.load(results_dir + 'r_list.npy')

def plot_rewards(r):
    """
    Plots the evolution of rewards
    Example: r_list = [[[1,2,3], 'reward_1'], [[2,3,2,1], 'a = 10']]
    """

    factor = 500
    if type(rewards[0]) == np.float64:
        series_length= np.arange(len(r)) * 500
        plt.plot(series_length, r)
    else:
        for p, lab in r:
            series_length = np.arange(len(p)) * 500
            plt.plot(series_length, p, label = lab)
    plt.title('Reward as a function of numer of series')
    plt.xlabel('Number of series')
    plt.ylabel('Reward')
    #plt.legend()

plot_rewards(rewards)