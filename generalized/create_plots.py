import gym
from gym import spaces
import numpy as np
from numpy import linalg
import scipy as sc
from scipy import linalg
import matplotlib.pyplot as plt
import qutip
import hyperparams as hp
import pickle
import os
import custom_functions as cf 

def plot_all():
    results_dir = hp.results_dir
    data = cf.data_load(results_dir + "data_list")

    #plot blochsphere final state
    f_state = data[-1][1][0][1][-1]
    #print(f_state)
    cf.blochsphere(f_state)

    cf.plot_rewards(data)
    cf.plot_bangbang_continuous(data, -1, 8)
    cf.plot_steps(data)


