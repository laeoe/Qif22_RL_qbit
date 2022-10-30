import gym
from gym import spaces
import numpy as np
from numpy import linalg
import scipy as sc
from scipy import linalg
import matplotlib.pyplot as plt
import qutip
import os


folder_name = "/oneQbit_2actions/"
cwd = os.getcwd()
results_dir = cwd + folder_name + "training_results/"

#Agent Hyperparams
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_steps = 100000
PRINT_INTERVAL = update_interval * 100

depth_firt_layer = 4
depth_second_layer = 256
depth_action_space = 2

#Environment hyperparams
maxSteps = 20
stepSize = 0.1
initialState = np.array([np.sqrt((5+np.sqrt(5))/10),np.sqrt(2/(5+np.sqrt(5)))])
targetState = np.array([-(1+np.sqrt(5))/(np.sqrt(2*(5+np.sqrt(5)))), np.sqrt(2.0/(5+np.sqrt(5)))])
targetFidelity = 1


