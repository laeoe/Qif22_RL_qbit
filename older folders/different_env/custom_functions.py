import gym
from gym import spaces
import numpy as np
from numpy import linalg
import scipy as sc
from scipy import linalg
import matplotlib.pyplot as plt
import qutip
import generalized.hyperparams as hp
import pickle
import os


results_dir = hp.results_dir



def data_save(lists, filename):
    """Takes list of lists and saves it into filename"""
    outfile = open(filename, 'wb')
    pickle.dump(lists, outfile)
    outfile.close()
    
def data_load(filename):
    infile = open(filename, 'rb')
    lists = pickle.load(infile)
    infile.close()
    return lists

def blochsphere(comp):
        s1 = np.array([[0,1],[1,0]])
        s2 = np.array([[0,-1j],[1j,0]])
        s3 = np.array([[1,0],[0,-1]])
        smats = [s1,s2,s3]

        # makes bloch sphere using four component representation, having also the initial and target state on it
        fig = plt.figure()
        bloch = qutip.Bloch(fig=fig)


        # make list of [resulting vector, target, intial]
        vector = np.array([comp[0]+1j*comp[1],comp[2]+1j*comp[3]])
        vectors = [vector, hp.targetState, hp.initialState]

        # make each vector in the bloch sphere representation
        for dummyvector in vectors:
            bloch.add_vectors([np.matmul(np.conj(dummyvector),np.matmul(paulimat,dummyvector)).real for paulimat in smats])
        bloch.render()

        plt.savefig(results_dir + 'Bloch_final.pdf')
        plt.show()




def plot_rewards(data):
    """r is a list of rewards"""
    r_list = []
    r_value = []
    r_error = []
    for i in range(len(data)):
        r_list.append([])
        for j in range(len(data[i][1])):
            r_list[-1].append(data[i][1][j][2])
        r_value.append(np.average(r_list[-1]))
        r_error.append(np.std(r_list[-1]))
    series = np.arange(len(r_value)) * 500
    plt.plot(series, r_value)
    r_value = np.array(r_value)
    r_error = np.array(r_error)
    plt.xlabel('Number of series')
    plt.ylabel('Reward')
    plt.title('Reward as a function of the number of series')
    plt.fill_between(series, r_value - r_error, r_value + r_error, color='blue', alpha=0.2)
    plt.savefig(results_dir + 'Errors.pdf')
    plt.close()
    

def plot_steps(data):
    step_list = []
    step_value = []
    step_error = []
    for i in range(len(data)):
        step_list.append([])
        for j in range(len(data[i][1])):
            step_list[-1].append(len(data[i][1][j][0]))
        step_value.append(np.average(step_list[-1]))
        step_error.append(np.std(step_list[-1]))
    series = np.arange(len(step_value)) * 500
    plt.plot(series, step_value)
    r_value = np.array(step_value)
    r_error = np.array(step_error)
    plt.xlabel('Number of series')
    plt.ylabel('Reward')
    plt.title('Reward as a function of the number of series')
    plt.fill_between(series, r_value - r_error, r_value + r_error, color='blue', alpha=0.2)
    plt.savefig(results_dir + 'Steps.pdf')
    plt.close()

def plot_bangbang(data, i, j):
    bb = []
    for k in range(len(data[i][1][j][0])):
        bb.append(data[i][1][j][0][k])
    time = np.arange(len(bb))
    plt.step(time, bb)
    plt.xlabel('Time')
    plt.ylabel('h-field')
    plt.title('Magnetic field strength over time')
    plt.savefig(results_dir + 'Bangbang.pdf')
    plt.close()

def plot_bangbang_continuous(data, i, j):
    bb = []
    for k in range(len(data[i][1][j][0])):
        bb.append(data[i][1][j][0][k])
    time = np.arange(len(bb))
    plt.step(time, bb)
    plt.xlabel('Time')
    plt.ylabel('h-field')
    plt.title('Magnetic field strength over time')
    plt.savefig(results_dir + 'Bangbang2.pdf')
    plt.close()

# data_list_1: 20k steps, both steps and fidelity bounded,same as _3, but 100k
# data_list_2: 100k steps, only  
# data_list_3: steps h field, 10 parts
# data_list_4: steps h field, 50 parts
# data_list_5: steps h field, 50 parts, different initial conditons


# data = data_load(results_dir + "data_list")

# plot_rewards(data)
# plot_bangbang(data, -1, 8)
# plot_steps(data)

