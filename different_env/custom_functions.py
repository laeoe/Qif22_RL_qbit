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

folder_name = "/oneQbit_2actions/"
cwd = os.getcwd()
results_dir = cwd + folder_name + "training_results/"


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

def blochsphere(self, comp):
        # makes bloch sphere using four component representation, having also the initial and target state on it
        fig = plt.figure()
        bloch = qutip.Bloch(fig=fig)


        # make list of [resulting vector, target, intial]
        vector = np.array([comp[0]+1j*comp[1],comp[2]+1j*comp[3]])
        vectors = [vector, self.targetState, self.initialState]

        # make each vector in the bloch sphere representation
        for dummyvector in vectors:
            bloch.add_vectors([np.matmul(np.conj(dummyvector),np.matmul(paulimat,dummyvector)).real for paulimat in self.smats])
        bloch.render()

        plt.show()