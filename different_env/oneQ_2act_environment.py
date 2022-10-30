import gym
from gym import spaces
import numpy as np
from numpy import linalg
import scipy as sc
from scipy import linalg
import matplotlib.pyplot as plt
import qutip
import hyperparams as hp

#hyperparams 



class GridWorldEnv(gym.Env):
    
    def __init__(self,maxSteps=20,stepSize=0.1):
        
        self.maxSteps = maxSteps
        self.stepSize = stepSize
        self.stepNum = 0
        self.initialState = hp.initialState
        self.targetState = hp.targetState
        self.state = self.initialState

        self.h = [-4,4]
        self.N_field = 10
        self.field = np.ndarray(self.N_field + 1)
        for i in range(self.N_field + 1):
            self.field[i] = self.h[0] + i* (self.h[1] - self.h[0]) / self.N_field

        self.Ham0 = sc.linalg.expm(-1j*stepSize*np.array([[-1,self.h[0]],[self.h[0],1]]))
        self.Ham1 = sc.linalg.expm(-1j*stepSize*np.array([[-1,self.h[1]],[self.h[1],1]]))
        self.Hams = [self.Ham0,self.Ham1]

        self.hamiltonians = [sc.linalg.expm(-1j*self.stepSize*np.array([[-1,i],[i,1]]) / self.N_field) for i in self.field]
        self.previous = np.nan
        
        s1 = np.array([[0,1],[1,0]])
        s2 = np.array([[0,-1j],[1j,0]])
        s3 = np.array([[1,0],[0,-1]])
        self.smats = [s1,s2,s3]
        
        #state encoded as a 4-component vector
        self.observation_space = spaces.Box(-1,1,shape=(4,),dtype=np.float64)
        
        # we can apply a positive or negative magnetic field
        self.action_space = spaces.Discrete(2)
        
    def vecTrans(self, state):
        return np.array([state[0].real,state[0].imag,state[1].real,state[1].imag])
    
    def reset(self, options=None):
        self.state = self.initialState
        observation = self.vecTrans(self.state)
        self.stepNum = 0 #comment
        return observation
    
    # def step(self, action, measure = False): #action 0 or 1
    #     self.state = np.dot(self.Hams[action],self.state)
    #     self.stepNum += 1
    #     fidelity = np.linalg.norm(np.dot(np.conj(self.state).T,self.targetState))**2
    #     if self.stepNum == self.maxSteps or fidelity > hp.targetFidelity:
    #         reward = fidelity
    #         terminated = True
    #     else:
    #         reward = 0
    #         terminated = False
    #     observation = self.vecTrans(self.state)
    #     info = {}
    #     return observation, reward, terminated, info

    def step(self, action):
        """Just replace the first line in the original code"""
        if self.previous is np.nan:
            self.previous = 1 if action == 1 else -1
            
        if self.previous > 0 and action == 0:
            for i in range(self.N_field):
                self.state = np.matmul(self.hamiltonians[self.N_field - i],self.state)
        elif self.previous < 0 and action == 1:
            for i in range(self.N_field):
                self.state = np.matmul(self.hamiltonians[i+1],self.state)
        elif self.previous > 0 and action == 1:
            for i in range(self.N_field):
                self.state = np.matmul(self.hamiltonians[-1],self.state)
        else:
            for i in range(self.N_field):
                self.state = np.matmul(self.hamiltonians[0],self.state)
        self.previous = 1 if action == 1 else -1
        
        self.fidelity = np.abs(np.matmul(self.state,self.targetState))

        if self.stepNum == self.maxSteps or self.fidelity > 0.99:
            reward = self.fidelity
            terminated = True
        else:
            reward = 0
            terminated = False

        self.stepNum += 1

        observation = self.vecTrans(self.state)
        info = {}

        return observation, reward, terminated, info

#thing = GridWorldEnv()

#thing.blochsphere([-0.52204711,0.53612235,0.52888725,-0.4003972 ])
