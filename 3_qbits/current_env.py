import gym
from gym import spaces
import numpy as np
from numpy import linalg
import scipy as sc
from scipy import linalg
from torch import initial_seed
import hyperparams as hp

class GridWorldEnv(gym.Env): 
    
    def __init__(self,maxSteps=20,stepSize=0.1):
        
        self.maxSteps = maxSteps
        self.stepSize = stepSize
        self.stepNum = 0
        #initial state is equal superposition
        self.initialState = np.array([1,1,1,1,1,1,1,1])/np.sqrt(8)
        #target state is GHZ state
        self.targetState = 1.0/np.sqrt(2)*np.array([1,0,0,0,0,0,0,1])
        self.state = self.initialState
        self.Ham0 = sc.linalg.expm(-1j*stepSize*np.array([[-6,4,4,0,4,0,0,0],[4,0,0,4,0,4,0,0],[4,0,0,4,0,0,4,0],                                                         [0,4,4,2,0,0,0,4],[4,0,0,0,0,4,4,0],[0,4,0,0,4,2,0,4],                                                         [0,0,4,0,4,0,2,4],[0,0,0,4,0,4,4,0]]))
        self.Ham1 = sc.linalg.expm(-1j*stepSize*np.array([[-6,-4,-4,0,-4,0,0,0],[-4,0,0,-4,0,-4,0,0],[-4,0,0,-4,0,0,-4,0],                                                         [0,-4,-4,2,0,0,0,-4],[-4,0,0,0,0,-4,-4,0],[0,-4,0,0,-4,2,0,-4],                                                         [0,0,-4,0,-4,0,2,-4],[0,0,0,-4,0,-4,-4,0]]))
        self.Hams = [self.Ham0,self.Ham1]
        
        #state encoded as a 4-component vector
        self.observation_space = spaces.Box(-1,1,shape=(16,),dtype=np.float64)
        
        # we can apply a positive or negative magnetic field
        self.action_space = spaces.Discrete(2)
        
    def vecTrans(self, state):
        return np.array([state[0].real,state[0].imag,state[1].real,state[1].imag, state[2].real,state[2].imag,                         state[3].real,state[3].imag,state[4].real,state[4].imag,state[5].real,state[5].imag,                         state[6].real,state[6].imag,state[7].real,state[7].imag])
    
    def reset(self, options=None):
        self.state = self.initialState
        self.stepNum = 0
        observation = self.vecTrans(self.state)
        return observation
    
    def step(self, action): #action 0 or 1
        self.state = np.dot(self.Hams[action],self.state)
        self.stepNum += 1
        fidelity = np.linalg.norm(np.dot(np.conj(self.state).T,self.targetState))**2
        if self.step == self.maxSteps or fidelity > hp.targetFidelity:
            reward = fidelity
            terminated = True
        else:
            reward = 0
            terminated = False
        observation = self.vecTrans(self.state)
        info = {}
        return observation, reward, terminated, info




