import gym
from gym import spaces
import numpy as np
from numpy import linalg
import scipy as sc
from scipy import linalg

class GridWorldEnv(gym.Env):
    
    def __init__(self,maxSteps=20,stepSize=0.1):
        
        self.maxSteps = maxSteps
        self.stepSize = stepSize
        self.stepNum = 0
        self.initialState = np.array([np.sqrt((5+np.sqrt(5))/10),np.sqrt(2/(5+np.sqrt(5)))])
        self.targetState = np.array([-(1+np.sqrt(5))/(np.sqrt(2*(5+np.sqrt(5)))), np.sqrt(2.0/(5+np.sqrt(5)))])
        self.state = self.initialState
        self.h = [-4,4]
        self.Ham0 = sc.linalg.expm(-1j*stepSize*np.array([[-1,self.h[0]],[self.h[0],1]]))
        self.Ham1 = sc.linalg.expm(-1j*stepSize*np.array([[-1,self.h[1]],[self.h[1],1]]))
        self.Hams = [self.Ham0,self.Ham1]
        
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
    
    def step(self, action): #action 0 or 1
        self.state = np.dot(self.Hams[action],self.state)
        self.stepNum += 1
        fidelity = np.linalg.norm(np.dot(np.conj(self.state).T,self.targetState))**2
        if self.step == self.maxSteps or np.abs(fidelity) > 0.99:
            reward = fidelity
            terminated = True
        else:
            reward = 0
            terminated = False
        observation = self.vecTrans(self.state)
        info = {}
        return observation, reward, terminated, info
                                 


if __name__ == '__main__':
    print('env ran')
