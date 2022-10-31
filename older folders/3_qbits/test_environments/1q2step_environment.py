# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:14:09 2022

@author: Filip
"""

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
        self.N_field = 10
        self.field = np.ndarray(self.N_field + 1)
        for i in range(self.N_field + 1):
            self.field[i] = self.h[0] + i* (self.h[1] - self.h[0]) / self.N_field
        self.Ham0 = sc.linalg.expm(-1j*stepSize*np.array([[-1,self.h[0]],[self.h[0],1]]))
        self.Ham1 = sc.linalg.expm(-1j*stepSize*np.array([[-1,self.h[1]],[self.h[1],1]]))
        self.Hams = [self.Ham0,self.Ham1]
        self.hamiltonians = [sc.linalg.expm(-1j*self.stepSize*np.array([[-1,i],[i,1]]) / self.N_field) for i in self.field]
        self.previous = np.nan
        
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
    
    def step(self, action):
      """Just replace the first line in the original code"""
      print('**********')
      if self.previous is np.nan:
          self.previous = 1 if action == 1 else -1
          
      if self.previous > 0 and action == 0:
          for i in range(self.N_field):
              self.state = np.matmul(self.hamiltonians[self.N_field - i],self.state)
              print(self.field[self.N_field - i - 1])
      elif self.previous < 0 and action == 1:
          for i in range(self.N_field):
              self.state = np.matmul(self.hamiltonians[i+1],self.state)
              print(self.field[i+1])
      elif self.previous > 0 and action == 1:
          for i in range(self.N_field):
              self.state = np.matmul(self.hamiltonians[-1],self.state)
      else:
          for i in range(self.N_field):
              self.state = np.matmul(self.hamiltonians[0],self.state)
      self.previous = 1 if action == 1 else -1
      
      self.fidelity = np.matmul(self.state,self.targetState)

      if self.fidelity > 0.99:
        reward = self.fidelity
        terminated = True
      else:
        reward = 0
        terminated = False
      print(reward)

      observation = self.vecTrans(self.state)
      info = {}

      return observation, reward, terminated, info
                                 


if __name__ == '__main__':
    print('env ran')
    prepare = GridWorldEnv()

    prepare.reset()
    prepare.step(0)