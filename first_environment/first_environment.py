import gym
from gym import spaces
import numpy as np
import scipy.linalg as sci
import stable_baselines3.common.env_checker as sb
import matplotlib.pyplot as plt
import qutip

class StatePreparation(gym.Env):

  def __init__(self, steps=20, deltat=0.1, h=[-4,4], threshold=0.99999999):
    self.steps = steps
    self.stepcount = 0
    self.deltat = deltat
    self.field = h
    self.threshold=threshold
    s1 = np.array([[0,1],[1,0]])
    s2 = np.array([[0,-1j],[1j,0]])
    s3 = np.array([[1,0],[0,-1]])
    self.smats = [s1,s2,s3]

    # Define initial and target states as eigenvalues of Hamiltonian with field strength /2
    self.states = np.linalg.eigh(np.array([[-1,self.field[0]/2],[self.field[0]/2,1]]))[1]
    self.initial = self.states[:,0]
      # only need conjugate for computing fidelity
    self.target = np.conj(self.states[:,1])
    
    # define hamiltonian of time evolution with matrix exponentiation
    self.hamiltonians = [sci.expm(-1j*self.deltat*np.array([[-1,i],[i,1]])) for i in self.field]

    # formal definition for frameowrk
    self.observation_space = spaces.Box(-1,1,shape=(4,), dtype=np.float64)

    # two options for action
    self.action_space= spaces.Discrete(2)

  # define this to work with 4 real numbers
  def state_to_obs(self):
    vec = np.array([self.state[0].real,self.state[0].imag,self.state[1].real,self.state[1].imag])
    return vec

  def reset(self):
    self.state = self.initial
    observation = self.state_to_obs()

    return observation

  def step(self, action):
    self.state = np.matmul(self.hamiltonians[action],self.state)
    fidelity = np.abs(np.matmul(self.state,self.target))

    if fidelity > self.threshold:
      reward = fidelity
      terminated = True
    else:
      reward = 0
      terminated = False

    observation = self.state_to_obs()
    info = {}

    return observation, reward, terminated, info
    
  def blochsphere(self):
    fig = plt.figure()
    bloch = qutip.Bloch(fig=fig)
    

    # compute bloch vector
    vector = [np.matmul(np.conj(self.initial),np.matmul(paulimat,self.initial)).real for paulimat in self.smats]
    print(vector)
    bloch.add_vectors(vector)
    vector = [np.matmul(np.conj(self.target),np.matmul(paulimat,self.target)).real for paulimat in self.smats]
    bloch.add_vectors(vector)
    bloch.render()

    plt.show()



# prepare = StatePreparation()

# prepare.reset()
# prepare.step(0)
# prepare.blochsphere()
