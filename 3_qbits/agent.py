import gym
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
import pickle
import current_env

import custom_functions
import hyperparams as hp

env_ = hp.current_env.GridWorldEnv()
states_ = list()
#folder_name = "/oneQbit_2actions/"
folder_name = hp.folder_name
results_dir = hp.results_dir

#cwd = os.getcwd()
#results_dir = cwd + folder_name + "training_results/"


data_list = list()

# Hyperparameters
n_train_processes = hp.n_train_processes  #number of parallel workers
learning_rate = hp.learning_rate
update_interval = hp.update_interval     #interval after which the learning is updated, currently 5
gamma = hp.gamma
max_train_steps = hp.max_train_steps
PRINT_INTERVAL = hp.PRINT_INTERVAL

depth_first_layer = hp.depth_firt_layer
depth_second_layer = hp.depth_second_layer
depth_action_space = hp.depth_action_space



class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(depth_first_layer, depth_second_layer)
        self.fc_pi = nn.Linear(depth_second_layer, depth_action_space)
        self.fc_v = nn.Linear(depth_second_layer, 1)

    def pi(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = env_
    env.seed(worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def test(step_idx, model):
    env = env_
    score = 0.0
    done = False
    num_test = 10
    list_s_ = list()
    list_s = list()
    list_g = list()

    list_a_ = list()
    list_a = list()
    list_r = list()

    for _ in range(num_test):
        s = env.reset()
        while not done:
            #s_ini = env.vecTrans()
            #list_s_.append(s_ini)
            prob = model.pi(torch.from_numpy(s).float(), softmax_dim=0)
            a = Categorical(prob).sample().numpy()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r
            list_s_.append(s)
            list_a_.append(a)
        done = False
        
        list_g.append([list_a_, list_s_, r])
        list_a_ = list()
        list_s_ = list()
    data_list.append([step_idx, list_g])
    env.close()

    #states_.append(np.mean(s_, axis = 0)) #computes the mean of final state after running the model 10 times until it finalized

    return(score/num_test)

    

def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

if __name__ == '__main__':
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    s = envs.reset()

    ### changes to output best model
    max_reward = 0
    global_r_list = list()
    #global_r_list.append("N train ")



    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        for _ in range(update_interval):
            prob = model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().numpy()
            s_prime, r, done, info = envs.step(a)

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

        s_final = torch.from_numpy(s_prime).float()
        v_final = model.v(s_final).detach().clone().numpy()
        td_target = compute_target(v_final, r_lst, mask_lst)

        td_target_vec = td_target.reshape(-1)
        s_vec = torch.tensor(s_lst).float().reshape(-1, depth_first_layer)  # 4 == Dimension of state
        a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
        advantage = td_target_vec - model.v(s_vec).reshape(-1)

        pi = model.pi(s_vec, softmax_dim=1)
        pi_a = pi.gather(1, a_vec).reshape(-1)
        loss = -(torch.log(pi_a) * advantage.detach()).mean() +\
            F.smooth_l1_loss(model.v(s_vec).reshape(-1), td_target_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        # changes
        if step_idx % PRINT_INTERVAL == 0:
            current_reward = test(step_idx, model)
            global_r_list.append(current_reward)
            print(f"Step # :{step_idx}, avg score :", current_reward)

            if current_reward > max_reward:
                max_reward = current_reward
                torch.save(pi, results_dir + "trained_Agent.pth")


    envs.close()
    #print("last state vector", s_vec)
    print("done, max reward was:",max_reward)
    #np.save(results_dir + "r_list", np.array(global_r_list))
    #np.save(results_dir + "states_list", np.array(states_))
    
    
    custom_functions.data_save(data_list, results_dir + "data_list")