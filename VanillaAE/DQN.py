######################################################################
# DQN Model Train
##############################################################################
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
from env import *
from collections import deque
from replay_buffer import ReplayMemory, Transition
from  torch.autograd import Variable
import torch
import torch.optim as optim
import random
from model import QNetwork

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

device = torch.device("cuda" if use_cuda else "cpu")
from  torch.autograd import Variable

from replay_buffer import ReplayMemory, Transition

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

#plt.ion()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 128
TAU = 0.005 
gamma = 0.99
LEARNING_RATE = 0.0004
TARGET_UPDATE = 10

num_episodes = 40000
print_every = 1  
hidden_dim = 16
min_eps = 0.01
max_eps_episode = 10



space_dim = 42 # n_spaces
action_dim = 27 # n_actions
print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

threshold = 200
env = Env(space_dim,action_dim,LEARNING_RATE)
print('threshold: ', threshold)
    
def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    ##  if i_epsiode --> max_episode, ret_eps --> min_eps
    ##  if i_epsiode --> 1, ret_eps --> 1  
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    return ret_eps        

def save(directory, filename):
    torch.save(env.q_local.state_dict(), '%s/%s_local.pth' % (directory, filename))
    torch.save(env.q_target.state_dict(), '%s/%s_target.pth' % (directory, filename))

def run_episode(env, eps):
    state = env.reset()
    #done = False
    total_reward = 0
    
    #env.render(1)
    n_done=0
    count=0
    success_count=0
    crash_count=0
    bt_count=0
    over_count=0
    while(1):
        count=count+1
        for i in range(len(env.uavs)):
            if env.uavs[i].done:
                continue
            action = env.get_action(FloatTensor(np.array([state[i]])) , eps)
            
            next_state, reward, uav_done, info= env.step(action.detach(),i)

            total_reward += reward
                        
            # Store the transition in memory
            env.replay_memory.push(
                    (FloatTensor(np.array([state[i]])), 
                    action, # action is already a tensor
                    FloatTensor([reward]), 
                    FloatTensor([next_state]), 
                    FloatTensor([uav_done])))
            """ if reward>0:
                for t in range(2):
                    env.replay_memory.push(
                        (FloatTensor(np.array([state[i]])), 
                        action, # action is already a tensor
                        FloatTensor([reward]), 
                        FloatTensor([next_state]), 
                        FloatTensor([uav_done]))) """
            if info==1:
                success_count=success_count+1
            elif info==2:
                crash_count+=1
            elif info==3: 
                bt_count+=1
            elif info==5: 
                over_count+=1

            if uav_done:
                env.uavs[i].done=True
                n_done=n_done+1
                continue
            state[i] = next_state
        #env.render()
        if count%5==0 and len(env.replay_memory) > BATCH_SIZE:
            #batch = env.replay_memory.sample(BATCH_SIZE) 
            env.learn(gamma,BATCH_SIZE)
        if n_done>=env.n_uav:
            break
        #plt.pause(0.001)
    if success_count>=0.8*env.n_uav and env.level<10:
        env.level=env.level+1
    return total_reward,[success_count,crash_count,bt_count,over_count]
def train():    

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []    
    
    time_start = time.time()

    check_point_Qlocal=torch.load('Qlocal.pth')
    check_point_Qtarget=torch.load('Qtarget.pth')
    env.q_target.load_state_dict(check_point_Qtarget['model'])
    env.q_local.load_state_dict(check_point_Qlocal['model'])
    env.optim.load_state_dict(check_point_Qlocal['optimizer'])
    epoch=check_point_Qlocal['epoch']

    for i_episode in range(num_episodes):
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score,info = run_episode(env, eps)

        scores_deque.append(score)
        scores_array.append(score)
        
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)
            
        if i_episode % print_every == 0 and i_episode > 0:
            print('sum_Episode: {:5} Episode: {:5} Score: {:5}  Avg.Score: {:.2f}, eps-greedy: {:5.2f} Time: {:02}:{:02}:{:02} level:{:5}  num_success:{:2}  num_crash:{:2}  num_none_energy:{:2}  num_overstep:{:2}'.\
                    format(i_episode+epoch,i_episode, score, avg_score, eps, dt//3600, dt%3600//60, dt%60,env.level,info[0],info[1],info[2],info[3]))
        if i_episode %100==0:
            state = {'model': env.q_target.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode+epoch}
            torch.save(state, "Qtarget.pth")
            state = {'model': env.q_local.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode+epoch}
            torch.save(state, "Qlocal.pth")

        if i_episode % TARGET_UPDATE == 0:
            env.q_target.load_state_dict(env.q_local.state_dict()) 
    
    return scores_array, avg_scores_array

  


if __name__ == '__main__':
    scores,avg_scores=train()
    print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))
