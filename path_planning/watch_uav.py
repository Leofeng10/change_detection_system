######################################################################
# Verification of UAV track planning model based on DQN
##############################################################################
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
from env import *
import torch
LEARNING_RATE = 0.00033
num_episodes = 80000
space_dim = 42 # n_spaces
action_dim = 27 # n_actions
threshold = 200 
env = Env(space_dim,action_dim,LEARNING_RATE)

if __name__ == '__main__':
    check_point_Qlocal=torch.load('path_planning/Qlocal.pth')
    check_point_Qtarget=torch.load('path_planning/Qtarget.pth')
    env.q_target.load_state_dict(check_point_Qtarget['model'])
    env.q_local.load_state_dict(check_point_Qlocal['model'])
    env.optim.load_state_dict(check_point_Qlocal['optimizer'])
    epoch=check_point_Qlocal['epoch']
    env.level= 8
    state = env.reset_test1(30, 30, 10)
    total_reward = 0
    env.render(1)
    n_done=0
    count=0
 
    n_test=1
    n_creash=0
    for i in range(n_test):
        while(1):
            if env.uavs[0].done:
                #无人机已结束任务，跳过
                break
            action = env.get_action(FloatTensor(np.array([state[0]])) , 0.01)

            next_state, reward, uav_done, info= env.step(action.item(),0)

            total_reward += reward

            print(action)
            env.render()
            plt.pause(0.01)  
            if uav_done:
                break
            if info==1:
                success_count=success_count+1

            state[0] = next_state
        print(env.uavs[0].step)
        env.ax.scatter(env.target[0].x, env.target[0].y, env.target[0].z,c='red')
        plt.pause(100) 



