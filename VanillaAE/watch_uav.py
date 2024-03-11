######################################################################
# Verification of UAV track planning model based on DQN
##############################################################################
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
import argparse
from env import *
import torch
LEARNING_RATE = 0.00033
num_episodes = 80000
space_dim = 42 # n_spaces
action_dim = 27 # n_actions
threshold = 200 
env = Env(space_dim,action_dim,LEARNING_RATE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=int)
    parser.add_argument('y', type=int)
    parser.add_argument('z', type=int)

    parser.add_argument('ox', type=int)
    parser.add_argument('oy', type=int)
    parser.add_argument('oz', type=int)

    args = parser.parse_args()
    print((args.x, args.y, args.z))
    check_point_Qlocal=torch.load('./VanillaAE/Qlocal.pth')
    check_point_Qtarget=torch.load('VanillaAE/Qtarget.pth')
    env.q_target.load_state_dict(check_point_Qtarget['model'])
    env.q_local.load_state_dict(check_point_Qlocal['model'])
    env.optim.load_state_dict(check_point_Qlocal['optimizer'])
    epoch=check_point_Qlocal['epoch']
    env.level= 8
    state = env.reset_test()

    # state = env.reset_test1(args.ox, args.oy, args.oz, args.x, args.y, args.z)
    total_reward = 0
    env.render(1)
    n_done=0
    count=0
 
    n_test=1
    n_creash=0

    cmd = "command\n"
    for i in range(n_test):
        while(1):
            if env.uavs[0].done:
                break
            action = env.get_action(FloatTensor(np.array([state[0]])) , 0.01)

            next_state, reward, uav_done, info, dx, dy, dz, x, y, z = env.step(action.item(),0)


            print(dx, dy, dz, x, y, z, uav_done)
            cmd += env.convert_to_cmd(dx, dy, dz)
            cmd += "(" + str(x) + "," +  str(y) + "," + str(z) + ")\n"

            total_reward += reward

            env.render()
            plt.pause(0.01)  
            if uav_done:
                break
            if info==1:
                success_count=success_count+1

            state[0] = next_state

        # with open("path_planning/command.txt", 'w') as file:
        #     file.write(cmd)
        env.ax.scatter(env.target[0].x, env.target[0].y, env.target[0].z,c='red')
        plt.pause(100)





