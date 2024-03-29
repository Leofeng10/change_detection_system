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

    # state = env.reset_test1(args.ox, args.oy, args.oz, args.x, args.y, args.z)
    total_reward = 0
    n_done=0
    count=0

    n_test=1
    n_creash=0


    cmd = ""
    # for l in range(12, 13):
    env.level = 5
    success_count = 0
    fail = 0
    for i in range(n_test):
        state = env.reset()
        env.render(1)
        while(1):
            if env.uavs[0].done:
                break
            action = env.get_action(FloatTensor(np.array([state[0]])) , 0.01)

            next_state, reward, uav_done, info, dx, dy, dz, x, y, z = env.step(action.item(),0)


            # print(dx, dy, dz, x, y, z, uav_done, info)
            cmd += env.convert_to_cmd_simulator(dy, dx, dz)
            cmd += "(" + str(x) + "," +  str(y) + "," + str(z) + ")\n"

            total_reward += reward

            env.render()
            plt.pause(0.01)

            if info==1:
                success_count=success_count+1
                print("success!!!!")
                break
            if uav_done:
                print(dx, dy, dz, x, y, z, uav_done, info)
                print("Fail!!!!")
                fail += 1
                break

            state[0] = next_state
        # print("success:", success_count, "fAIL", fail)

        # with open("VanillaAE/command.txt", 'w') as file:
        #     file.write(cmd)
        env.ax.scatter(env.target[0].x, env.target[0].y, env.target[0].z,c='red')
        plt.pause(100)