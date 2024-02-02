######################################################################
# Environment build
##############################################################################
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim
import random
from model import QNetwork
from UAV import *
from  torch.autograd import Variable
from replay_buffer import ReplayMemory, Transition
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")
class building():
    def __init__(self,x,y,l,w,h):
        self.x=x   #center x
        self.y=y   #center y
        self.l=l        #half length
        self.w=w       #half width
        self.h=h    #hight
class sn():
    def __init__(self,x,y,z):
        self.x=x 
        self.y=y 
        self.z=z
class Env(object):
    def __init__(self,n_states,n_actions,LEARNING_RATE):
        #定义规划空间大小
        self.len=100
        self.width=100
        self.h=22
        self.map=np.zeros((self.len,self.width,self.h))
        self.WindField=[1,0]
        self.uavs=[]
        self.bds=[]
        self.target=[]
        self.n_uav=1
        self.v0=40
        self.fig=plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        plt.ion()  #interactive mode on
        self.level=1

        self.q_local = QNetwork(n_states, n_actions, hidden_dim=16).to(device)
        self.q_target = QNetwork(n_states, n_actions, hidden_dim=16).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)
        self.n_states = n_states
        self.n_actions = n_actions

        self.replay_memory = ReplayMemory(10000)

    def get_action(self, state, eps, check_eps=True):
        global steps_done
        sample = random.random()

        if check_eps==False or sample > eps:
            with torch.no_grad():
                return self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
           return torch.tensor([[random.randrange(self.n_actions)]], device=device)
    def learn(self, gamma,BATCH_SIZE):
        
        if len(self.replay_memory.memory) < BATCH_SIZE:
            return
            
        transitions = self.replay_memory.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
                        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)
        

        Q_expected = self.q_local(states).gather(1, actions)

        Q_targets_next = self.q_target(next_states).detach().max(1)[0]

        # Compute the expected Q values
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        self.q_local.train(mode=True)        
        self.optim.zero_grad()
        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))
        # backpropagation of loss to NN        
        loss.backward()
        self.optim.step()
               
        
    def soft_update(self, local_model, target_model, tau):
        """ tau (float): interpolation parameter"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)     
            
    def hard_update(self, local, target):
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(param.data)
    def render(self,flag=0):
        if flag==1:
            z=0
            for ob in self.bds:
                x=ob.x
                y=ob.y
                z=0
                dx=ob.l 
                dy=ob.w 
                dz=ob.h 
                xx = np.linspace(x-dx, x+dx, 2)
                yy = np.linspace(y-dy, y+dy, 2)
                zz = np.linspace(z, z+dz, 2)

                xx2, yy2 = np.meshgrid(xx, yy)

                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz))
            

                yy2, zz2 = np.meshgrid(yy, zz)
                self.ax.plot_surface(np.full_like(yy2, x-dx), yy2, zz2)
                self.ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2)

                xx2, zz2= np.meshgrid(xx, zz)
                self.ax.plot_surface(xx2, np.full_like(yy2, y-dy), zz2)
                self.ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2)
            for sn in self.target:
                self.ax.scatter(sn.x, sn.y, sn.z,c='red')
        
        for uav in self.uavs:
            self.ax.scatter(uav.x, uav.y, uav.z,c='blue')


    def step(self, action,i):
        reward=0.0
        done=False
        #self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=0
        reward,done,info=self.uavs[i].update(action)  #无人机执行行为,info为是否到达目标点
        #self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
        next_state = self.uavs[i].state()
        return next_state,reward,done,info
    def reset(self):
        self.uavs=[]
        self.bds=[]
        self.map=np.zeros((self.len,self.width,self.h))
        self.WindField=[]
        self.WindField.append(np.random.normal(40,5))
        self.WindField.append(2*math.pi*random.random())
        for i in range(random.randint(self.level,2*self.level)):
            self.bds.append(building(random.randint(10,self.len-10),random.randint(10,self.width-10),random.randint(1,10),random.randint(1,10),random.randint(9,13)))
            self.map[self.bds[i].x-self.bds[i].l:self.bds[i].x+self.bds[i].l,self.bds[i].y-self.bds[i].w:self.bds[i].y+self.bds[i].w,0:self.bds[i].h]=1

        x=0
        y=0
        z=0
        while(1):
            x=random.randint(60,90)
            y=random.randint(10,90)
            z=random.randint(3,15)
            if self.map[x,y,z]==0:
                break
        self.target=[sn(x,y,z)]
        self.map[x,y,z]=2
        for i in range(self.n_uav):
            x=0
            y=0
            z=0
            while(1):
                x=random.randint(15,30)
                y=random.randint(10,90)
                z=random.randint(3,7)
                if self.map[x,y,z]==0:
                    break
            self.uavs.append(UAV(x,y,z,self))

        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state
    def reset_test(self):
        self.uavs=[]
        self.bds=[]
        self.map=np.zeros((self.len,self.width,self.h))
        self.WindField=[]
        self.WindField.append(np.random.normal(40,5))
        self.WindField.append(2*math.pi*random.random())
        for i in range(random.randint(self.level,2*self.level)):
            self.bds.append(building(random.randint(10,self.len-10),random.randint(10,self.width-10),random.randint(1,10),random.randint(1,10),random.randint(9,13)))
            self.map[self.bds[i].x-self.bds[i].l:self.bds[i].x+self.bds[i].l,self.bds[i].y-self.bds[i].w:self.bds[i].y+self.bds[i].w,0:self.bds[i].h]=1
        x=0
        y=0
        z=0
        while(1):
            x=random.randint(60,90)
            y=random.randint(10,90)
            z=random.randint(3,15)
            if self.map[x,y,z]==0:
                break
        self.target=[sn(x,y,z)]
        self.map[x,y,z]=2
        self.uavs.append(UAV(20,20,3,self))
        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state

    def reset_test1(self, new_ox, new_oy, new_oz, new_x, new_y, new_z):
        self.uavs = []
        self.bds = []
        self.map = np.zeros((self.len, self.width, self.h))  # 重置障碍物
        self.WindField = []
        self.WindField.append(1)
        self.WindField.append(0)
        l = 20
        w = 10
        self.bds.append(building(40, 10, l, w, 5))
        self.bds.append(building(100, 10, l, w, 5))
        self.bds.append(building(80, 60,l, w, 5))
        self.bds.append(building(10, 80, l, w, 5))

        new_l = 5
        new_w = 5
        self.bds.append(building(new_ox, new_oy, new_l, new_w, new_oz))

        for i in range(len(self.bds)):
            self.map[self.bds[i].x - self.bds[i].l:self.bds[i].x + self.bds[i].l,
            self.bds[i].y - self.bds[i].w:self.bds[i].y + self.bds[i].w, 0:self.bds[i].h] = 1

        x = 90
        y = 90
        z = 5

        self.target = [sn(x, y, z)]
        self.map[x, y, z] = 2
        print(new_x, new_y, new_z)
        self.uavs.append(UAV(new_x, new_y, new_z, self))
        self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state

if __name__ == "__main__":
    env=Env()
  
    env.reset()
    env.render()
    plt.pause(30)

