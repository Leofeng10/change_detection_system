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
from  torch.autograd import Variable
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")
import torch
import torch.nn as nn
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, batch):
        self.memory.append(batch)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class UAV():
    def __init__(self, x, y, z, ev):
        self.x = x
        self.y = y
        self.z = z
        self.target = [ev.target[0].x, ev.target[0].y, ev.target[0].z]
        self.ev = ev
        self.bt = 5000
        self.dir = 0
        self.p_bt = 10
        self.now_bt = 4
        self.cost = 0
        self.detect_r = 5
        self.ob_space = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.nearest_distance = 10
        self.dir_ob = None
        self.p_crash = 0
        self.done = False
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2])
        self.d_origin = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2])
        self.step = 0

    def cal(self, num):
        if num == 0:
            return -1
        elif num == 1:
            return 0
        elif num == 2:
            return 1
        else:
            raise NotImplementedError

    def state(self):
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dz = self.target[2] - self.z
        state_grid = [self.x, self.y, self.z, dx, dy, dz, self.target[0], self.target[1], self.target[2], self.d_origin,
                      self.step, self.distance, self.dir, self.p_crash, self.now_bt, self.cost]
        self.ob_space = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    if self.x + i < 0 or self.x + i >= self.ev.len or self.y + j < 0 or self.y + j >= self.ev.width or self.z + k < 0 or self.z + k >= self.ev.h:
                        self.ob_space.append(1)
                        state_grid.append(1)
                    else:
                        self.ob_space.append(self.ev.map[self.x + i, self.y + j, self.z + k])
                        state_grid.append(self.ev.map[self.x + i, self.y + j, self.z + k])
        return state_grid

    def update(self, action):
        dx, dy, dz = [0, 0, 0]
        temp = action
        b = 3
        wt = 0.005
        wc = 0.01
        we = 0
        c = 0.05
        crash = 0
        Ddistance = 0

        dx = self.cal(temp % 3)
        temp = int(temp / 3)
        dy = self.cal(temp % 3)
        temp = int(temp / 3)
        dz = self.cal(temp)
        # if drone doesn't move, max penalty
        if dx == 0 and dy == 0 and dz == 0:
            return -1000, False, False, 0, 0, 0, self.x, self.y, self.z
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz
        Ddistance = self.distance - (
                    abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2]))
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(self.z - self.target[2])
        self.step += abs(dx) + abs(dy) + abs(dz)

        flag = 1
        if abs(dy) == dy:
            flag = 1
        else:
            flag = -1

        if dx * dx + dy * dy != 0:
            self.dir = math.acos(dx / math.sqrt(dx * dx + dy * dy)) * flag

        r_ob = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                if self.x + i < 0 or self.x + i >= self.ev.len or self.y + j < 0 or self.y + j >= self.ev.width or self.z < 0 or self.z >= self.ev.h:
                    continue
                if self.ev.map[self.x + i, self.y + j, self.z] == 1 and abs(i) + abs(j) < self.nearest_distance:
                    self.nearest_distance = abs(i) + abs(j)
                    flag = 1
                    if abs(j) == -j:
                        flag = -1
                    self.dir_ob = math.acos(i / (i * i + j * j)) * flag
        if self.nearest_distance >= 4 or self.ev.WindField[0] <= self.ev.v0:
            self.p_crash = 0
        else:
            self.p_crash = math.exp(-b * self.nearest_distance * self.ev.v0 * self.ev.v0 / (0.5 * math.pow(
                self.ev.WindField[0] * math.cos(abs(self.ev.WindField[1] - self.dir_ob) - self.ev.v0), 2)))

        r_climb = 0
        r_climb = -wc * (abs(self.z - self.target[2]))
        if self.distance > 1:
            r_target = 2 * (self.d_origin / self.distance) * Ddistance
        else:
            r_target = 2 * (self.d_origin) * Ddistance

        r = r_climb + r_target - crash * self.p_crash

        if self.x <= 0 or self.x >= self.ev.len - 1 or self.y <= 0 or self.y >= self.ev.width - 1 or self.z <= 0 or self.z >= self.ev.h - 1 or \
                self.ev.map[self.x, self.y, self.z] == 1 or random.random() < self.p_crash:
            return r - 200, True, 2, dx, dy, dz, self.x, self.y, self.z
        if self.distance <= 2:
            return r + 200, True, 1, dx, dy, dz, self.x, self.y, self.z
        if self.step >= self.d_origin + 2 * self.ev.h:
            return r - 20, True, 5, dx, dy, dz, self.x, self.y, self.z

        return r, False, 4, dx, dy, dz, self.x, self.y, self.z


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim) -> None:
        super(QNetwork, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)

        return x
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

        l = 20
        w = 10
        self.replay_memory = ReplayMemory(10000)
        self.bds.append(building(40, 10, l, w, 5))
        self.bds.append(building(100, 10, l, w, 5))
        self.bds.append(building(80, 60, l, w, 5))
        self.bds.append(building(10, 80, l, w, 5))

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
        reward,done,info, dx, dy, dz, x, y, z=self.uavs[i].update(action)
        #self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
        next_state = self.uavs[i].state()
        return next_state,reward,done,info, dx, dy, dz, x, y, z

    def reset(self):
        self.uavs=[]
        self.bds=[]
        self.map=np.zeros((self.len,self.width,self.h))
        self.WindField=[]
        self.WindField.append(np.random.normal(40,5))
        self.WindField.append(2*math.pi*random.random())
        for i in range(random.randint(self.level,2*self.level)):
            self.bds.append(building(random.randint(10,self.len-10),random.randint(10,self.width-10),random.randint(1,10),random.randint(1,10),random.randint(9,13)))
            self.map[self.bds[i].x-self.bds[i].l - 5:self.bds[i].x+self.bds[i].l + 5,self.bds[i].y-self.bds[i].w - 5:self.bds[i].y+self.bds[i].w + 5,0:self.bds[i].h]=1

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
        self.uavs = []
        self.bds = []
        self.map = np.zeros((self.len, self.width, self.h))
        self.WindField = []
        self.WindField.append(1)
        self.WindField.append(0)
        self.bds.append(building(35, 40, 20, 5, 5))
        self.bds.append(building(80, 60, 5, 10, 5))
        for i in range(len(self.bds)):
            self.map[self.bds[i].x - self.bds[i].l - 5:self.bds[i].x + self.bds[i].l + 5,
            self.bds[i].y - self.bds[i].w - 5:self.bds[i].y + self.bds[i].w + 5, 0:self.bds[i].h] = 1
        x = 70
        y = 55
        z = 5

        self.target = [sn(x, y, z)]
        self.map[x, y, z] = 2
        self.uavs.append(UAV(42, 21, 6, self))
        self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state

    def reset_test1(self, new_ox, new_oy, new_oz, new_x, new_y, new_z):
        self.uavs = []
        self.bds = []
        self.map = np.zeros((self.len, self.width, self.h))
        self.WindField = []
        self.WindField.append(1)
        self.WindField.append(0)

        new_l = 2
        new_w = 2
        self.bds.append(building(new_ox, new_oy, new_l, new_w, new_oz))
        self.bds.append(building(35, 40, 20, 5, 5))
        self.bds.append(building(80, 60, 5, 10, 5))

        for i in range(len(self.bds)):
            self.map[self.bds[i].x - self.bds[i].l:self.bds[i].x + self.bds[i].l,
            self.bds[i].y - self.bds[i].w:self.bds[i].y + self.bds[i].w, 0:self.bds[i].h] = 1

        x = 70
        y = 55
        z = 6

        self.target = [sn(x, y, z)]
        self.map[x, y, z] = 2
        print(new_x, new_y, new_z)
        self.uavs.append(UAV(new_x, new_y, new_z, self))
        self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state

    def convert_to_cmd(self, dx, dy, dz):
        cmd = ""
        if dx == 1:
            cmd += "forward 1\n"
        elif dx == -1:
            cmd += "ccw 180"
            cmd += "forward 1\n"

        if dy == 1:
            cmd += "ccw 90\n"
            cmd += "forward 1\n"
        elif dy == -1:
            cmd += "cw 90\n"
            cmd += "forward 1\n"

        if dz == 1:
            cmd += "up 1\n"
        elif dz == -1:
            cmd += "down 1\n"

        return cmd

    def convert_to_cmd_simulator(self, dx, dy, dz):
        cmd = ""
        if dx == 1 and dy == 1:
            cmd = "2, 45\n"
        elif dx == -1 and dy == -1:
            cmd = "2, -135\n"
        elif dx == 1 and dy == 0:
            cmd += "1, 0\n"
        elif dx == -1 and dy == 0:
            cmd += "1, 180\n"
        elif dx == 0 and dy == 1:
            cmd += "1, 90\n"
        elif dx == 0 and dy == -1:
            cmd += "1, -90\n"
        elif dx == 1 and dy == -1:
            cmd += "2, -45\n"
        elif dx == -1 and dy == 1:
            cmd += "2, 135\n"

        if dz == 1:
            cmd += "4, 0\n"
        elif dz == -1:
            cmd += "5, 0\n"

        return cmd

if __name__ == "__main__":
    env=Env()

    env.reset()
    env.render()
    plt.pause(30)

#
# ######################################################################
# # Environment build
# # ---------------------------------------------------------------
# # author by younghow
# # email: younghowkg@gmail.com
# # --------------------------------------------------------------
# # env类对城市环境进行三维构建与模拟，利用立方体描述城市建筑，
# # 同时用三维坐标点描述传感器。对环境进行的空间规模、风况、无人机集合、
# # 传感器集合、建筑集合、经验池进行初始化设置，并载入DQN神经网络模型。
# # env类成员函数能实现UAV行为决策、UAV决策经验学习、环境可视化、单时间步推演等功能。
# # ----------------------------------------------------------------
# # The env class constructs and simulates the urban environment in 3D,
# # uses cubes to describe urban buildings, and uses 3D coordinate points
# # to describe sensors. Initialize the environment's spatial scale, wind conditions,
# # UAV collection, sensor collection, building collection, and experience pool,
# # and load the DQN neural network model.
# # The env class member function can implement UAV behavioral decision-making,
# # UAV decision-making experience learning, environment visualization, and single-time-step deduction.
# ##############################################################################
# import time
# import numpy as np
# import random
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import torch
# import torch.optim as optim
# import random
# from model import QNetwork
# from UAV import *
# from torch.autograd import Variable
# from replay_buffer import ReplayMemory, Transition
#
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# device = torch.device("cuda" if use_cuda else "cpu")  # 使用GPU进行训练
#
#
# class building():
#     def __init__(self, x, y, l, w, h):
#         self.x = x  # 建筑中心x坐标
#         self.y = y  # 建筑中心y坐标
#         self.l = l  # 建筑长半值
#         self.w = w  # 建筑宽半值
#         self.h = h  # 建筑高度
#
#
# class sn():
#     def __init__(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.z = z
#
#
# class Env(object):
#     def __init__(self, n_states, n_actions, LEARNING_RATE):
#         # 定义规划空间大小
#         self.len = 100
#         self.width = 100
#         self.h = 22
#         self.map = np.zeros((self.len, self.width, self.h))
#         self.WindField = [30, 0]  # 风场(风速,风向角)
#         # self.action_space=spaces.Discrete(27)  #定义无人机动作空间(0-26),用三进制对动作进行编码 0:-1 1：0 2：+1
#         # self.observation_space=spaces.Box(shape=(self.len,self.width,self.h),dtype=np.uint8)  #定义观测空间（规划空间）,能描述障碍物情况与风向
#         self.uavs = []  # 无人机对象集合
#         self.bds = []  # 建筑集合
#         self.target = []  # 无人机对象
#         self.n_uav = 15  # 训练环境中的无人机个数
#         self.v0 = 40  # 无人机可控风速
#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
#         plt.ion()  # interactive mode on
#         self.level = 1  # 训练难度等级(0-10)
#
#         # 神经网络参数
#         self.q_local = QNetwork(n_states, n_actions, hidden_dim=16).to(device)  # 初始化Q网络
#         self.q_target = QNetwork(n_states, n_actions, hidden_dim=16).to(device)  # 初始化目标Q网络
#         self.mse_loss = torch.nn.MSELoss()  # 损失函数：均方误差
#         self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)  # 设置优化器，使用adam优化器
#         self.n_states = n_states  # 状态空间数目？
#         self.n_actions = n_actions  # 动作集数目
#
#         #  ReplayMemory: trajectory is saved here
#         self.replay_memory = ReplayMemory(10000)  # 初始化经验池
#
#     def get_action(self, state, eps, check_eps=True):
#         """Returns an action  返回行为值
#
#         Args:
#             state : 2-D tensor of shape (n, input_dim)
#             eps (float): eps-greedy for exploration  eps贪心策略的概率
#
#         Returns: int: action index    动作索引
#         """
#         global steps_done
#         sample = random.random()
#
#         if check_eps == False or sample > eps:
#             with torch.no_grad():
#                 return self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)  # 根据Q值选择行为
#         else:
#             ## return LongTensor([[random.randrange(2)]])
#             return torch.tensor([[random.randrange(self.n_actions)]], device=device)  # 随机选取动作
#
#     def learn(self, gamma, BATCH_SIZE):
#         """Prepare minibatch and train them  准备训练
#
#         Args:
#         experiences (List[Transition]): batch of `Transition`
#         gamma (float): Discount rate of Q_target  折扣率
#         """
#
#         if len(self.replay_memory.memory) < BATCH_SIZE:
#             return
#
#         transitions = self.replay_memory.sample(BATCH_SIZE)  # 获取批量经验数据
#
#         batch = Transition(*zip(*transitions))
#
#         states = torch.cat(batch.state)
#         actions = torch.cat(batch.action)
#         rewards = torch.cat(batch.reward)
#         next_states = torch.cat(batch.next_state)
#         dones = torch.cat(batch.done)
#
#         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#         # columns of actions taken. These are the actions which would've been taken
#         # for each batch state according to newtork q_local (current estimate)
#         Q_expected = self.q_local(states).gather(1, actions)  # 获得Q估计值
#
#         Q_targets_next = self.q_target(next_states).detach().max(1)[0]  # 计算Q目标值估计
#
#         # Compute the expected Q values
#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))  # 更新Q目标值
#         # 训练Q网络
#         self.q_local.train(mode=True)
#         self.optim.zero_grad()
#         loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))  # 计算误差
#         # backpropagation of loss to NN
#         loss.backward()
#         self.optim.step()
#
#     def soft_update(self, local_model, target_model, tau):
#         """ tau (float): interpolation parameter"""
#         # 更新Q网络与Q目标网络
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
#
#     def hard_update(self, local, target):
#         for target_param, param in zip(target.parameters(), local.parameters()):
#             target_param.data.copy_(param.data)
#
#     def render(self, flag=0):
#         # 绘制封闭立方体
#         # 参数
#         # x,y,z立方体中心坐标
#         # dx,dy,dz 立方体长宽高半长
#         # fig = plt.figure()
#         if flag == 1:
#             # 第一次渲染，需要渲染建筑
#             z = 0
#             # ax = self.fig.add_subplot(1, 1, 1, projection='3d')
#             for ob in self.bds:
#                 # 绘画出所有建筑
#                 x = ob.x
#                 y = ob.y
#                 z = 0
#                 dx = ob.l
#                 dy = ob.w
#                 dz = ob.h
#                 xx = np.linspace(x - dx, x + dx, 2)
#                 yy = np.linspace(y - dy, y + dy, 2)
#                 zz = np.linspace(z, z + dz, 2)
#
#                 xx2, yy2 = np.meshgrid(xx, yy)
#
#                 self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
#                 self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z + dz))
#
#                 yy2, zz2 = np.meshgrid(yy, zz)
#                 self.ax.plot_surface(np.full_like(yy2, x - dx), yy2, zz2)
#                 self.ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2)
#
#                 xx2, zz2 = np.meshgrid(xx, zz)
#                 self.ax.plot_surface(xx2, np.full_like(yy2, y - dy), zz2)
#                 self.ax.plot_surface(xx2, np.full_like(yy2, y + dy), zz2)
#             for sn in self.target:
#                 # 绘制目标坐标点
#                 self.ax.scatter(sn.x, sn.y, sn.z, c='red')
#
#         for uav in self.uavs:
#             # 绘制无人机坐标点
#             self.ax.scatter(uav.x, uav.y, uav.z, c='blue')
#
#     def step(self, action, i):
#         """环境的主要驱动函数，主逻辑将在该函数中实现。该函数可以按照时间轴，固定时间间隔调用
#
#         参数:
#             action (object): an action provided by the agent
#             i:i号无人机执行更新动作
#
#         返回值:
#             observation (object): agent对环境的观察，在本例中，直接返回环境的所有状态数据
#             reward (float) : 奖励值，agent执行行为后环境反馈
#             done (bool): 该局游戏时候结束，在本例中，只要自己被吃，该局结束
#             info (dict): 函数返回的一些额外信息，可用于调试等
#         """
#         reward = 0.0
#         done = False
#         # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=0
#         reward, done, info = self.uavs[i].update(action)  # 无人机执行行为,info为是否到达目标点
#         # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
#         next_state = self.uavs[i].state()
#         return next_state, reward, done, info
#
#     def reset(self):
#         """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
#         """
#         # 重置画布
#         # plt.close()
#         # self.fig=plt.figure()
#         # self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
#
#         # plt.clf()
#         self.uavs = []
#         self.bds = []
#         self.map = np.zeros((self.len, self.width, self.h))  # 重置障碍物
#         self.WindField = []
#         # 生成随机风力和风向
#         self.WindField.append(np.random.normal(40, 5))
#         self.WindField.append(2 * math.pi * random.random())
#         """ #设置边界障碍物
#         self.map[0:self.len,0:self.width,0]=1
#         self.map[0:self.len,0:self.width,self.h-1]=1
#         self.map[0:self.len,0:self.width,0:self.h]=1
#         self.map[self.len-1,0:self.width,0:self.h]=1
#         self.map[0:self.len,0,0:self.h]=1
#         self.map[0:self.len,self.width-1,0:self.h]=1 """
#         # 随机生成建筑物
#         for i in range(random.randint(self.level, 2 * self.level)):
#             self.bds.append(
#                 building(random.randint(10, self.len - 10), random.randint(10, self.width - 10), random.randint(1, 10),
#                          random.randint(1, 10), random.randint(9, 13)))
#             self.map[self.bds[i].x - self.bds[i].l:self.bds[i].x + self.bds[i].l,
#             self.bds[i].y - self.bds[i].w:self.bds[i].y + self.bds[i].w, 0:self.bds[i].h] = 1
#
#         # 随机生成目标点位置
#         x = 0
#         y = 0
#         z = 0
#         while (1):
#             x = random.randint(60, 90)
#             y = random.randint(10, 90)
#             z = random.randint(3, 15)
#             if self.map[x, y, z] == 0:
#                 # 随机生成在无障碍区域
#                 break
#         self.target = [sn(x, y, z)]
#         self.map[x, y, z] = 2
#
#         # 随机生成无人机位置
#         for i in range(self.n_uav):
#             x = 0
#             y = 0
#             z = 0
#             while (1):
#                 x = random.randint(15, 30)
#                 y = random.randint(10, 90)
#                 z = random.randint(3, 7)
#                 if self.map[x, y, z] == 0:
#                     # 随机生成在无障碍区域
#                     break
#             self.uavs.append(UAV(x, y, z, self))  # 重新生成无人机位置
#             # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
#
#         # 更新无人机状态
#         self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])
#
#         return self.state
#
#     def reset_test(self):
#         # 环境重组测试
#         self.uavs = []
#         self.bds = []
#         self.map = np.zeros((self.len, self.width, self.h))  # 重置障碍物
#         self.WindField = []
#         # 生成随机风力和风向
#         self.WindField.append(np.random.normal(40, 5))
#         self.WindField.append(2 * math.pi * random.random())
#         # 随机生成建筑物
#         for i in range(random.randint(self.level, 2 * self.level)):
#             self.bds.append(
#                 building(random.randint(10, self.len - 10), random.randint(10, self.width - 10), random.randint(1, 10),
#                          random.randint(1, 10), random.randint(9, 13)))
#             self.map[self.bds[i].x - self.bds[i].l:self.bds[i].x + self.bds[i].l,
#             self.bds[i].y - self.bds[i].w:self.bds[i].y + self.bds[i].w, 0:self.bds[i].h] = 1
#
#         # 随机生成目标点位置
#         x = 0
#         y = 0
#         z = 0
#         while (1):
#             x = random.randint(60, 90)
#             y = random.randint(10, 90)
#             z = random.randint(3, 15)
#             if self.map[x, y, z] == 0:
#                 # 随机生成在无障碍区域
#                 break
#         self.target = [sn(x, y, z)]
#         self.map[x, y, z] = 2
#
#         # 生成无人机位置
#         self.uavs.append(UAV(20, 20, 3, self))  # 重新生成无人机位置
#         # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
#
#         # 更新无人机状态
#         self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])
#
#         return self.state
