from __future__ import print_function
import argparse
import os

import torch.backends.cudnn as cudnn
from model_cbam import UC
from utils import get_dataloader, print_and_write_log, set_random_seed
import matplotlib.pyplot as pp
from env import *

import tello as tello
import ast


def update(frame):
    x.append(len(x))
    y.append(frame)
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.001)


x = []
y = []
fig, ax = pp.subplots()
ax.set_ylim(0, 200)
ax.set_xlim(0, 1000)
line, = ax.plot(x, y)



def main():

    # drone
    drone = tello.Tello('', 8880)
    drone.begin()








if __name__ == '__main__':
    main()
