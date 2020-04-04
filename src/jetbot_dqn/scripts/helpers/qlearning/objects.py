#! /usr/bin/env python3

import numpy as np
from math import *
import matplotlib.pyplot as mp

ACTIONMAT = np.array([-1, 0, 1])
reward = 0.0

class QLearning():
    def __init__(self):
        self.state = 0.0  # [angle]
        self.position = [0.0, 0.0]  # [x, y]
        self.goal = 0.0  # [angle]
        self.angle2goal = 0.0  # [angle]
        self.reward = 0.0  # point

    def yaw(self, point):
        goal = degrees(atan((point[0]-320)/(480-point[1])))
        return goal