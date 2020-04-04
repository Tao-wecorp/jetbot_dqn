#! /usr/bin/env python3

import numpy as np
from math import *
import matplotlib.pyplot as mp

ACTIONMAT = np.array([-1, 0, 1])
reward = 0.0

class QLearning():
    def __init__(self):
        self.state = -10.0  # [angle]
        self.position = [0.0, 0.0]  # [x, y]
        self.goal = 0.0  # [angle]
        self.full_angel = 0.0  # [angle]
        self.reward = 0.0  # point

    def setState(self, state):
        self.state = state
    
    def setGoal(self, position):
        self.goal = degrees(atan((position[0]-320)/(480-position[1])))
        self.full_angel = 2 * degrees(atan(320/(480-position[1])))
       
    def calcReward(self):
        self.reward  = (1-abs(self.goal-self.state)/self.full_angel)*100

    def yaw(self, position):
        self.goal = degrees(atan((position[0]-320)/(480-position[1])))
        self.full_angel = 2 * degrees(atan(320/(480-position[1])))
        self.reward = (1-abs(self.goal-self.state)/self.full_angel)*100
        return self.state, self.goal, self.reward