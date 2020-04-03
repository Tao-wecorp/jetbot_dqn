#!/bin/bash

source ~/.virtualenvs/py3venv/bin/activate
# chmod +x src/jetbot_dqn/scripts/*.py
source devel/setup.bash
rosrun jetbot_dqn pose_mobilenet.py
