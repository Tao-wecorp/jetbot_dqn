# Jetbot DQN model in ROS

## To-do
1. Add Jetbot Gazebo package;
2. Add DQN;

## Env
1. Run ./env.sh
2. Clone jetbot_gazebo

## Run
    git clone https://github.com/kimbring2/jetbot_gazebo.git
    git rm -f --cached src/jetbot_gazebo
    git submodule add https://github.com/kimbring2/jetbot_gazebo.git src/jetbot_gazebo
    git submodule init
    git submodule update
