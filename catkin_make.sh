#!/bin/bash

catkin_make -DPYTHON_EXECUTABLE:FILEPATH=~/.virtualenvs/py3venv/bin/python3.6

# wstool init
# wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
# wstool up
# rosdep install --from-paths src --ignore-src -y -r

# catkin_make --cmake-args \
#             -DCMAKE_BUILD_TYPE=Release \
#             -DPYTHON_EXECUTABLE=/usr/bin/python3 \
#             -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
#             -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so