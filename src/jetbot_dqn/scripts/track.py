#!/usr/bin/env python3

import os
import sys

# ROS imports
import rospy
import rospkg
from sensor_msgs.msg import Image

# trt_pose related imports
import json
import trt_pose.coco
import trt_pose.models
import torch

import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import numpy
from openpose_ros_msgs.msg import PersonDetection, BodypartDetection

follow_people_configfiles_path = rospkg.RosPack().get_path('ignisbot_people_follower_pkg')+"/trt_config_files"
PKL_OPTIMIZED_MODEL = "optimised_model.pickle"
pkl_optimized_model_weights_path = os.path.join(follow_people_configfiles_path, PKL_OPTIMIZED_MODEL)
humanPose_file_path = os.path.join(follow_people_configfiles_path, 'human_pose.json')

print(pkl_optimized_model_weights_path)
print(humanPose_file_path)

with open(humanPose_file_path, 'r') as f:
    human_pose = json.load(f)
    print(human_pose)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
print(topology)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
print(num_links, num_parts)

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model_weights_path = os.path.join(follow_people_configfiles_path, MODEL_WEIGHTS)

HEIGHT = WIDTH = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = model
