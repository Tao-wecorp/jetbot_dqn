import trt_pose.coco
import trt_pose.models
import torch

import torch2trt
from torch2trt import TRTModule

follow_people_configfiles_path = rospkg.RosPack().get_path('ignisbot_people_follower_pkg')+"/trt_config_files"
PKL_OPTIMIZED_MODEL = "optimised_model.pickle"
pkl_optimized_model_weights_path = os.path.join(follow_people_configfiles_path, PKL_OPTIMIZED_MODEL)
humanPose_file_path = os.path.join(follow_people_configfiles_path, 'human_pose.json')

with open(humanPose_file_path, 'r') as f:
    human_pose = json.load(f)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)