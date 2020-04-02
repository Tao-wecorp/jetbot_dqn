#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
from copy import deepcopy
import os
import rospy
from sensor_msgs.msg import Image
import time

from helpers.openpose import OpenPose
openpose = OpenPose()

class Pose(object):
    def __init__(self):
        rospy.init_node('pose_node', anonymous=True)
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw",Image,self.camera_callback)
        self.bridge_object = CvBridge()
        self.frame = None

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.frame is not None:
                start_time = time.time()
                frame = deepcopy(self.frame)
                # detect hip
                points = openpose.detect(frame)
                x, y = points[11]
                # show hip point
                frame = cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.imshow("", frame)
                cv2.waitKey(1)
                # check excution time
                print("%s seconds" % (time.time() - start_time))
            rate.sleep()
            
    
    def camera_callback(self,data):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame = cv_image

def main():
    try:
        Pose()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()