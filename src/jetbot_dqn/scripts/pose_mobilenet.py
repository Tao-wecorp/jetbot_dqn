#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError

import os
import rospy
import rospkg
rospack = rospkg.RosPack()

from sensor_msgs.msg import Image

openpose_folder = os.path.join(rospack.get_path("jetbot_dqn"), "scripts/openpose/")
net = cv2.dnn.readNetFromTensorflow(openpose_folder + "graph_opt.pb")

nPoints = 18
class Pose(object):
    def __init__(self):
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw",Image,self.camera_callback)
    
    def camera_callback(self,data):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        frameWidth = cv_image.shape[1]
        frameHeight = cv_image.shape[0]

        inHeight = 368
        inWidth = int((inHeight/frameHeight)*frameWidth)
        net.setInput(cv2.dnn.blobFromImage(cv_image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        output = net.forward()
        output = output[:, :nPoints, :, :]

        H = output.shape[2]
        W = output.shape[3]

        points = []
        threshold = 0.1
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if (prob > threshold):
                cv2.circle(cv_image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(cv_image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else :
                points.append(None)

        cv2.imshow("Keypoints",cv_image)
        cv2.waitKey(1)

def main():
    rospy.init_node('pose_node', anonymous=True)
    pose = Pose()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()