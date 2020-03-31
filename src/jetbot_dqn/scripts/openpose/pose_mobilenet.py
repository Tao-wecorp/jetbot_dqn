#!/usr/bin/env python3

import cv2
import timeit

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

start = timeit.default_timer()

cv_image = cv2.imread("single.jpg")
frameWidth = cv_image.shape[1]
frameHeight = cv_image.shape[0]

inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)


net.setInput(cv2.dnn.blobFromImage(cv_image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
output = net.forward()
output = output[:, :19, :, :]

H = output.shape[2]
W = output.shape[3]

points = []
threshold = 0.1
for i in range(len(BODY_PARTS)):
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

stop = timeit.default_timer()
print('Time: ', stop - start)  
cv2.imshow("Keypoints",cv_image)
cv2.waitKey(0)