import cv2
from circle_workflows import Circle_Workflow, Hugh_circle_Workflow
from ball_finder_machine import Ball_Finder_Machine
from Robot_Vision import RobotVision
from computer_vision import Color_Filter
import numpy as np
rgb = cv2.imread("new.jpg")
# circle=Hugh_circle_Workflow()
# runner=Ball_Finder_Machine(circle,(256,256))
# image=robotvision.get_numpy_image_bgra(rgb)
hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
gray_scale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
print(rgb.shape)

c = 2


def cc(k):
    global c
    c = k


cv2.namedWindow("debug")
cv2.createTrackbar("c", "debug", 1, 255, cc)
while True:
    # re = cv2.adaptiveThreshold(
    #     rgb[:, :, 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, c)

    re = cv2.Canny(hsv[:, :, 1], 50, 200)

    cv2.imshow("debug", re)

    cv2.waitKey(1)

cv2.destroyAllWindows()
# print(runner.run(rgb))
