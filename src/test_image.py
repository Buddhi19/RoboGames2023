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
h_sv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
print(rgb.shape)

hl = 0
hh = 179
sl = 0
sh = 255


def change_hl(k):
    global hl
    hl = k


def change_hh(k):
    global hh
    hh = k


def change_sl(k):
    global sl
    sl = k


def change_sh(k):
    global sh
    sh = k


cv2.namedWindow("debug")
cv2.createTrackbar("hl", "debug", 0, 179, change_hl)
cv2.createTrackbar("hh", "debug", 0, 179, change_hh)
cv2.createTrackbar("sl", "debug", 0, 255, change_sl)
cv2.createTrackbar("sh", "debug", 0, 255, change_sh)

while True:
    high = np.array([hh, sh, 255])
    low = np.array([hl, sl, 0])
    filtered_image = cv2.inRange(h_sv, low, high)
    cv2.imshow("debug", filtered_image)

    cv2.waitKey(1)

cv2.destroyAllWindows()
# print(runner.run(rgb))
