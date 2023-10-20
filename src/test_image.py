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


# cv2.namedWindow("debug")
# cv2.createTrackbar("hl", "debug", 0, 179, change_hl)
# cv2.createTrackbar("hh", "debug", 0, 179, change_hh)
# cv2.createTrackbar("sl", "debug", 0, 255, change_sl)
# cv2.createTrackbar("sh", "debug", 0, 255, change_sh)


x1=0
y1=280
x2=100
y2=280

def change_x1(k):
    global x1
    x1=k

def change_x2(k):
    global x2
    x2=k

def change_y1(k):
    global y1
    y1=k

def change_y2(k):
    global y2
    y2=k

# cv2.createTrackbar("x1", "debug", 0, 480, change_x1)
# cv2.createTrackbar("y1", "debug", 0, 540, change_y1)
# cv2.createTrackbar("x2", "debug", 0, 480, change_x2)
# cv2.createTrackbar("y2", "debug", 0, 540, change_y2)

sampled_image=rgb[300:480, :540]


while True:
    high = np.array([hh, sh, 255])
    low = np.array([hl, sl, 0])
    filtered_image = cv2.inRange(h_sv, low, high)
    cv2.imshow("debug", sampled_image)

    cv2.waitKey(1)

cv2.destroyAllWindows()
# print(runner.run(rgb))
