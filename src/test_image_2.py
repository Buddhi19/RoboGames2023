import cv2
from circle_workflows import Circle_Workflow, Hugh_circle_Workflow
from ball_finder_machine import Ball_Finder_Machine
from Robot_Vision import RobotVision
from computer_vision import Color_Filter_HSV, Combined_Filter
import numpy as np
rgb = cv2.imread("17.png")


class Red:
    hl = 0
    hh = 63
    sl = 88
    sh = 208


class Blue:
    hl = 112
    hh = 132
    sl = 54
    sh = 255


filr = Color_Filter_HSV(Red.hl, Red.hh, Red.sl, Red.sh)
filb = Color_Filter_HSV(Blue.hl, Blue.hh, Blue.sl, Blue.sh)
fil = Combined_Filter([filr, filb])
cw = Hugh_circle_Workflow(fil)

while True:
    filtered_image = fil.get_mask_for_bgra(rgb)
    circle = cw.find_circles(rgb)
    drawn = cw.draw_circles(circle, rgb)
    print(circle)
    cv2.imshow("debug", drawn)

    cv2.waitKey(1)

cv2.destroyAllWindows()
# print(runner.run(rgb))
