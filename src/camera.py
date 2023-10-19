import cv2
import freenect
from circle_workflows import Circle_Workflow, Hugh_circle_Workflow
from ball_finder_machine import Ball_Finder_Machine
from Robot_Vision import RobotVision
from computer_vision import Color_Filter_HSV, Combined_Filter
import numpy as np

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

class Circle:
    def __init__(self) -> None:
        self.filr = Color_Filter_HSV(Red.hl, Red.hh, Red.sl, Red.sh)
        self.filb = Color_Filter_HSV(Blue.hl, Blue.hh, Blue.sl, Blue.sh)
        self.fil = Combined_Filter([self.filr, self.filb])
        self.cw = Hugh_circle_Workflow(self.fil)

    def get_depth_and_bgr(self):
        depth, timestamp = freenect.sync_get_depth()
        rgb, timestamp = freenect.sync_get_video()
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return depth, bgr
    
    def identify_circles(self):
        bgr=self.get_depth_and_bgr()[1]
        filtered_image = self.fil.get_mask_for_bgra(bgr)
        circle, dilated_image, gray_masked_image = self.cw.find_circles(bgr)
        drawn = self.cw.draw_circles(circle, bgr)
        print(circle)
        cv2.imshow("debug", drawn)
        cv2.waitKey(10000)

