import cv2
from circle_workflows import Circle_Workflow,Hugh_circle_Workflow
from ball_finder_machine import Ball_Finder_Machine
from Robot_Vision import RobotVision
from computer_vision import Color_Filter

rgb=cv2.imread("new.jpg")
# circle=Hugh_circle_Workflow()
# runner=Ball_Finder_Machine(circle,(256,256))
# image=robotvision.get_numpy_image_bgra(rgb)
h_sv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
print(rgb.shape)

color_filter=Color_Filter(0xDB6A64,20,20)

filtered_image=color_filter.get_mask_for_hsv(h_sv)

cv2.imshow("1",h_sv[:,:,0])
cv2.imshow("2",h_sv[:,:,1])
cv2.waitKey(0)

# print(runner.run(rgb))