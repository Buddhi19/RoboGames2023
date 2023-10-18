import cv2
from circle_workflows import Circle_Workflow
from ball_finder_machine import Ball_Finder_Machine
from Robot_Vision import RobotVision

rgb=cv2.imread("new.jpg")
robotvision=RobotVision()
circle=Circle_Workflow()
runner=Ball_Finder_Machine(circle,(256,256))
image=robotvision.get_numpy_image_bgra(rgb)
cv2.imshow("",rgb)
cv2.waitKey(1000)
while True:
    runner.run(image)
