import KobukiDriver as kobuki
import time
import cv2
import freenect
from circle_workflows import Circle_Workflow, Hugh_circle_Workflow
from ball_finder_machine import Ball_Finder_Machine
from Robot_Vision import RobotVision
from computer_vision import Color_Filter_HSV, Combined_Filter
from sqaure_detection import Square
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

def get_depth_and_bgr():
    depth, timestamp = freenect.sync_get_depth()
    rgb, timestamp = freenect.sync_get_video()
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return depth, bgr

def check_ball(bgr,filter):
    sampled_image=bgr[300:480, :540]
    filtered_image = filter.get_mask_for_bgra(bgr)
    return filtered_image

def main():
    my_kobuki = kobuki.Kobuki()
    filr = Color_Filter_HSV(Red.hl, Red.hh, Red.sl, Red.sh)
    filb = Color_Filter_HSV(Blue.hl, Blue.hh, Blue.sl, Blue.sh)
    fil = Combined_Filter([filr, filb])


    cw = Hugh_circle_Workflow(fil)
    sq = Square()
    i=0
    j=0
    while True:
        bgr=get_depth_and_bgr()[1]
        filtered_image = fil.get_mask_for_bgra(bgr)
        circle, dilated_image, gray_masked_image = cw.find_circles(bgr)
        drawn = cw.draw_circles(circle, bgr)
        cv2.imshow("debug", drawn)
        # cv2.waitKey(1)

        key = input("Enter command: ")
        if key == "w":
            # Move forward
            my_kobuki.move(200, 200, 0)
        elif key == "s":
            # Move backward
            my_kobuki.move(0, 0, 0)
        elif key == "a":
            # Turn left
            my_kobuki.move(0, -200, 0)
        elif key == "d":
            # Turn right
            my_kobuki.move(-200, 200, 0)
        elif key == "x":
            # Stop
            my_kobuki.move(0, 0, 0)
        elif key == "1":
            # Play sound
            my_kobuki.play_button_sound()
        elif key == "2":
            # LED Control
            my_kobuki.set_led1_green_colour()
            time.sleep(1)
            my_kobuki.set_led2_red_colour()
            time.sleep(1)
            my_kobuki.clr_led1()
            my_kobuki.clr_led2()

        elif key=="c":
            cv2.imwrite(f"image{i}.png",bgr)
            cv2.imwrite(f"image{i}_detected.png",drawn)
            i+=1

        elif key=="m":
            drw=sq.find_squares(bgr)
            cv2.imwrite(f"image_squre_detected{j}.png",drw)
            cv2.imwrite(f"image_squre{j}.png",bgr)
            j+=1
        elif key=="o":
            cv2.imwrite(f"{i}.png",bgr)
            i+=1
        elif key == "q":
            # Quit
            break

        # Print sensor data
        # print(my_kobuki.encoder_data())
        print(my_kobuki.inertial_sensor_data())


if __name__ == "__main__":
    main()
