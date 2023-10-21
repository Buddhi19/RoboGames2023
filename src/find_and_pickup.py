
from ball_finder_machine import Ball_Finder_Machine, Filtered_Value
# from KobukiDriver import Kobuki
from simple_pid import PID
from random import choice
import numpy as np
from circle_workflows import Circle_Workflow, Hugh_circle_Workflow
from computer_vision import Color_Filter_HSV, Combined_Filter
from sqaure_detection import Square
import cv2
class Kobuki:
    def move(self, a, b, c):
        print(a,b,c)
        
    def inertial_sensor_data(self):
        return {"angle":90}
        

class Find_and_Pickup:
    def __init__(self, ball_finder: Ball_Finder_Machine, kuboki_controller: Kobuki, turn_limit: float):
        self.ball_finder = ball_finder
        self.kuboki_controller = kuboki_controller
        self.limit = turn_limit

    def run_find(self, image):
        self.ball_finder.run(image)
        deviation = self.ball_finder.deviation_from_center()
        if(self.ball_finder.circle_present is False or deviation is None):
            self.kuboki_controller.move(100, 0, 0)
            return

        control = deviation*400 
        if(control is None):
            return
        
        CONST = 100
        if(abs(deviation) > self.limit):
            self.kuboki_controller.move(control+CONST, -control+CONST, 0)
            return

        self.kuboki_controller.move(CONST*2, CONST*2, 0)

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
    
def check_ball(bgr,filter):
    sampled_image=bgr[360:480, :540]
    filtered_image = filter.get_mask_for_bgra(sampled_image)
    check_val=np.mean(filtered_image)
    print(check_val)
    # cv2.imshow("1",filtered_image)
    # cv2.waitKey(1)
    return check_val>=5

def check_dropped(bgr,filter):
    sampled_image = bgr[300:480, :540]
    filtered_image = filter.get_mask_for_bgra(sampled_image)
    check_val = np.mean(filtered_image)
    print(check_val)
    return check_val >= 100

def turn_180(kobuki:Kobuki):
    sensor_data = kobuki.inertial_sensor_data()
    angle = sensor_data["angle"]
    error = 180 - angle
    if abs(error)<0.1:
        return True
    kobuki.move(0, error, 0)

    

class Super_Machine:
    FINDING = 0
    COLLECTED = 1
    GOING_BACK = 2
    SEARCHING_TURN_RIGHT = 3
    SEARCHING_TURN_LEFT = 4
    GO_FORWARD_A_BIT = 5

    def __init__(self) -> None:
        self.state = self.COLLECTED
        self.filr = Color_Filter_HSV(Red.hl, Red.hh, Red.sl, Red.sh)
        self.filb = Color_Filter_HSV(Blue.hl, Blue.hh, Blue.sl, Blue.sh)
        self.fil = Combined_Filter([self.filr, self.filb])
        self.kuboki = Kobuki()
        self.circle=Hugh_circle_Workflow(self.fil)
        self.ball_finder=Ball_Finder_Machine(self.circle,(480,540))
        self.find_and_pickup = Find_and_Pickup(self.ball_finder,self.kuboki,0.1)
        self.find_squares = Square()
        self.ball_finder_for_square = Ball_Finder_Machine(self.find_squares,(480,540))
        self.find_and_drop_squares = Find_and_Pickup(self.ball_finder_for_square,self.kuboki,0.1)
        self.Time = 0

    def run(self,image):
        """
        state machine
        """
        if self.state == self.FINDING:
            print("FINDING")
            self.find_and_pickup.run_find(image)
            if check_ball(image,self.fil):
                self.state = self.COLLECTED
            
        if self.state == self.COLLECTED:
            print("COLLECTING")
            sampled_image = image[:300,:540]
            # cv2.imshow("1",sampled_image)
            # cv2.waitKey(1)
            self.find_squares.find_circles(sampled_image)
            if check_dropped(image,self.fil):
                self.state = self.GOING_BACK
                
        if self.state ==  self.GOING_BACK:
            print("GOING BACK")
            self.Time+=1
            if self.Time >= 2000:
                self.state = self.SEARCHING_TURN_LEFT
                self.Time = 0
                
            self.kuboki.move(-100,-100,0)
            
        if self.state == self.SEARCHING_TURN_LEFT:
            print("TURNING LEFT")
            if turn_180(self.kuboki):
                self.state = self.FINDING