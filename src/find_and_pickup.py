
from ball_finder_machine import Ball_Finder_Machine, Filtered_Value
from KobukiDriver import Kobuki
from simple_pid import PID
from random import choice
import numpy as np


class Find_and_Pickup:
    def __init__(self, ball_finder: Ball_Finder_Machine, kuboki_controller: Kobuki, turn_limit: float, pid: PID):
        self.ball_finder = ball_finder
        self.kuboki_controller = kuboki_controller
        self.limit = turn_limit
        self.pid = pid

    def run_find(self, image, time_step):
        self.ball_finder.run(image)
        deviation = self.ball_finder.deviation_from_center()
        if(self.ball_finder.circle_present is False or deviation is None):
            self.kuboki_controller.move(0, 0, 0.05)
            return

        control = self.pid(-deviation, dt=time_step/1000)
        if(control is None):
            return
        if(abs(control) > self.limit):
            self.kuboki_controller.move(0, 0, control)
            return

        self.kuboki_controller.move(1, 0, 0)


class Super_Machine:
    FINDING = 0
    PUTTING_AND_FINISH = 1
    SEARCHING_TURN_RIGHT = 2
    SEARCHING_TURN_LEFT = 3
    GO_FORWAD_A_BIT = 4

    def __init__(self, find_and_pickup: Find_and_Pickup, super_halfway: Super_Halfway):
        self.find_and_pickup = find_and_pickup
        self.super_halfway = super_halfway
        self.state = self.FINDING

    def run(self, image: np.ndarray, time_step: int):
        print(self.state)
        if(self.state == self.FINDING):
            self.find_and_pickup.run_find(image, time_step)
            if not(self.find_and_pickup.ball_finder.circle_present):
                self.state = choice(
                    [self.SEARCHING_TURN_LEFT, self.SEARCHING_TURN_RIGHT])
            if(self.find_and_pickup.ball_finder.circle_present):
                if (self.super_halfway.collector.ball_present()):
                    self.state = self.PUTTING_AND_FINISH
                    self.find_and_pickup.ball_finder.reset()
            return
        if(self.state == self.PUTTING_AND_FINISH):
            if(self.super_halfway.collector.state != 0 and self.super_halfway.collector.state < 4 and not self.super_halfway.collector.ball_present_center()):
                self.super_halfway.collector.state=0
                self.super_halfway.state=0
                self.state = self.FINDING
                return
            ret = self.super_halfway.super_state_machine('b')
            if(ret):
                self.state = self.SEARCHING_TURN_RIGHT
            return

        if(self.state == self.SEARCHING_TURN_RIGHT):
            self.find_and_pickup.ball_finder.run(image)

            if(self.find_and_pickup.ball_finder.circle_present):
                print("Found a circle")
                print("radius", self.find_and_pickup.ball_finder.current_radius)
                if(self.find_and_pickup.ball_finder.current_radius < 20):
                    self.state = self.GO_FORWAD_A_BIT
                    return
                self.state = self.FINDING
                return
            if(self.super_halfway.navigator.make_angle_theta(90, -1)):
                self.state = self.SEARCHING_TURN_LEFT
            return

        if(self.state == self.SEARCHING_TURN_LEFT):
            self.find_and_pickup.ball_finder.run(image)
            if(self.find_and_pickup.ball_finder.circle_present):
                print("Found a circle")
                print("radius", self.find_and_pickup.ball_finder.current_radius)
                if(self.find_and_pickup.ball_finder.current_radius < 20):
                    self.state = self.GO_FORWAD_A_BIT
                    return
                self.state = self.FINDING
                return
            if(self.super_halfway.navigator.make_angle_theta(270, 1)):
                self.state = self.SEARCHING_TURN_RIGHT
            return
        if(self.state == self.GO_FORWAD_A_BIT):
            self.find_and_pickup.ball_finder.run(image)
            self.super_halfway.collector.kuka.wheel_velocity(1, 0, 0)
            if not(self.find_and_pickup.ball_finder.circle_present):
                d = self.find_and_pickup.ball_finder.deviation_from_center()
                print(f"{d=}")
                if(d < 0):
                    self.state = self.SEARCHING_TURN_LEFT
                    return
                else:
                    self.state = self.SEARCHING_TURN_RIGHT
                # self.state = choice(
                #     [self.SEARCHING_TURN_LEFT, self.SEARCHING_TURN_RIGHT])
            if(self.find_and_pickup.ball_finder.current_radius >= 20):
                self.state = self.FINDING
                return