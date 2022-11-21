#!/usr/bin/env python3

from enum import Enum

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# Proportional coefficient for linear velocity
KP_LIN = 0.5
# Proportional coefficient for angular velocity
KP_ANG = 0.01
# Angular velocity for turning corners
TURN_VEL = 180
# Conversion from distance to angle
DIST_TO_ANG = 90
# Max linear velocity
LIN_MAX = 0.26
# Max angular velocity
ANG_MAX = 1.82

class State(Enum):
    INIT = 0
    FORWARD = 1
    TURN = 2
    STABILIZE = 3

class Ranges:


class CornerHeuristic(object):
    def __init__(self):
        self.state = State.INIT
        rospy.init_node('corner_heuristic')
        self.cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.scan = rospy.Subscriber("/scan", LaserScan, self.check_state)
        self.bot_vel = Twist()
        self.corner_side = 0 # -1 for corner on left, 1 for corner on right

        self.state = State.FORWARD

    def detect_ranges(self, data: LaserScan):
        for i, d in enumerate(data.ranges):
            

    def move_forward(self, data: LaserScan):
        new_twist = Twist()
        new_twist.linear.x = LIN_MAX

    def turn(self, data: LaserScan):
        closest_distance = np.amin(data.ranges)
        closest_angle = np.argmin(data.ranges)
        self.bot_vel.linear.x = KP_LIN * max(closest_distance, THRESHOLD)
        
        # if closest wall is in front
        if (closest_angle < 30 or closest_angle > 330):
            self.bot_vel.linear.x = 0
            # if previous wall is on the right
            if self.wall_side == 1:
                self.bot_vel.angular.z = KP_ANG * TURN_VEL
            # if previous wall is on the left
            else:
                self.bot_vel.angular.z = KP_ANG * -TURN_VEL
        # if closest wall is to the left
        elif (closest_angle < 180):
            self.bot_vel.angular.z = KP_ANG * (closest_angle - 90 - 
                                               (DIST_TO_ANG * (THRESHOLD - closest_distance)))
            self.wall_side = -1
        # if closest wall is to the right
        else:
            self.bot_vel.angular.z = KP_ANG * (closest_angle - 270 +
                                               (DIST_TO_ANG * (THRESHOLD - closest_distance)))
            self.wall_side = 1

        self.cmd_vel.publish(self.bot_vel)
    def stabilize(self, data):
        pass
    
    def turn(self, data):
        closest_distance = np.amin(data.ranges)
        closest_angle = np.argmin(data.ranges)
        self.bot_vel.linear.x = KP_LIN * max(closest_distance, THRESHOLD)
        
        # if closest wall is in front
        if (closest_angle < 30 or closest_angle > 330):
            self.bot_vel.linear.x = 0
            # if previous wall is on the right
            if self.wall_side == 1:
                self.bot_vel.angular.z = KP_ANG * TURN_VEL
            # if previous wall is on the left
            else:
                self.bot_vel.angular.z = KP_ANG * -TURN_VEL
        # if closest wall is to the left
        elif (closest_angle < 180):
            self.bot_vel.angular.z = KP_ANG * (closest_angle - 90 - 
                                               (DIST_TO_ANG * (THRESHOLD - closest_distance)))
            self.wall_side = -1
        # if closest wall is to the right
        else:
            self.bot_vel.angular.z = KP_ANG * (closest_angle - 270 +
                                               (DIST_TO_ANG * (THRESHOLD - closest_distance)))
            self.wall_side = 1

        self.cmd_vel.publish(self.bot_vel)

    def check_state(self, data):
        if self.state == State.INIT:
            pass
        else:
            if self.state == State.FORWARD:
                self.move_forward(data)
            elif self.state == State.TURN:
                self.turn(data)
            elif self.state == State.STABILIZE:
                self.stabilize(data)
            self.cmd_vel.publish(self.bot_vel)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = CornerHeuristic()
    node.run()