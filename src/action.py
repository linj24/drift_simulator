#!/usr/bin/env python3

from enum import Enum

import rospy

from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import UInt8

LIN_VEL = 0.80
ANG_VEL = 1.30

class Turn(Enum):
    """The action space for controlling a robot.
    """
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2

class Action:
    """Convert an action into a robot velocity.
    """
    def __init__(self):
        rospy.init_node("RL_action")
        self.cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/action", UInt8, self.perform_action)

    def perform_action(self, data: UInt8) -> None:
        """Map an action onto a Twist message.

        Parameters
        ----------
        data : UInt8
            The action to map in ID form.
        """
        if data.data == Turn.LEFT.value:
            ang = ANG_VEL
        elif data.data == Turn.RIGHT.value:
            ang = -ANG_VEL
        else:
            ang = 0
        self.cmd_vel.publish(
            Twist(
                linear=Vector3(x=LIN_VEL, y=0, z=0), angular=Vector3(x=0, y=0, z=ang),
            )
        )

    def run(self):
        """Listen for incoming actions.
        """
        rospy.spin()


if __name__ == "__main__":
    node = Action()
    node.run()
