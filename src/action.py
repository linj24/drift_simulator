#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, Vector3

LIN_VEL = 1.00
ANG_VEL = 1.82


class Action:
    def __init__(self):
        rospy.init_node("RL_action")
        self.cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/action", int, self.perform_action)

    def perform_action(self, data: int) -> None:
        if data == 0:
            ang = ANG_VEL
        elif data == 1:
            ang = -ANG_VEL
        else:
            ang = 0
        self.cmd_vel.publish(
            Twist(
                linear=Vector3(x=LIN_VEL, y=0, z=0), angular=Vector3(x=0, y=0, z=ang),
            )
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = Action()
    node.run()
