#!/usr/bin/env python3

import rospy

from drift_simulator.msg import StateReward


class CornerHeuristic(object):
    def __init__(self):
        rospy.init_node('corner_heuristic')
        self.action_pub = rospy.Publisher("/action", int, queue_size=10)
        rospy.Subscriber("/state_reward", StateReward, self.handle_state_reward, queue_size=1)

    def handle_state_reward(self):
        pass

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = CornerHeuristic()
    node.run()