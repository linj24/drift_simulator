#!/usr/bin/env python3

import rospy

from drift_simulator.msg import StateReward
import state
from action import Turn

class CornerHeuristic(object):
    def __init__(self):
        rospy.init_node('corner_heuristic')
        self.action_pub = rospy.Publisher("/action", int, queue_size=10)
        rospy.Subscriber("/state_reward", StateReward, self.handle_state_reward, queue_size=1)

    def handle_state_reward(self, data: StateReward):
        s = state.State.from_id(data.state)
        if s.within_dist:
            if s.closest_side == 
                a = Turn.LEFT

        self.action_pub.publish(a.value)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = CornerHeuristic()
    node.run()