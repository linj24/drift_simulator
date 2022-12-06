#!/usr/bin/env python3

import os

import rospy
import numpy as np

from std_msgs.msg import UInt8
from drift_simulator.msg import StateReward

import state
import learning

class FollowPolicy:
    def __init__(
        self, policy_name: str
    ):

        rospy.init_node("follower")
        self.policy = np.loadtxt(os.path.join(learning.CHECKPOINT_DIR, policy_name, "policy.csv"), dtype=int)
        self.action_pub = rospy.Publisher("/action", UInt8, queue_size=10)
        self.state_reward_sub = rospy.Subscriber(
            "/state_reward", StateReward, self.process_SR
        )

    def process_SR(self, data: StateReward):
        r, s, t = data.reward, data.state, data.terminal
        a = self.policy[s]
        print(a)
        self.action_pub.publish(a)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = FollowPolicy("qlearning_0.9_0.1_0.1_10430")
    node.run()