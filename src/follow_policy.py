#!/usr/bin/env python3

import os

import rospy
import numpy as np

from std_msgs.msg import UInt8
from drift_simulator.msg import StateReward

import learning

class FollowPolicy:
    def __init__(self):

        rospy.init_node("follower")
        model_name = rospy.get_param('~model', "heuristic")
        rospy.loginfo(f"Loading model {model_name}...")
        self.policy = np.loadtxt(os.path.join(learning.CHECKPOINT_DIR, model_name, "policy.csv"), dtype=int)
        self.action_pub = rospy.Publisher("/action", UInt8, queue_size=10)
        self.state_reward_sub = rospy.Subscriber(
            "/state_reward", StateReward, self.process_SR
        )

    def process_SR(self, data: StateReward):
        _, s, _ = data.reward, data.state, data.terminal
        a = self.policy[s]
        self.action_pub.publish(a)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = FollowPolicy()
    node.run()