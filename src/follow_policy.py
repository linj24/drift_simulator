#!/usr/bin/env python3

import os

import rospy
import numpy as np

from std_msgs.msg import UInt8
from drift_simulator.msg import StateReward

import utils.checkpoint as cp
import utils.state

class FollowPolicy:
    """Perform actions according to a specified policy and
    record metrics.
    """
    def __init__(self):

        rospy.init_node("follower")
        self.model_name = rospy.get_param('~model', "heuristic")
        rospy.loginfo(f"Loading model {self.model_name}...")
        self.iterations = 0
        self.iteration_start = rospy.Time.now()
        self.checkpoint = cp.Checkpoint(f"follow_{self.model_name}", [cp.Metric.SUCCESSES, cp.Metric.TIMES])

        # if we launch with the "manual" configuration, only record metrics
        if self.model_name != "manual":
            self.policy = np.loadtxt(os.path.join(cp.CHECKPOINT_DIR, self.model_name, "policy.csv"), dtype=int)
            self.action_pub = rospy.Publisher("/action", UInt8, queue_size=10)
        self.state_reward_sub = rospy.Subscriber(
            "/state_reward", StateReward, self.process_SR
        )

    def process_SR(self, data: StateReward):
        """Execute an action upon receiving a state and record metrics for that state.

        Parameters
        ----------
        data : StateReward
            The state and reward for the current time step.
        """
        _, s, t = data.reward, data.state, data.terminal
        if t:
            self.checkpoint.add_datapoint(cp.Metric.SUCCESSES, s == utils.state.Terminal.GOAL.id)
            self.checkpoint.add_datapoint(cp.Metric.TIMES, (rospy.Time.now() - self.iteration_start).to_sec())
            self.checkpoint.save_checkpoint()
            self.iteration_start = rospy.Time.now()
        else:
            if self.model_name != "manual":
                a = self.policy[s]
                self.action_pub.publish(a)

    def run(self):
        """Listen for state/reward messages.
        """
        rospy.spin()

if __name__ == "__main__":
    node = FollowPolicy()
    node.run()