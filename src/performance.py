#!/usr/bin/env python3

import os
import sys

from abc import ABC, abstractmethod

import rospy
import numpy as np

from std_msgs.msg import UInt8
from drift_simulator.msg import StateReward

import state
import action
import heuristic

# save q-matrix and policy after every 10 iterations
SAVE_ITERATIONS = 10

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "checkpoints")
TIMES_FILENAME = os.path.join(CHECKPOINT_DIR, "times.csv")

class PerformanceProfiler:
    def __init__(
        self,
    ):

        rospy.init_node("performance_profiler")
        try:
            self.policy_updates = list(np.loadtxt(TIMES_FILENAME, dtype=int))
        except OSError:
            print("Creating default policy updates...")
            self.policy_updates = []

        self.state_reward_sub = rospy.Subscriber(
            "/state_reward", StateReward, self.process_SR
        )

        self.iteration = -1
        self.successes = 0
        self.time_average = None
        self.time_std = None

    def process_SR(self, data: StateReward):
        r, s, t = data.reward, data.state, data.terminal
        if t:
            self.last_state = None
            self.last_action = None
            self.policy_updates.append(self.updates_in_current_episode)
            self.updates_in_current_episode = 0
            self.iteration += 1
            if self.iteration % SAVE_ITERATIONS == 0:
                print(f"Checkpoint: iteration {self.iteration}")
                np.savetxt(Q_MATRIX_FILENAME, self.q_function)
                np.savetxt(POLICY_FILENAME, self.policy, fmt='%i')
                np.savetxt(POLICY_UPDATES_FILENAME, self.policy_updates, fmt='%i')
        else:
            self.last_state = s
            self.last_action = a
            self.action_pub.publish(self.last_action)

    @abstractmethod 
    def next_action_value(self, s: int, a: int) -> float:
        pass

    def run(self):
        rospy.spin()


class Sarsa(RL):
    def next_action_value(self, s: int, a: int) -> float:
        return self.q_function[s, a]


class QLearning(RL):
    def next_action_value(self, s: int, a: int) -> float:
        return np.max([self.q_function[s, a] for a in range(self.nA)])


if __name__ == "__main__":
    node = QLearning(state.N_STATES, len(action.Turn), epsilon=0.1)
    node.run()