#!/usr/bin/env python3

import os
import sys

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
Q_MATRIX_FILENAME = os.path.join(PARENT_DIR, "checkpoints", "q.csv")
POLICY_FILENAME = os.path.join(PARENT_DIR, "checkpoints", "policy.csv")

print(Q_MATRIX_FILENAME, POLICY_FILENAME)
class RL:
    def __init__(
        self,
        nS: int,
        nA: int,
        gamma: float = 0.9,
        epsilon: float = 0.8,
        alpha: float = 0.1,
    ):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        rospy.init_node("learner")
        self.q_function = np.loadtxt(Q_MATRIX_FILENAME)
        if np.shape(self.q_function) != (self.nS, self.nA):
            print("Creating default q function...")
            self.q_function = np.zeros((self.nS, self.nA))
        self.policy = np.loadtxt(POLICY_FILENAME, dtype=int)
        if np.shape(self.q_function) != (self.nS,):
            print("Creating default policy...")
            self.policy = np.zeros(self.nS, dtype=int)
            for id in range(self.nS):
                action = heuristic.policy(id)
                if action is not None:
                    self.policy[id] = action

        self.action_pub = rospy.Publisher("/action", UInt8, queue_size=10)
        self.state_reward_sub = rospy.Subscriber(
            "/state_reward", StateReward, self.process_SR
        )

        self.last_action = None
        self.last_state = None

        self.iteration = 0

    def process_SR(self, data: StateReward):
        r, s, t = data.reward, data.state, data.terminal
        a = (
            np.random.choice(self.nA)
            if np.random.random() < self.epsilon
            else self.policy[s]
        )
        if self.last_action is not None and self.last_state is not None:
            self.update_model(r, s, a)
        if t:
            self.last_state = None
            self.last_action = None
            self.iteration += 1
            if self.iteration % SAVE_ITERATIONS == 0:
                print(f"Checkpoint: iteration {self.iteration}")
                np.savetxt(Q_MATRIX_FILENAME, self.q_function)
                np.savetxt(POLICY_FILENAME, self.policy, fmt='%i')
        else:
            self.last_state = s
            self.last_action = a
            self.action_pub.publish(self.last_action)

    def update_model(self, r: float, s: int, a: int):
        pass

    def run(self):
        rospy.spin()


class Sarsa(RL):
    def update_model(self, r: float, s: int, a: int) -> None:
        self.q_function[self.last_state, self.last_action] = self.q_function[
            self.last_state, self.last_action
        ] + self.alpha * (
            r
            + self.gamma * self.q_function[s, a]
            - self.q_function[self.last_state, self.last_action]
        )
        self.policy[self.last_state] = np.argmax(self.q_function[self.last_state])


class QLearning(RL):
    def update_model(self, r: float, s: int, a: int) -> None:
        self.q_function[self.last_state, self.last_action] = self.q_function[
            self.last_state, self.last_action
        ] + self.alpha * (
            r
            + self.gamma * np.max([self.q_function[s, a] for a in range(self.nA)])
            - self.q_function[self.last_state, self.last_action]
        )
        self.policy[self.last_state] = np.argmax(self.q_function[self.last_state])


if __name__ == "__main__":
    node = Sarsa(state.N_STATES, len(action.Turn))
    node.run()