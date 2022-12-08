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
Q_MATRIX_FILENAME = os.path.join(CHECKPOINT_DIR, "q.csv")
POLICY_FILENAME = os.path.join(CHECKPOINT_DIR, "policy.csv")
POLICY_UPDATES_FILENAME = os.path.join(CHECKPOINT_DIR, "policy_updates.csv")

class RL(ABC):
    def __init__(
        self,
        nS: int,
        nA: int,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        alpha: float = 0.1,
    ):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        rospy.init_node("learner")
        try:
            self.q_function = np.loadtxt(Q_MATRIX_FILENAME)
        except OSError:
            print("Creating default q function...")
            self.q_function = np.zeros((self.nS, self.nA))
            for id in range(self.nS):
                action = heuristic.policy(id)
                if action is not None:
                    self.q_function[id, action] = 0.1
        try:
            self.policy = np.loadtxt(POLICY_FILENAME, dtype=int)
        except OSError:
            print("Creating default policy...")
            self.policy = np.zeros(self.nS, dtype=int)
            for id in range(self.nS):
                action = heuristic.policy(id)
                if action is not None:
                    self.policy[id] = action
        try:
            self.policy_updates = list(np.loadtxt(POLICY_UPDATES_FILENAME, dtype=int))
        except OSError:
            print("Creating default policy updates...")
            self.policy_updates = []

        self.action_pub = rospy.Publisher("/action", UInt8, queue_size=10)
        self.state_reward_sub = rospy.Subscriber(
            "/state_reward", StateReward, self.process_SR
        )

        self.last_action = None
        self.last_state = None
        self.updates_in_current_episode = 0

        self.iteration = 0

    def process_SR(self, data: StateReward):
        r, s, t = data.reward, data.state, data.terminal
        if np.random.random() < self.epsilon:
            a = np.random.choice(self.nA)
            # rospy.loginfo(f"RANDOM: {a}")
        else:
            a = self.policy[s]
            # rospy.loginfo(f"H = {heuristic.policy(s)}")
            # rospy.loginfo(f"A = {a}")
            # rospy.loginfo(f"S = {s}")
            # rospy.loginfo(state.State(s).state)
        if self.last_action is not None and self.last_state is not None:
            self.q_function[self.last_state, self.last_action] = self.q_function[
                self.last_state, self.last_action
            ] + self.alpha * (
                r
                + self.gamma * self.next_action_value(s, a)
                - self.q_function[self.last_state, self.last_action]
            )
            best_action = np.argmax(self.q_function[self.last_state])
            if self.policy[self.last_state] != best_action:
                self.updates_in_current_episode += 1
            self.policy[self.last_state] = best_action
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
    node = Sarsa(state.N_STATES, len(action.Turn), epsilon=0.1)
    node.run()