#!/usr/bin/env python3

import numpy as np
import rospy

from drift_simulator.msg import StateReward

class RL:
    def __init__(self, nS: int, nA: int, gamma: float = 0.9, epsilon: float = 0.8, alpha: float = 0.1):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        rospy.init_node("learner")
        try:
            self.q_function = np.loadtxt("../checkpoints/q.csv").resize((self.nS, self.nA))
        except:
            self.q_function = np.zeros((self.nS, self.nA))
        try:
            self.policy = np.loadtxt("../checkpoints/policy.csv")
        except:
            self.policy = np.zeros(self.nS)

        self.action_pub = rospy.Publisher("action", int, queue_size=10)
        self.state_reward_sub = rospy.Subscriber("state_reward", StateReward, self.process_SR)

        self.last_action = None
        self.last_state = None
    
    def process_SR(self, data: StateReward):
        r, s, t = data.reward, data.state, data.terminal
        a = np.random.choice(self.nA) if np.random.random() < self.epsilon else self.policy[s]
        if self.last_action is not None and self.last_state is not None:
            self.update_model(r, s, a)
        if t:
            self.last_state = None
            self.last_action = None
        else:
            self.last_state = s
            self.last_action = a
            self.action_pub.publish(self.last_action)

    def update_model(self, r: int, s: int, a: int):
        pass

class Sarsa(RL):
    def update_model(self, r: int, s: int, a: int) -> None:
        self.q_function[self.last_state, self.last_action] = self.q_function[self.last_state, self.last_action] + self.alpha * (
            r + self.gamma * self.q_function[s, a] - self.q_function[self.last_state, self.last_action]
        )
        self.policy[self.last_state] = np.argmax(self.q_function[self.last_state])

class QLearning(RL):
    def update_model(self, r: int, s: int, a: int) -> None:
        self.q_function[self.last_state, self.last_action] = self.q_function[self.last_state, self.last_action] + self.alpha * (
            r
            + self.gamma * np.max([self.q_function[s, a] for a in range(self.nA)])
            - self.q_function[self.last_state, self.last_action]
        )
        self.policy[self.last_state] = np.argmax(self.q_function[self.last_state])

