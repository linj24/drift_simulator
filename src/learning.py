#!/usr/bin/env python3

from abc import ABC, abstractmethod

import rospy
import numpy as np

from std_msgs.msg import UInt8
from drift_simulator.msg import StateReward

import utils.state as state
import action
import utils.checkpoint as cp
import heuristic

# save q-matrix and policy after every 10 iterations
SAVE_ITERATIONS = 10
DEFAULT_Q_VALUE = 0.5

class RL(ABC):
    def __init__(
        self,
        model: str,
        nS: int,
        nA: int,
        gamma: float,
        epsilon: float,
        alpha: float,
    ):
        self.model = model
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.checkpoint = cp.Checkpoint(self.model, cp.ALL_METRICS)

        rospy.init_node("learner")

        # load checkpoint data, initialize with heuristic if not available
        try:
            self.q_function = np.loadtxt(self.checkpoint.checkpoint_filename("q"))
        except OSError:
            rospy.loginfo("Creating default q function...")
            self.q_function = np.zeros((self.nS, self.nA))
            for id in range(self.nS):
                action = heuristic.policy(id)
                if action is not None:
                    self.q_function[id, action] = DEFAULT_Q_VALUE
        try:
            self.policy = np.loadtxt(self.checkpoint.checkpoint_filename("policy"), dtype=int)
        except OSError:
            rospy.loginfo("Creating default policy...")
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
        self.updates_in_current_episode = 0
        self.iteration_start = rospy.Time.now()

        self.iteration = 0

    def process_SR(self, data: StateReward):
        r, s, t = data.reward, data.state, data.terminal
        if np.random.random() < self.epsilon:
            a = np.random.choice(self.nA)
        else:
            a = self.policy[s]
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
            self.checkpoint.add_datapoint(cp.Metric.POLICY_UPDATES, self.updates_in_current_episode)
            self.checkpoint.add_datapoint(cp.Metric.SUCCESSES, s == state.Terminal.GOAL.id)
            self.checkpoint.add_datapoint(cp.Metric.TIMES, (rospy.Time.now() - self.iteration_start).to_sec())
            self.updates_in_current_episode = 0
            self.iteration_start = rospy.Time.now()
            self.iteration += 1
            if self.iteration % SAVE_ITERATIONS == 0:
                print(f"Checkpoint: iteration {self.iteration}")
                np.savetxt(self.checkpoint.checkpoint_filename("q"), self.q_function)
                np.savetxt(self.checkpoint.checkpoint_filename("policy"), self.policy, fmt='%i')
                self.checkpoint.save_checkpoint()
        else:
            self.last_state = s
            self.last_action = a
            self.action_pub.publish(self.last_action)

    @abstractmethod 
    def name(self) -> str:
        pass

    @abstractmethod 
    def next_action_value(self, s: int, a: int) -> float:
        pass

    def run(self):
        rospy.spin()


class Sarsa(RL):
    def name(self) -> str:
        return "sarsa"

    def next_action_value(self, s: int, a: int) -> float:
        return self.q_function[s, a]


class QLearning(RL):
    def name(self) -> str:
        return "qlearning"

    def next_action_value(self, s: int, a: int) -> float:
        return np.max([self.q_function[s, a] for a in range(self.nA)])


if __name__ == "__main__":
    model_name = rospy.get_param('/learner/model_name', "my_model")
    gamma = rospy.get_param('/learner/gamma', 0.9)
    epsilon = rospy.get_param('/learner/epsilon', 0.1)
    alpha = rospy.get_param('/learner/alpha', 0.1)
    algorithm = rospy.get_param('/learner/algorithm', "qlearning")
    if algorithm == "sarsa":
        node = Sarsa(model_name, state.N_STATES, len(action.Turn), gamma=gamma, epsilon=epsilon, alpha=alpha)
    else:
        # default to Q-learning if invalid name given
        node = QLearning(model_name, state.N_STATES, len(action.Turn), gamma=gamma, epsilon=epsilon, alpha=alpha)
    node.run()