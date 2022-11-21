#!/usr/bin/env python3

import numpy as np
import rospy

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3


def sarsa(
    env, nS, nA, gamma=0.9, epsilon=0.8, alpha=0.1, max_steps=100, max_episodes=10000
):

    """
    Learn the action value function and policy by the sarsa
    method for a given environment, gamma, epsilon, alpha, and a given
    maximum number of episodes to train for, and a given maximum number
    of steps to train for in an episode.

    Parameters:
    ----------
    env: environment object (which is not the same as P) 
    nS, nA, gamma: defined at beginning of file
    episilon: for the epsilon greedy policy used
    alpha: step update parameter
    max_steps: maximum number of steps to train for in an episode
    max_episodees: maximum number of episodes to train for

    Returns:
    ----------
    q_function: np.ndarray[nS, nA]
    policy: np.ndarray[nS]
    """
    q_function = np.zeros((nS, nA))
    policy = np.zeros(nS, dtype=int)
    for _ in range(max_episodes):
        s = env.reset()
        a = np.random.choice(nA) if np.random.random() < epsilon else policy[s]
        for _ in range(max_steps):
            s_p, r, is_terminal, _ = env.step(a)
            a_p = np.random.choice(nA) if np.random.random() < epsilon else policy[s_p]
            q_function[s, a] = q_function[s, a] + alpha * (
                r + gamma * q_function[s_p, a_p] - q_function[s, a]
            )
            policy[s] = np.argmax(q_function[s])
            s = s_p
            a = a_p
            if is_terminal:
                break
    return q_function, policy


def q_learning(
    env, nS, nA, gamma=0.9, epsilon=0.8, alpha=0.1, max_steps=100, max_episodes=10000
):

    """
    Learn the action value function and policy by the q-learning
    method for a given environment, gamma, epsilon, alpha, and a given
    maximum number of episodes to train for, and a given maximum number
    of steps to train for in an episode.

    Parameters:
    ----------
    env: environment object (which is not the same as P) 
    nS, nA, gamma: defined at beginning of file
    episilon: for the epsilon greedy policy used
    alpha: step update parameter
    max_steps: maximum number of steps to train for in an episode
    max_episodees: maximum number of episodes to train for

    Returns:
    ----------
    q_function: np.ndarray[nS, nA]
    policy: np.ndarray[nS]
    """
    q_function = np.zeros((nS, nA))
    policy = np.zeros(nS, dtype=int)
    for _ in range(max_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = np.random.choice(nA) if np.random.random() < epsilon else policy[s]
            s_p, r, is_terminal, _ = env.step(a)
            q_function[s, a] = q_function[s, a] + alpha * (
                r
                + gamma * np.max([q_function[s_p, a_p] for a_p in range(nA)])
                - q_function[s, a]
            )
            policy[s] = np.argmax(q_function[s])
            s = s_p
            if is_terminal:
                break
    return q_function, policy
