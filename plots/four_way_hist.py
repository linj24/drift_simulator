#!/usr/bin/env python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
PARENT_DIR = os.path.dirname(PLOTS_DIR)
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "checkpoints")

MODELS = ["follow_heuristic", "follow_manual", "follow_qlearning_0.5_q_7610", "follow_sarsa_0.5_q_7610"]
LABELS = ["Heuristic", "Manual", "Q-Learning", "Sarsa"]

if __name__ == "__main__":
    # generate a plot for each model in the MODELS list
    for model, label in zip(MODELS, LABELS):

        # load checkpoint
        successes_filename = os.path.join(
            CHECKPOINT_DIR, model, "SUCCESSES.csv"
        )
        times_filename = os.path.join(
            CHECKPOINT_DIR, model, "TIMES.csv"
        )
        successes = np.loadtxt(successes_filename, dtype=bool)
        times = np.loadtxt(times_filename)
        succ_times = times[successes]
        valid_times = succ_times[(8 < succ_times) & (succ_times < 24)][:60]
        plt.title(f"Goal Times: {label}")
        plt.xlabel("Count")
        plt.ylabel("Goal time (s)")
        plt.hist(valid_times)
        plt.savefig(os.path.join(PLOTS_DIR, f"{model}_hist.png"))
        plt.clf()
        print(f"AVG: {np.average(valid_times)}")
        print(f"STD: {np.std(valid_times)}")
        print(f"FAILURE: {1 - np.count_nonzero(successes) / len(successes)}")