#!/usr/bin/env python3

import os
import sys

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
PARENT_DIR = os.path.dirname(PLOTS_DIR)
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "checkpoints")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_filename = sys.argv[1]
        policy_updates_filename = os.path.join(
            CHECKPOINT_DIR, checkpoint_filename, "POLICY_UPDATES.csv"
        )
        policy_updates = list(np.loadtxt(policy_updates_filename, dtype=int))
        plt.title(f"Total Q-Matrix updates: {checkpoint_filename}")
        plt.xlabel("Episode")
        plt.ylabel("Total # updates")
        plt.plot(np.arange(len(policy_updates)), np.cumsum(policy_updates))
        plt.savefig(os.path.join(PLOTS_DIR, f"{checkpoint_filename}.png"))
    else:
        print("Usage: python3 performance.py CHECKPOINT_MODEL_NAME")
