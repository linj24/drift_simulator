from __future__ import annotations

from enum import Enum

import os
import sys
import numpy as np

class Metric(Enum):
    POLICY_UPDATES = (0, int)
    SUCCESSES = (1, int)
    TIMES = (2, float)


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "checkpoints")

class Checkpoint:
    model: str
    iterations: int = 0

    metrics: dict[Metric, list] = {}

    def __init__(self, model: str, metrics: list[Metric]):
        self.model = model
        for metric in metrics:
            self.load_metric(metric)
    
    def checkpoint_folder(self) -> str:
        return os.path.join(CHECKPOINT_DIR, self.model)

    def checkpoint_filename(self, name: str) -> str:
        return os.path.join(self.checkpoint_folder(), f"{name}.csv")

    def load_metric(self, metric: Metric):
        try:
            self.metrics[metric] = list(np.loadtxt(self.checkpoint_filename(metric.name), dtype=metric.value[1]))
        except OSError:
            print(f"Creating default {metric.name} metric...")
            os.makedirs(self.checkpoint_folder(), exist_ok=True)
            self.metrics[metric] = []
        self.iterations = max(len(self.metrics[metric]), self.iterations)

    def add_datapoint(self, metric: Metric, data):
        self.metrics[metric].append(data)

    def save_checkpoint(self):
        for metric in self.metrics:
            _, typ = metric.value
            if typ is int:
                np.savetxt(self.checkpoint_filename(metric.name), self.metrics[metric], fmt='%i')
            else:
                np.savetxt(self.checkpoint_filename(metric.name), self.metrics[metric])