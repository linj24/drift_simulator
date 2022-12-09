from __future__ import annotations

from enum import Enum

import os
import sys
import numpy as np

class Metric(Enum):
    """Allow specific metrics to be calculated during learning, with data saved
    as the specified type.
    """
    POLICY_UPDATES = (0, int)
    SUCCESSES = (1, int)
    TIMES = (2, float)

ALL_METRICS = [Metric.POLICY_UPDATES, Metric.SUCCESSES, Metric.TIMES]

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "checkpoints")

class Checkpoint:
    """Load and save metrics such as success rate and time to goal for a given
    model. Implicitly indexed by episode.
    """
    model: str
    iterations: int = 0

    metrics: dict[Metric, list] = {}

    def __init__(self, model: str, metrics: list[Metric]):
        self.model = model
        for metric in metrics:
            self.load_metric(metric)
    
    def checkpoint_folder(self) -> str:
        """Get the path to the folder for the specified model.

        Returns
        -------
        str
            The folder path in the checkpoints folder.
        """
        return os.path.join(CHECKPOINT_DIR, self.model)

    def checkpoint_filename(self, name: str) -> str:
        """Get the path to a specific CSV file in the checkpoint folder for a model.

        Parameters
        ----------
        name : str
            The name of the CSV file (no extension).

        Returns
        -------
        str
            The path to the CSV file.
        """
        return os.path.join(self.checkpoint_folder(), f"{name}.csv")

    def load_metric(self, metric: Metric):
        """Load a checkpoint to append new data for a metric.

        Parameters
        ----------
        metric : Metric
            The metric to load.
        """
        try:
            self.metrics[metric] = list(np.loadtxt(self.checkpoint_filename(metric.name), dtype=metric.value[1]))
        except OSError:
            print(f"Creating default {metric.name} metric...")
            os.makedirs(self.checkpoint_folder(), exist_ok=True)
            self.metrics[metric] = []
        self.iterations = max(len(self.metrics[metric]), self.iterations)

    def add_datapoint(self, metric: Metric, data):
        """Add a datapoint for the specified metric.

        Parameters
        ----------
        metric : Metric
            _description_
        data : _type_
            _description_
        """
        self.metrics[metric].append(data)

    def save_checkpoint(self):
        """Write a metric to the checkpoints model folder.
        """
        for metric in self.metrics:
            _, typ = metric.value
            filename = self.checkpoint_filename(metric.name)
            if typ is int:
                np.savetxt(filename, self.metrics[metric], fmt='%i')
            else:
                np.savetxt(filename, self.metrics[metric])