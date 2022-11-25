#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
from sensor_msgs.msg import LaserScan, Odometry

N_ACTIONS = 3

N_CORNERS = 5
N_GOALS = 5
N_STATES = (N_CORNERS * N_GOALS) ** 2

DIST_TO_WALL = 1


class State:

    corner: int
    goal: int
    turned_corner: int
    closest_side: bool
    within_dist: bool
    last_corner: int
    last_goal: int

    @property
    def id(self) -> int:
        id = self.corner
        id = id * N_CORNERS + self.goal
        id = id * N_GOALS + self.last_corner
        id = id * N_CORNERS + self.last_goal
        return id


def calculate_state(
    scan: LaserScan,
    dist: float,
    odom: Odometry,
    corner: tuple[float, float],
    goal: tuple[float, float],
) -> State:
    state = State()

    closest_angle = np.argmin(scan.ranges)
    closest_dist = scan.ranges[closest_angle]
    state.closest_side = bool(closest_angle < 180)
    state.within_dist = closest_dist < DIST_TO_WALL


def angle_between(odom: Odometry, point: tuple[float, float]) -> float:
    pass
