#!/usr/bin/env python3

from __future__ import annotations

from enum import Enum
import numpy as np
import rospy

from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

N_ACTIONS = 3


class Sector(Enum):
    CENTER = 0
    TOP_RIGHT = 1
    TOP_LEFT = 2
    BOT_RIGHT = 3
    BOT_LEFT = 4

# 2 terminal states, collision and goal
N_STATES = (len(Sector) ** 2) ** 2 * 8 + 2

class State:

    corner: Sector
    goal: Sector
    turned_corner: bool
    closest_side: bool
    within_dist: bool
    last_changed_timestamp: rospy.rostime.Time
    last_corner: Sector
    last_goal: Sector

    @staticmethod
    def from_id(id: int) -> State:
        assert 0 <= id < N_STATES
        state = State()
        state.within_dist = id % 2
        id //= 2
        state.closest_side = id % 2
        id //= 2
        state.turned_corner = id % 2
        id //= 2
        state.last_goal = Sector(id % N_GOALS)
        id //= N_GOALS
        state.last_corner = Sector(id % N_CORNERS)
        id //= N_CORNERS
        state.last_goal = Sector(id % N_GOALS)
        id //= N_GOALS
        state.corner = Sector(id)
        return state

    @property
    def id(self) -> int:
        id = self.corner
        id = id * N_GOALS + self.goal
        id = id * N_CORNERS + self.last_corner
        id = id * N_GOALS + self.last_goal
        id = id * 2 + self.turned_corner
        id = id * 2 + self.closest_side
        id = id * 2 + self.within_dist
        return id + 2

    def __eq__(self, state: State) -> bool:
        return (
            self.corner == state.corner
            and self.goal == state.goal
            and self.turned_corner == state.turned_corner
            and self.closest_side == state.closest_side
            and self.within_dist == state.within_dist
            and self.last_changed_timestamp == state.last_changed_timestamp
            and self.last_corner == state.last_corner
            and self.last_goal == state.last_goal
        )


def calculate_state(
    scan: LaserScan,
    pose: Pose,
    time: rospy.rostime.Time,
    dist_threshold: float,
    time_threshold: rospy.Duration,
    corner: Point,
    goal: Point,
    last_state: State | None,
) -> State:
    state = State()

    closest_angle = np.argmin(scan.ranges)
    closest_dist = scan.ranges[closest_angle]
    state.closest_side = bool(closest_angle < 180)
    state.within_dist = closest_dist < dist_threshold

    goal_angle = angle_between(pose, goal)
    corner_angle = angle_between(pose, corner)
    state.goal = angle_to_int(goal_angle)
    state.corner = angle_to_int(corner_angle)
    state.turned_corner = (last_state and last_state.turned_corner) or (
        corner_angle < goal_angle
        if -np.pi / 2 < corner_angle < np.pi / 2
        else goal_angle < corner_angle
    )

    if last_state is not None:
        state.last_changed_timestamp = last_state.last_changed_timestamp
        state.last_corner = last_state.last_corner
        state.last_goal = last_state.last_goal
        time_diff = time - last_state.last_changed_timestamp
        if time_diff < time_threshold and last_state == state:
            return state
        else:
            state.last_corner = last_state.corner
            state.last_goal = last_state.goal
    else:
        state.last_corner = state.corner
        state.last_goal = state.goal

    state.last_changed_timestamp = time
    return state


def angle_to_int(angle: float) -> int:
    degrees = angle / (2 * np.pi) * 360
    if -50 <= angle < -10:
        return Sector.TOP_RIGHT.value
    elif 10 <= angle < 50:
        return Sector.TOP_LEFT.value
    elif -180 < angle < -50:
        return Sector.BOT_RIGHT.value
    elif 50 <= angle < 180:
        return Sector.BOT_LEFT.value
    else:
        return Sector.CENTER.value


def angle_between(pose: Pose, point: Point) -> float:
    robot_pos = pose.position
    point_angle = np.arctan2(point.y - robot_pos.y, point.x - robot_pos.x)

    robot_heading = euler_from_quaternion(pose.orientation)[2]

    return point_angle - robot_heading
