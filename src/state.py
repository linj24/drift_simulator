#!/usr/bin/env python3

from __future__ import annotations
from enum import Enum

import numpy as np
import rospy

from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

N_ACTIONS = 3


def unit_circle_range(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class TargetSector(Enum):
    CENTER = 0
    TOP_RIGHT = 1
    TOP_LEFT = 2
    BOT_RIGHT = 3
    BOT_LEFT = 4

    @staticmethod
    def from_angle(angle: float) -> TargetSector:
        degrees = unit_circle_range(angle) / (2 * np.pi) * 360
        if -50 <= degrees < -10:
            return TargetSector.TOP_RIGHT
        elif 10 <= degrees < 50:
            return TargetSector.TOP_LEFT
        elif -180 < degrees < -50:
            return TargetSector.BOT_RIGHT
        elif 50 <= degrees < 180:
            return TargetSector.BOT_LEFT
        else:
            return TargetSector.CENTER

    @staticmethod
    def embed(s: TargetSector, id: int) -> int:
        return id * len(TargetSector) + s.value

    @staticmethod
    def extract(id: int) -> tuple[TargetSector, int]:
        return TargetSector(id % len(TargetSector)), id // len(TargetSector)


class ObstacleSector(Enum):
    BEHIND = 0
    LEFT = 1
    RIGHT = 2

    @staticmethod
    def from_angle(angle: float) -> ObstacleSector:
        degrees = unit_circle_range(angle) / (2 * np.pi) * 360
        if -90 <= degrees < 0:
            return ObstacleSector.RIGHT
        elif 0 <= degrees < 90:
            return ObstacleSector.LEFT
        else:
            return ObstacleSector.BEHIND

    @staticmethod
    def embed(s: ObstacleSector, id: int) -> int:
        return id * len(ObstacleSector) + s.value

    @staticmethod
    def extract(id: int) -> tuple[ObstacleSector, int]:
        return ObstacleSector(id % len(ObstacleSector)), id // len(ObstacleSector)


class Terminal(Enum):
    GOAL = 0
    CRASH = 1
    TIMEOUT = 2

    @property
    def id(self) -> int:
        return self.value


# 2 terminal states, collision and goal
N_NONTERMINAL = (len(TargetSector) ** 2 * len(ObstacleSector) * 2 * 2) ** 2
N_STATES = N_NONTERMINAL + len(Terminal)


class NonTerminal:
    corner: TargetSector
    goal: TargetSector
    closest: ObstacleSector
    within_dist: bool
    turned_corner: bool
    last_changed_timestamp: rospy.rostime.Time
    last_state: NonTerminal | None

    def __init__(self, state_id: int | None = None):
        if state_id is not None:
            id = state_id
            assert 0 <= id < N_NONTERMINAL
            last_state = NonTerminal()
            last_state.last_state = None

            last_state.turned_corner, id = bool(id % 2), id // 2
            last_state.within_dist, id = bool(id % 2), id // 2
            last_state.closest, id = ObstacleSector.extract(id)
            last_state.goal, id = TargetSector.extract(id)
            last_state.corner, id = TargetSector.extract(id)
            self.turned_corner, id = bool(id % 2), id // 2
            self.within_dist, id = bool(id % 2), id // 2
            self.closest, id = ObstacleSector.extract(id)
            self.goal, id = TargetSector.extract(id)
            self.corner, _ = TargetSector.extract(id)

            self.last_state = last_state
        else:
            self.last_state = None

    def copy(self: NonTerminal) -> NonTerminal:
        state = NonTerminal(self.id)
        state.last_changed_timestamp = self.last_changed_timestamp
        return state

    @property
    def id(self) -> int:
        id = TargetSector.embed(self.corner, 0)
        id = TargetSector.embed(self.goal, id)
        id = ObstacleSector.embed(self.closest, id)
        id = id * 2 + self.within_dist
        id = id * 2 + self.turned_corner
        last_state = self.last_state if self.last_state is not None else self
        id = TargetSector.embed(last_state.corner, id)
        id = TargetSector.embed(last_state.goal, id)
        id = ObstacleSector.embed(last_state.closest, id)
        id = id * 2 + last_state.within_dist
        id = id * 2 + last_state.turned_corner
        return id

    def __eq__(self, state: NonTerminal) -> bool:
        return (
            self.corner == state.corner
            and self.goal == state.goal
            and self.within_dist == state.within_dist
            and self.turned_corner == state.turned_corner
            and self.closest == state.closest
        )

    def __repr__(self) -> str:
        return (f"STATE\n------\ncorner: {self.corner}\ngoal: {self.goal}\nclosest: {self.closest}\nwithin: {self.within_dist}\nturned: {self.turned_corner}\n" + 
        f"LAST STATE\n------\n" + ("None" if self.last_state is None else f"corner: {self.last_state.corner}\ngoal: {self.last_state.goal}\nclosest: {self.last_state.closest}\nwithin: {self.last_state.within_dist}\nturned: {self.last_state.turned_corner}\n"))


class State:
    state: NonTerminal | Terminal

    def __init__(self, state_id: int):
        id = state_id
        assert 0 <= id < N_STATES
        if id < len(Terminal):
            self.state = Terminal(id)
        else:
            self.state = NonTerminal(id - len(Terminal))

    @property
    def id(self):
        if isinstance(self.state, Terminal):
            return self.state.id
        else:
            return len(Terminal) + self.state.id


def calculate_state(
    scan: LaserScan,
    pose: Pose,
    time: rospy.rostime.Time,
    dist_threshold: float,
    time_threshold: rospy.Duration,
    corner: Point,
    goal: Point,
    last_state: NonTerminal | None,
) -> NonTerminal:
    state = NonTerminal()

    state = extract_scan_info(state, scan, dist_threshold)

    state = extract_pose_info(state, pose, corner, goal, last_state)

    state.last_changed_timestamp = (
        last_state.last_changed_timestamp if last_state is not None else time
    )

    time_diff = time - state.last_changed_timestamp

    if last_state is None:
        last_state = state.copy()
    elif time_diff < time_threshold and last_state == state:
        last_state = last_state.last_state

    # we only update the last state if the state has changed or if enough time has elapsed
    state.last_state = last_state

    state.last_changed_timestamp = time
    return state


def extract_scan_info(
    state: NonTerminal, scan: LaserScan, dist_threshold: float
) -> NonTerminal:
    closest_angle = np.argmin(scan.ranges)
    closest_dist = scan.ranges[closest_angle]
    state.closest = ObstacleSector.from_angle(np.radians(closest_angle))
    state.within_dist = closest_dist < dist_threshold
    return state


def extract_pose_info(
    state: NonTerminal,
    pose: Pose,
    corner: Point,
    goal: Point,
    last_state: NonTerminal | None,
) -> NonTerminal:
    goal_angle = unit_circle_range(angle_between(pose, goal))
    corner_angle = unit_circle_range(angle_between(pose, corner))
    state.goal = TargetSector.from_angle(goal_angle)
    state.corner = TargetSector.from_angle(corner_angle)
    # rospy.loginfo(f"GOAL ANGLE: {unit_circle_range(goal_angle) / (2 * np.pi) * 360, goal_angle}")
    # rospy.loginfo(f"CORNER ANGLE: {unit_circle_range(corner_angle) / (2 * np.pi) * 360, corner_angle}")
    # rospy.loginfo(f"TURNED: {last_state is not None}, {last_state is not None and last_state.turned_corner}, {np.abs(corner_angle) > np.abs(goal_angle)}")
    state.turned_corner = last_state is not None and (last_state.turned_corner or (
        np.abs(corner_angle) > np.abs(goal_angle)))
    return state


def angle_between(pose: Pose, point: Point) -> float:
    robot_pos = pose.position
    point_angle = np.arctan2(point.y - robot_pos.y, point.x - robot_pos.x)
    robot_heading = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]

    return point_angle - robot_heading
