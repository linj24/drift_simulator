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
    """Cycle an angle through to the range [-pi, pi].

    Parameters
    ----------
    angle : float
        The angle to cycle.

    Returns
    -------
    float
        The angle when in the range [-pi, pi].
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class TargetSector(Enum):
    """Represent an angle range that a target object falls in.
    """
    CENTER = 0
    TOP_RIGHT = 1
    TOP_LEFT = 2
    BOT_RIGHT = 3
    BOT_LEFT = 4

    @staticmethod
    def from_angle(angle: float) -> TargetSector:
        """Map an angle to the target sector it falls within.

        Parameters
        ----------
        angle : float
            The angle to discretize.

        Returns
        -------
        TargetSector
            The sector that the angle falls in.
        """
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
        """Create an ID by combining existing state with a target sector.

        Parameters
        ----------
        s : TargetSector
            The sector to embed in the ID.
        id : int
            The ID for the rest of the state.

        Returns
        -------
        int
            The ID for the combined state.
        """
        return id * len(TargetSector) + s.value

    @staticmethod
    def extract(id: int) -> tuple[TargetSector, int]:
        """Decompose the ID of a combined state into a target sector and
        everything else.

        Parameters
        ----------
        id : int
            The ID to decompose.

        Returns
        -------
        tuple[TargetSector, int]
            The extracted target sector and ID for the remaining state.
        """
        return TargetSector(id % len(TargetSector)), id // len(TargetSector)


class ObstacleSector(Enum):
    """Represent an angle range that an obstacle falls in.
    """
    BEHIND = 0
    LEFT = 1
    RIGHT = 2

    @staticmethod
    def from_angle(angle: float) -> ObstacleSector:
        """Map an angle to the obstacle sector that it falls in.

        Parameters
        ----------
        angle : float
            The angle to discretize.

        Returns
        -------
        ObstacleSector
            The sector that the angle falls in.
        """
        degrees = unit_circle_range(angle) / (2 * np.pi) * 360
        if -90 <= degrees < 0:
            return ObstacleSector.RIGHT
        elif 0 <= degrees < 90:
            return ObstacleSector.LEFT
        else:
            return ObstacleSector.BEHIND

    @staticmethod
    def embed(s: ObstacleSector, id: int) -> int:
        """Create an ID by combining existing state with an obstacle sector.

        Parameters
        ----------
        s : ObstacleSector
            The sector to embed in the ID.
        id : int
            The ID for the rest of the state.

        Returns
        -------
        int
            The ID for the combined state.
        """
        return id * len(ObstacleSector) + s.value

    @staticmethod
    def extract(id: int) -> tuple[ObstacleSector, int]:
        """Decompose the ID of a combined state into an obstacle sector and
        everything else.

        Parameters
        ----------
        id : int
            The ID to decompose.

        Returns
        -------
        tuple[ObstacleSector, int]
            The extracted obstacle sector and ID for the remaining state.
        """
        return ObstacleSector(id % len(ObstacleSector)), id // len(ObstacleSector)


class Terminal(Enum):
    """A terminal state.
    """
    GOAL = 0
    CRASH = 1
    TIMEOUT = 2

    @property
    def id(self) -> int:
        """Get the ID for a terminal state. Terminal states occupy the smallest IDs.

        Returns
        -------
        int
            The ID for the state.
        """
        return self.value


# 2 terminal states, collision and goal
N_NONTERMINAL = (len(TargetSector) ** 2 * len(ObstacleSector) * 2 * 2) ** 2
N_STATES = N_NONTERMINAL + len(Terminal)


class NonTerminal:
    """A nonterminal state consisting of:
    - The target sector the corner is in
    - The target sector the goal is in
    - The obstacle sector the closest wall is in
    - Whether the closest wall is within some distance
    - Whether the corner has been turned
    - The last state
    The timestamp where the state was last changed is also stored for convenience.
    """
    corner: TargetSector
    goal: TargetSector
    closest: ObstacleSector
    within_dist: bool
    turned_corner: bool
    last_changed_timestamp: rospy.rostime.Time
    last_state: NonTerminal | None

    def __init__(self, state_id: int | None = None):
        if state_id is not None:
            id = state_id - len(Terminal)
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

    def copy(self) -> NonTerminal:
        """Copy the contents of the current state to a different object.

        Returns
        -------
        NonTerminal
            A copy of the current state.
        """
        state = NonTerminal(self.id)
        state.last_changed_timestamp = self.last_changed_timestamp
        return state

    @property
    def id(self) -> int:
        """Get the ID for a nonterminal state. Nonterminal states occupy the
        largest IDs.

        Returns
        -------
        int
            The ID for the state.
        """
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
        return id + len(Terminal)

    def __eq__(self, state: NonTerminal) -> bool:
        """Check to see if two states are equal. Two states are equal if
        everything but the last state matches.

        Parameters
        ----------
        state : NonTerminal
            The state to compare the current object to.

        Returns
        -------
        bool
            True if the states are equal. False otherwise.
        """
        return (
            self.corner == state.corner
            and self.goal == state.goal
            and self.within_dist == state.within_dist
            and self.turned_corner == state.turned_corner
            and self.closest == state.closest
        )

    def __repr__(self) -> str:
        """A string representation of a state listing its components.

        Returns
        -------
        str
            The string representation of the current state.
        """
        return (f"STATE\n------\ncorner: {self.corner}\ngoal: {self.goal}\nclosest: {self.closest}\nwithin: {self.within_dist}\nturned: {self.turned_corner}\n" + 
        f"LAST STATE\n------\n" + ("None" if self.last_state is None else f"corner: {self.last_state.corner}\ngoal: {self.last_state.goal}\nclosest: {self.last_state.closest}\nwithin: {self.last_state.within_dist}\nturned: {self.last_state.turned_corner}\n"))


class State:
    """A wrapper class for Terminal and NonTerminal states.
    """
    state: NonTerminal | Terminal

    def __init__(self, state_id: int):
        id = state_id
        assert 0 <= id < N_STATES
        if id < len(Terminal):
            self.state = Terminal(id)
        else:
            self.state = NonTerminal(id)

    @property
    def id(self) -> int:
        """Get the ID for a state.

        Returns
        -------
        int
            The ID of the current state.
        """
        return self.state.id


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
    """Calculate a NonTerminal state from the given sensor and simulator information.

    Parameters
    ----------
    scan : LaserScan
        The most recent Lidar message.
    pose : Pose
        The pose of the robot in the simulation.
    time : rospy.rostime.Time
        The simulation time at which the scan and pose were taken.
    dist_threshold : float
        The maximum distance required to trigger the "within_distance" state.
    time_threshold : rospy.Duration
        The time threshold at which the last state will update if not changed sooner.
    corner : Point
        The location of the corner in the simulation.
    goal : Point
        The location of the goal in the simulation.
    last_state : NonTerminal | None
        The state that was calculated at the last timestep.

    Returns
    -------
    NonTerminal
        A state representing all the input information.
    """
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
    """Determine the closest and within_dist components of a NonTerminal state
    from a Lidar scan.

    Parameters
    ----------
    state : NonTerminal
        The state object to store the results in.
    scan : LaserScan
        The laser scan message to process.
    dist_threshold : float
        The maximum distance to trigger within_dist.

    Returns
    -------
    NonTerminal
        The state object, updated with closest and within_dist.
    """
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
    """Determine the corner sector, goal sector, and whether the corner has been
    turned for a nonterminal state from a robot pose.

    Parameters
    ----------
    state : NonTerminal
        The state object to store the results in.
    pose : Pose
        The pose of the robot in the simulation.
    corner : Point
        The location of the corner in the simulation.
    goal : Point
        The location of the goal in the simulation.
    last_state : NonTerminal | None
        The state that was calculated at the last timestep.

    Returns
    -------
    NonTerminal
        The state object, updated with corner, goal and turned_corner.
    """
    goal_angle = unit_circle_range(angle_between(pose, goal))
    corner_angle = unit_circle_range(angle_between(pose, corner))
    state.goal = TargetSector.from_angle(goal_angle)
    state.corner = TargetSector.from_angle(corner_angle)
    state.turned_corner = last_state is not None and (last_state.turned_corner or (
        np.abs(corner_angle) > np.abs(goal_angle)))
    return state


def angle_between(pose: Pose, point: Point) -> float:
    """Calculate the angle between a pose's heading and a point on the map.

    Parameters
    ----------
    pose : Pose
        The pose whose heading is the base vector.
    point : Point
        The point that forms the angle with the base vector.

    Returns
    -------
    float
        The angle between the pose heading and the point.
    """
    robot_pos = pose.position
    point_angle = np.arctan2(point.y - robot_pos.y, point.x - robot_pos.x)
    robot_heading = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]

    return point_angle - robot_heading
