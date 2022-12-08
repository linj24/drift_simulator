from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from tf.transformations import quaternion_from_euler

WALL_HEIGHT = 1
WALL_LEN_SHORT = 5
WALL_LEN_LONG = 10
WALL_THICKNESS = 0.2
PATH_WIDTH = 4
WALL_LENS = [
    WALL_LEN_SHORT,
    WALL_LEN_SHORT,
    WALL_LEN_LONG,
    WALL_LEN_LONG,
    WALL_LEN_SHORT,
]

GOAL_TOLERANCE = 1
COLLISION_TOLERANCE = 0.2


@dataclass
class Wall:
    model_name: str
    length: float
    width: float
    center: Point = Point()
    angle: float = 0.0

    def move(self, center: Point, angle: float):
        self.center = center
        self.angle = angle
        return self

    def collided(self, pos: Point) -> bool:
        pos_to_center = np.array([self.center.x - pos.x, self.center.y - pos.y])
        wall_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
        wall_proj = np.dot(wall_vec, pos_to_center) * wall_vec
        dist_par = np.sqrt(np.dot(wall_proj, wall_proj))
        dist_vec = pos_to_center - wall_proj
        dist_perp = np.sqrt(np.dot(dist_vec, dist_vec))
        return dist_perp < COLLISION_TOLERANCE and dist_par < self.length / 2 + COLLISION_TOLERANCE

    @property
    def model_state(self) -> ModelState:
        return ModelState(
            model_name=self.model_name,
            pose=Pose(
                position=self.center,
                orientation=Quaternion(*quaternion_from_euler(0, 0, self.angle)),
            ),
            twist=Twist(),
        )


class Corner:
    s1: Wall
    s2: Wall
    l1: Wall
    l2: Wall
    n: Wall
    goal: Point

    def __init__(self, angle: float):
        self.s1 = Wall("wall_short_1", WALL_LEN_SHORT, WALL_THICKNESS)
        self.s2 = Wall("wall_short_2", WALL_LEN_SHORT, WALL_THICKNESS)
        self.l1 = Wall("wall_long_1", WALL_LEN_LONG, WALL_THICKNESS)
        self.l2 = Wall("wall_long_2", WALL_LEN_LONG, WALL_THICKNESS)
        self.n = Wall("normal", WALL_LEN_SHORT, WALL_THICKNESS)
        self.move(angle)

    def move(self, angle: float):
        # create corner out of 4 or 5 walls
        s1_x = self.s1.length / 2
        s1_y = np.sign(np.sin(angle)) * (- PATH_WIDTH / 2)
        s2_x = self.s1.length + (self.s2.length / 2) * np.cos(angle)
        s2_y = s1_y - (self.s2.length / 2) * np.sin(angle)
        l1_y = -s1_y
        if np.abs(angle) < np.pi / 2:
            # add an extra wall for acute angles
            n_x = self.s1.length + PATH_WIDTH * np.cos(angle / 2)
            n_y = s1_y - PATH_WIDTH * np.sin(angle / 2)
            l1_x = n_x - (n_y + l1_y) * np.tan(angle / 2) - self.l1.length / 2
            l2_x = (
                n_x
                + (n_y + l1_y) * np.tan(angle / 2)
                - (self.l1.length / 2) * np.cos(angle)
            )
            l2_y = (
                n_x
                + (n_y + l1_y) * np.tan(angle / 2)
                - (self.l1.length / 2) * np.sin(angle)
            )
            n_angle = angle / 2 + np.pi / 2
        else:
            l1_x = self.s1.length + PATH_WIDTH * np.tan((np.pi - angle) / 2) - (self.l1.length / 2)
            l2_x = self.s1.length + PATH_WIDTH * np.tan((np.pi - angle) / 2) + (self.l2.length / 2) * np.cos(angle)
            l2_y = l1_y - (self.l2.length / 2) * np.sin(angle)
            n_x = 15
            n_y = 0
            n_angle = 0

        self.s1.move(Point(s1_x, s1_y, 0), 0)
        self.s2.move(Point(s2_x, s2_y, 0), angle)
        self.l1.move(Point(l1_x, l1_y, 0), 0)
        self.l2.move(Point(l2_x, l2_y, 0), angle)
        self.n.move(Point(n_x, n_y, 0), n_angle)

        self.goal_position = Point(
            x=s2_x + np.cos(np.pi / 2 - angle) * PATH_WIDTH / 2,
            y=s2_y + np.sin(np.pi / 2 - angle) * PATH_WIDTH / 2,
            z=0.0,
        )

    @property
    def walls(self) -> list[Wall]:
        return [self.s1, self.s2, self.l1, self.l2, self.n]

    def at_start(self, pose: Pose) -> bool:
        return (
            np.sqrt(((pose.position.x) ** 2 + (pose.position.y) ** 2)) <= GOAL_TOLERANCE
        )

    def at_goal(self, pose: Pose) -> bool:
        return (
            np.sqrt(
                (
                    (self.goal_position.x - pose.position.x) ** 2
                    + (self.goal_position.y - pose.position.y) ** 2
                )
            )
            <= GOAL_TOLERANCE
        )

    def collided(self, pose: Pose) -> bool:
        robot_pos = pose.position
        return any([wall.collided(robot_pos) for wall in self.walls])

if __name__ == "__main__":
    c = Corner(np.pi/2)