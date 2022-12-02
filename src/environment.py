#!/usr/bin/env python3

import rospy
import numpy as np

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import LaserScan
from drift_simulator.msg import StateReward

import state
from corner import Corner


DIST_TO_WALL = 1
TIME_TO_CHANGE = rospy.Duration.from_sec(1)


class Environment:
    def __init__(self):

        # initialize this node
        rospy.init_node("drift_environment")

        self.corners = Corner(np.pi / 2)

        # reset position and orientation of the robot
        self.robot_model_name = "turtlebot3"
        self.robot_reset_position = Point(x=0.0, y=0.0, z=0.0)
        self.robot_reset_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # goal location
        self.goal_position = Point(x=7, y=-7, z=0)
        self.corner_position = Point(x=2, y=-5, z=0)

        # flag to keep track of the state of when we're resetting the world and when we're not
        # to avoid sending too many duplicate messages
        self.reset_world_in_progress = False

        # reinforcement learning algorithm iteration number
        self.iteration_num = 0

        # ROS subscribers
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.handle_model_states, queue_size=1,
        )
        rospy.Subscriber("/scan", LaserScan, self.handle_scan, queue_size=1)

        # ROS publishers
        self.model_states_pub = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=10
        )
        self.state_reward_pub = rospy.Publisher(
            "state_reward", StateReward, queue_size=10
        )

        # Current state
        self.current_robot_pose = None
        self.current_scan = None
        self.last_state = None

    def handle_scan(self, scan: LaserScan) -> None:
        self.current_scan = scan

    def handle_model_states(self, data: ModelStates):
        robot_idx = data.name.index("turtlebot3")
        if robot_idx is not None and data.pose is not None:
            self.current_robot_pose = data.pose[robot_idx]

    def reset_world(self):
        if not self.reset_world_in_progress:

            self.reset_world_in_progress = True

            # create new corner
            # self.generate_random_corner()

            # reset robot position
            p = Pose(
                position=self.robot_reset_position,
                orientation=self.robot_reset_orientation,
            )
            t = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0, 0, 0))
            robot_model_state = ModelState(
                model_name=self.robot_model_name, pose=p, twist=t
            )
            self.model_states_pub.publish(robot_model_state)

    def run(self):
        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            if self.current_robot_pose is not None and self.current_scan is not None:
                if self.reset_world_in_progress and self.corners.at_start(
                    self.current_robot_pose
                ):
                    self.reset_world_in_progress = False
                    self.iteration_num += 1
                elif self.corners.at_goal(self.current_robot_pose):
                    self.state_reward_pub.publish(
                        StateReward(state=0, reward=100, terminal=True)
                    )
                    self.reset_world()
                elif self.corners.collided(self.current_robot_pose):
                    self.state_reward_pub.publish(
                        StateReward(state=1, reward=-1, terminal=True)
                    )
                    self.reset_world()
                else:
                    s = state.calculate_state(
                        self.current_scan,
                        self.current_robot_pose,
                        rospy.rostime.get_rostime(),
                        DIST_TO_WALL,
                        TIME_TO_CHANGE,
                        self.corner_position,
                        self.goal_position,
                        self.last_state,
                    )
                    self.state_reward_pub.publish(
                        StateReward(state=s, reward=0, terminal=False)
                    )

            r.sleep()


if __name__ == "__main__":
    node = Environment()
    node.run()
