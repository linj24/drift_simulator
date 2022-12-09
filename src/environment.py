#!/usr/bin/env python3

import rospy
import numpy as np

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import LaserScan
from drift_simulator.msg import StateReward

import utils.state as state
from utils.corner import Corner


DIST_TO_WALL = 1
TIME_TO_CHANGE = rospy.Duration(secs=1)
TIMEOUT = rospy.Duration(secs=12)


class Environment:
    """
    Read in the simulator state and convert to a reinforcement learning state.
    """
    def __init__(self):

        # initialize this node
        rospy.init_node("drift_environment")

        # model the corner as a group of walls at an angle
        self.corners = Corner(np.pi / 2)

        # reset position and orientation of the robot
        self.robot_model_name = "turtlebot3"
        self.robot_reset_position = Point(x=0.0, y=0.0, z=0.0)
        self.robot_reset_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # goal location
        self.goal_position = Point(x=7, y=-7, z=0)
        self.corner_position = Point(x=5, y=-2, z=0)

        # flag to avoid sending too many duplicate messages
        self.reset_world_in_progress = False

        # reinforcement learning algorithm iteration number
        self.iteration_num = 0
        self.iteration_start = rospy.Time.now()

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
        """Update the node with the most recent Lidar scan.

        Parameters
        ----------
        scan : LaserScan
            most recent Lidar scan
        """
        self.current_scan = scan

    def handle_model_states(self, data: ModelStates) -> None:
        """Update the node with the most recent Turtlebot3 pose in Gazebo.

        Parameters
        ----------
        data : ModelStates
            most recent Gazebo model states
        """
        robot_idx = data.name.index("turtlebot3")
        if robot_idx is not None and data.pose is not None:
            self.current_robot_pose = data.pose[robot_idx]

    def reset_world(self) -> None:
        """Reset the world by moving the robot back to the start and clearing
        internal state.
        """
        if not self.reset_world_in_progress:

            self.reset_world_in_progress = True

            self.current_robot_pose = None
            self.current_scan = None
            self.last_state = None

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
        """Tick at a fixed rate and publish the current state.
        """
        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():

            # make sure Gazebo has been initialized
            if self.current_robot_pose is not None and self.current_scan is not None:
                iteration_time = rospy.Time.now() - self.iteration_start

                # world reset complete
                if self.reset_world_in_progress and self.corners.at_start(
                    self.current_robot_pose
                ):
                    self.reset_world_in_progress = False
                    self.iteration_num += 1
                    self.iteration_start = rospy.Time.now()

                # robot at goal
                elif self.corners.at_goal(self.current_robot_pose):
                    print(f"GOAL: {iteration_time.to_sec()}")
                    self.state_reward_pub.publish(
                        StateReward(state=state.Terminal.GOAL.id, reward=200 * (TIMEOUT - iteration_time).to_sec(), terminal=True)
                    )
                    self.reset_world()

                # robot crashed
                elif self.corners.collided(self.current_robot_pose):
                    self.state_reward_pub.publish(
                        StateReward(state=state.Terminal.CRASH.id, reward=-1, terminal=True)
                    )
                    self.reset_world()

                # robot timed out
                elif iteration_time > TIMEOUT:
                    self.state_reward_pub.publish(
                        StateReward(state=state.Terminal.TIMEOUT.id, reward=-1, terminal=True)
                    )
                    self.reset_world()

                # nonterminal state
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
                        StateReward(state=s.id, reward=0, terminal=False)
                    )
                    self.last_state = s

            r.sleep()


if __name__ == "__main__":
    node = Environment()
    node.run()
