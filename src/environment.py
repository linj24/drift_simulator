#!/usr/bin/env python3

import rospy

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from drift_simulator.msg import StateReward

import state

WALL_HEIGHT = 1
WALL_LEN_SHORT = 5
WALL_LEN_LONG = 10
WALL_THICKNESS = 0.2
PATH_WIDTH = 4

GOAL_TOLERANCE = 1
COLLISION_TOLERANCE = 0.5

DIST_TO_WALL = 1
TIME_TO_CHANGE = rospy.Duration.from_sec(1)


import numpy as np
from tf.transformations import quaternion_from_euler


class Environment:

    def __init__(self):

        # initialize this node
        rospy.init_node('drift_environment')

        # db model names
        self.wall_model_names = {
            "s1": "wall_short_1",
            "s2": "wall_short_2",
            "l1": "wall_long_1",
            "l2": "wall_long_2",
            "n": "normal"
        }

        # current wall positions
        self.wall_positions = [
            Point(x=WALL_LEN_SHORT / 2, y=PATH_WIDTH / 2, z=WALL_HEIGHT / 2),
            Point(x=WALL_LEN_LONG / 2, y=-PATH_WIDTH / 2, z=WALL_HEIGHT / 2),
            Point(x=WALL_LEN_SHORT / 2 + WALL_LEN_LONG / 2, y=PATH_WIDTH / 2, z=WALL_HEIGHT / 2),
            Point(x=WALL_LEN_LONG / 2 + WALL_LEN_SHORT / 2, y=-PATH_WIDTH / 2, z=WALL_HEIGHT / 2),
        ]
        self.wall_angles = [0.0, 0.0, np.pi/2, np.pi/2]

        # reset position and orientation of the robot
        self.robot_model_name = "turtlebot3"
        self.robot_reset_position = Point(x=0.0, y=0.0, z=0.0)
        self.robot_reset_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # goal location
        self.goal_position = Point(x = 7, y = -7, z = 0)
        self.corner_position = Point(x = 2, y = -5, z = 0)

        # flag to keep track of the state of when we're resetting the world and when we're not
        # to avoid sending too many duplicate messages
        self.reset_world_in_progress = False

        # reinforcement learning algorithm iteration number
        self.iteration_num = 0
        
        # ROS subscribers
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_received, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.handle_odom, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.handle_scan, queue_size=1)

        # ROS publishers
        self.model_states_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.state_reward_pub = rospy.Publisher("state_reward", StateReward, queue_size=10)

        # Current state
        self.current_odom = None
        self.current_scan = None
        self.last_state = None

    def handle_scan(self, scan: LaserScan) -> None:
        self.current_scan = scan
    
    def handle_odom(self, odom: Odometry) -> None:
        self.current_odom = odom

    def generate_random_corner(self, angle_min: float = -np.pi, angle_max: float = np.pi) -> None:

        # get random angle in range
        angle = np.random.uniform(-np.pi, np.pi)

        # create corner out of 4 or 5 walls
        s1_x = WALL_LEN_SHORT / 2
        s1_y = PATH_WIDTH / 2
        s2_x = s1_x - s1_x * np.cos(angle)
        s2_y = s1_y + s1_x * np.sin(angle)
        n_x = s1_x + PATH_WIDTH * np.cos(angle / 2)
        n_y = s1_y - PATH_WIDTH * np.sin(angle / 2)
        l1_y = -PATH_WIDTH / 2
        if np.abs(angle) < np.pi / 2:
            # add an extra wall for acute angles
            l1_x = n_x - (n_y + l1_y) * np.tan(angle/2) - WALL_LEN_LONG / 2
            l2_x = n_x + (n_y + l1_y) * np.tan(angle/2) - (WALL_LEN_LONG / 2) * np.cos(angle)
            l2_y = n_x + (n_y + l1_y) * np.tan(angle/2) - (WALL_LEN_LONG / 2) * np.sin(angle)
        else:
            l1_x = n_x - WALL_LEN_LONG / 2
            l2_x = n_x - (WALL_LEN_LONG / 2) * np.cos(angle)
            l2_y = n_y + (WALL_LEN_LONG / 2) * np.sin(angle)

        motionless_twist = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))

        wall_xs = [s1_x, s2_x, l1_x, l2_x]
        wall_ys = [s1_y, s2_y, l1_y, l2_y]
        wall_angles = [0, 0, angle, angle]

        if np.abs(angle) < np.pi / 2:
            wall_xs.append(n_x)
            wall_ys.append(n_y)
            wall_angles.append(angle / 2 + np.pi/2)
        else:
            wall_xs.append(15)
            wall_ys.append(0)
            wall_angles.append(0)

        for model_name, x, y, theta in zip(self.wall_model_names.values(), wall_xs, wall_ys, wall_angles):
            pose = Pose(position=Point(x, y, 0), orientation=Quaternion(*quaternion_from_euler(0, 0, theta)))
            self.model_states_pub.publish(ModelState(model_name=model_name, pose=pose, twist=motionless_twist))

        self.wall_positions = [Point(x, y, 0) for x, y in zip(wall_xs, wall_ys)]
        self.wall_angles = wall_angles

        self.goal_position = Point(x=s2_x + np.cos(np.pi/2 - angle) * PATH_WIDTH / 2, y = s2_y + np.sin(np.pi/2 - angle) * PATH_WIDTH / 2, z=0.0)


    def at_goal(self, pose: Pose) -> bool:
        rospy.loginfo(f"Distance: {np.sqrt(((self.goal_position.x - pose.position.x) ** 2 + (self.goal_position.y - pose.position.y) ** 2))}")
        return np.sqrt(((self.goal_position.x - pose.position.x) ** 2 + (self.goal_position.y - pose.position.y) ** 2)) <= GOAL_TOLERANCE

    def collided(self, pose: Pose) -> bool:
        for point, angle in zip(self.wall_positions, self.wall_angles):
            pass
        return False


    def model_states_received(self, data: ModelStates):

        robot_idx = data.name.index("turtlebot3")
        if robot_idx is not None and data.pose is not None:
            robot_pose = data.pose[robot_idx]
            rospy.loginfo(f"At goal: {self.at_goal(robot_pose)}")

            # if the robot has reached the goal, send a reward
            if self.at_goal(robot_pose) and not self.reset_world_in_progress:
                # TODO: Send reward

                if (not self.reset_world_in_progress):

                    self.reset_world_in_progress = True

                    # create new corner
                    # self.generate_random_corner()

                # reset robot position
                p = Pose(position=self.robot_reset_position, orientation=self.robot_reset_orientation)
                t = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))
                robot_model_state = ModelState(model_name=self.robot_model_name, pose=p, twist=t)
                self.model_states_pub.publish(robot_model_state)


            elif self.collided(robot_pose) and not self.reset_world_in_progress:
                pass
            elif (not self.at_goal(robot_pose) and self.reset_world_in_progress):
                self.reset_world_in_progress = False
                self.iteration_num += 1


    def run(self):
        r = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            if self.current_odom is not None and self.current_scan is not None:
                if self.reset_world_in_progress:
                    if self.at_goal(self.current_odom.pose.pose):
                        self.state_reward_pub.publish(StateReward(state = 0, reward = 100, terminal = True))
                    else:
                        self.state_reward_pub.publish(StateReward(state = 1, reward = -1, terminal = True))
                else:
                    s = state.calculate_state(self.current_scan, self.current_odom, DIST_TO_WALL, TIME_TO_CHANGE, self.corner_position, self.goal_position, self.last_state)
                    self.state_reward_pub.publish(StateReward(state = s, reward = 0, terminal = False))
            r.sleep()


if __name__=="__main__":
    node = Environment()
    node.run()