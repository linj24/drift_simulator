#!/usr/bin/env python3

import rospy

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

WALL_HEIGHT = 1
WALL_LEN_SHORT = 5
WALL_LEN_LONG = 10
WALL_THICKNESS = 0.2
PATH_WIDTH = 4

GOAL_TOLERANCE = 1

import numpy as np
from tf.transformations import quaternion_from_euler


class ResetWorld(object):

    def __init__(self):

        # initialize this node
        rospy.init_node('reset_world_drift')

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
        self.wall_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # reset position and orientation of the robot
        self.robot_model_name = "robot"
        self.robot_reset_position = Point(x=0.0, y=0.0, z=0.0)
        self.robot_reset_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # goal location
        self.goal_position = Point(x = WALL_LEN_LONG / 2 + WALL_LEN_SHORT / 2, y = 0, z = 0)

        # flag to keep track of the state of when we're resetting the world and when we're not
        # to avoid sending too many duplicate messages
        self.reset_world_in_progress = False

        # reinforcement learning algorithm iteration number
        self.iteration_num = 0
        
        # ROS subscribers
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_received)

        # ROS publishers
        self.model_states_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)


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

        s1_o = quaternion_from_euler(0, 0, 0)
        l1_o = quaternion_from_euler(0, 0, 0)
        s2_o = quaternion_from_euler(0, 0, angle)
        l2_o = quaternion_from_euler(0, 0, angle)
        n_0 = quaternion_from_euler(0, 0, angle / 2 + np.pi/2)

        motionless_twist = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))

        m_s1 = ModelState(model_name=self.wall_model_names["s1"], pose=Pose(position=Point(s1_x, s1_y, 0), orientation=s1_o), twist=motionless_twist)
        m_s2 = ModelState(model_name=self.wall_model_names["s2"], pose=Pose(position=Point(s2_x, s2_y, 0), orientation=s2_o), twist=motionless_twist)
        m_l1 = ModelState(model_name=self.wall_model_names["l1"], pose=Pose(position=Point(l1_x, l1_y, 0), orientation=l1_o), twist=motionless_twist)
        m_l2 = ModelState(model_name=self.wall_model_names["l2"], pose=Pose(position=Point(l2_x, l2_y, 0), orientation=l2_o), twist=motionless_twist)

        self.model_states_pub.publish(m_s1)
        self.model_states_pub.publish(m_s2)
        self.model_states_pub.publish(m_l1)
        self.model_states_pub.publish(m_l2)


    def at_goal(self, pose: Pose) -> bool:
        return ((self.goal_position.x - pose.position.x) ** 2 + (self.goal_position.y - pose.position.y) ** 2) <= GOAL_TOLERANCE


    def model_states_received(self, data: ModelStates):

        robot_idx = data.name.index("robot")
        if robot_idx is not None:
            robot_pose = data.pose[robot_idx].position

            # if the robot has reached the goal, send a reward
            if self.at_goal(robot_pose):
                # TODO: Send reward

                if (not self.reset_world_in_progress):

                    self.reset_world_in_progress = True

                    # create new corner
                    self.generate_random_corner()

                # reset robot position
                p = Pose(position=self.robot_reset_position, orientation=self.robot_reset_orientation)
                t = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))
                robot_model_state = ModelState(model_name=self.robot_model_name, pose=p, twist=t)
                self.model_states_pub.publish(robot_model_state)


            elif (not self.at_goal(robot_pose) and self.reset_world_in_progress):
                self.reset_world_in_progress = False
                self.iteration_num += 1


    def run(self):
        rospy.spin()


if __name__=="__main__":
    node = ResetWorld()
    node.run()