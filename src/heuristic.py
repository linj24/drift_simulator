#!/usr/bin/env python3

from __future__ import annotations

import rospy

from std_msgs.msg import UInt8

from drift_simulator.msg import StateReward

from action import Turn
from utils.state import State, NonTerminal, ObstacleSector, TargetSector

def policy(state_id: int) -> int | None:
    s = State(state_id).state
    # rospy.loginfo(f"State: {s}")
    if isinstance(s, NonTerminal):
        a = Turn.STRAIGHT

        # avoid collisions at all costs
        if s.within_dist:
            if s.closest == ObstacleSector.RIGHT:
                a = Turn.LEFT
            elif s.closest == ObstacleSector.LEFT:
                a = Turn.RIGHT

        # approaching turn
        elif not s.turned_corner:
            if s.corner == TargetSector.BOT_RIGHT:
                a = Turn.RIGHT
            elif s.corner == TargetSector.BOT_LEFT:
                a = Turn.LEFT

        # after turn
        else:
            if s.goal == TargetSector.TOP_RIGHT or s.goal == TargetSector.BOT_RIGHT:
                a = Turn.RIGHT
            elif s.goal == TargetSector.TOP_LEFT or s.goal == TargetSector.BOT_LEFT:
                a = Turn.LEFT
        return a.value
    return None

class CornerHeuristic:
    def __init__(self):
        rospy.init_node("corner_heuristic")
        self.action_pub = rospy.Publisher("/action", UInt8, queue_size=10)
        rospy.Subscriber(
            "/state_reward", StateReward, self.handle_state_reward, queue_size=1
        )

    def handle_state_reward(self, data: StateReward):

            # rospy.loginfo(f"Action: {a}")
            a = policy(data.state)

            self.action_pub.publish(a)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = CornerHeuristic()
    node.run()
