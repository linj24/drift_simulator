from sensor_msgs.msg import LaserScan

N_ACTIONS = 3
N_STATES = 1 << 12

class State:

    corner: int


    @property
    def id(self) -> int:
        pass

def scan_to_state(scan: LaserScan) -> State:
    pass
