# Drift Simulator: High-Speed Cornering for Mobile Robots

All code was written by me.

## Directory Structure
This repository follows the general directory structure for a ROS package.

- `checkpoints`: Contains .csv files for various models. Each model has its own
  folder, inside which there are files for the policy, q-matrix, and measured
  metrics.
  
- `launch`: Launch files to run combinations of nodes simultaneously. Requires
  ROS (Robot Operating System) to use `roslaunch`, the command-line tool to run
  these files.

- `models`: .dae mesh files generated by Blender for the walls in the
  simulation. Two lengths are provided: a long wall 10 meters long, and a short
  wall 5 meters long. Unfortunately, the folder name `models` is required for
  Gazebo to read the files.

- `msg`: .msg files representing custom messages that get passed between ROS
  nodes. Only one custom message is used in this project: `StateReward`, which
  contains a state ID, a reward, and a boolean representing whether the state is
  terminal.
  
- `plots`: Python scripts for plotting the metrics recorded in the `checkpoints`
  folder, as well as the .png plots themselver.
  - `four_way_hist.py`
  - `policy_updates.py`

- `src`: The source code for the ROS nodes. This folder contains all of my code
  (except for the plot utils).

- `worlds`: .world files representing a Gazebo simulation environment. A single
  world, `turtlebot3_drift.world`, is used for the cornering task; it loads in
  the walls and is reused upon the world reset.

- `CMakeLists.txt` and `package.xml`: Files required for the ROS package and
  messages to compile.