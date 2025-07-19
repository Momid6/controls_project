# PID CONTROL SIMULATION

## INTRODUCTION
This is sub-project is a simulation of a crazyflie drone in Gazebo using ROS2 PID control code.

## REQUIREMENTS
- [Gazebo Classic](https://classic.gazebosim.org/tutorials?tut=install_ubuntu)
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [Crazyflie firmware and tools](https://www.bitcraze.io/2024/09/crazyflies-adventures-with-ros-2-and-gazebo/)
- Python
- `rosdep`, `colcon`, and other ROS build tools

## HOW TO USE
1. Add the pid_control package to /crazyflie_mapping_demo/ros2_ws/src
2. Open a terminal, source the code, and build the package
3. Open another terminal and do the same 
4. On the first terminal, launch gazebo, RVIZ, and Crazyflie drone through ros2 launch crazyflie_ros2_multiranger_bringup simple_mapper_simulation.launch.py
5. On the second terminal, run the PID control ros2 run pid_control pid_control_w_topic
6. Don't forget to check RVIZ for the drone's path