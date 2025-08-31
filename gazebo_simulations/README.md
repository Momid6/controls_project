# CONTROL METHODS SIMULATIONS

## INTRODUCTION
This sub-project includes simulations of a crazyflie drone in Gazebo using ROS2. It involves PID, finite- and infite-horizon LQR, convex, convex_mpc, and TinyMPC control methods. 

## REQUIREMENTS
- [Gazebo Classic](https://classic.gazebosim.org/tutorials?tut=install_ubuntu)
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [Crazyflie firmware and tools](https://www.bitcraze.io/2024/09/crazyflies-adventures-with-ros-2-and-gazebo/)
- Python
- `rosdep`, `colcon`, and other ROS build tools

## HOW TO USE
1. Add the XYZ_control package to /crazyflie_mapping_demo/ros2_ws/src
2. Open a terminal, source the code, and build the package
3. Open another terminal and do the same 
4. On the first terminal, launch gazebo, RVIZ, and Crazyflie drone through this launch file: `ros2 launch crazyflie_ros2_multiranger_bringup simple_mapper_simulation.launch.py`
5. On the second terminal, run the control method you want `ros2 run XYZ_control XYZ_node_name` (check the pertinent node name in the packages)
6. Don't forget to check RVIZ for the drone's path

## MORE INSTRUCTIONS
For tracking files, visit:
    - Convex_mpc: "Controls Project\gazebo_simulations\convex_mpc\convex_mpc\convex_mpc_with_tracking_v3.py"
    - FHLQR: "Controls Project\gazebo_simulations\lqr_control\lqr_control\ihlqr_with_tracking.py"
    - IHLQR: "Controls Project\gazebo_simulations\lqr_control\lqr_control\fhlqr_with_tracking.py"
    - PID: "Controls Project\gazebo_simulations\pid_control\pid_control\pid_control_w_tracking.py"
    - TinyMPC: "Controls Project\gazebo_simulations\tinympc\tinympc\tinmypc.py"