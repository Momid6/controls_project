#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np




class PidControlNode(Node):
    def __init__(self):
        super().__init__('pid_control_w_tracking')
        #Initializing inner time variable
        self.t = 0.0
        self.dt = 0.01
        self.i = 0
        #Initiializing pervious error variables for derivative calculation
        self.error_prev_x = 0.0
        self.error_prev_y = 0.0
        self.error_prev_z = 0.0
        #Initializing sum variables for integral calculation
        self.sum_x = 0
        self.sum_y = 0
        self.sum_z = 0
        #Setting maxium velocity so that drone does not go too fast
        self.max_vel = 0.7
        #Initializing current position variables [Later I can organize this into a vector]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        #Setting the desired trajectory
        R = 1.0
        omega = 2
        cycles = 2
        points = 500*cycles
        theta_start = np.pi/2
        theta_end = theta_start + cycles * 2 * np.pi
        theta = np.linspace(theta_start, theta_end, points)
        x = R * np.cos(omega*theta)
        y = R * np.sin(2*omega*theta)
        z = np.ones(x.size) * 3.0
        x_goal_list = []
        for i in range(200):
            x_goal_list.append([0.0, 0.0, 3.0])
        for i in range(len(x)):
            x_goal_list.append([x[i], y[i], z[i]])

        self.x_goal = np.array(x_goal_list)
        #Odometer Subscriber
        self.subscriber = self.create_subscription(Odometry, '/crazyflie/odom', self.odometry_callback, 10)
        #Velocity Publisher
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        #Calling the velocity callback every 0.03 seconds
        self.timer = self.create_timer(0.1, self.velocity_callback)
        #This is for showing the path of the drone in RViz
        self.path_pub = self.create_publisher(Path, '/drone_path', 10)
        self.path = Path()
        self.path.header.frame_id = 'map'


    
    def odometry_callback(self, msg):
        #Setting current position variables from the odometry
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        #This is for showing the path of the drone in RViz
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'world'  # same as above
        pose.pose = msg.pose.pose
        self.path.poses.append(pose)
        self.path.header.stamp = pose.header.stamp
        self.path_pub.publish(self.path)
        #Displaying the current position of the drone
        self.get_logger().info(f"Current position: x={self.current_x}, y={self.current_y}, z={self.current_z}")

    def velocity_callback(self):
        #Setting the time step for the PID control
        dt = 0.01
        T_period = 5.0  # seconds for one full cycle
        N_total = len(self.x_goal)
        t_mod = self.t % T_period
        idx = int((t_mod / T_period) * N_total)
        goal = self.x_goal[idx]
        #Calculating the error in position by comparing the current position with the desired position
        self.error_x = goal[0] - self.current_x
        self.error_y = goal[1] - self.current_y
        self.error_z = goal[2] - self.current_z
        #Calculating the integral of the error
        self.sum_x += dt*self.error_x
        self.sum_y += dt*self.error_y
        self.sum_z += dt*self.error_z
        #Clamping the integral values to prevent windup
        self.sum_x = max(min(self.sum_x, 5.0), - 5.0)
        self.sum_y = max(min(self.sum_y, 5.0), - 5.0)
        self.sum_z = max(min(self.sum_z, 5.0), - 5.0)
        #Calculating the derivative of the error
        self.derivative_x = (self.error_x - self.error_prev_x) / dt
        self.derivative_y = (self.error_y - self.error_prev_y) / dt
        self.derivative_z = (self.error_z - self.error_prev_z) / dt
        #Setting cmd to be velocity command
        cmd = Twist()
        #Setting the PID coefficients
        k_p = 0.5
        k_i = 0.05
        k_d = 0.1
        #Calculating the velocity command using PID control
        cmd.linear.x = k_p * self.error_x + k_i * self.sum_x + k_d * self.derivative_x
        cmd.linear.y = k_p * self.error_y + k_i * self.sum_y + k_d * self.derivative_y
        cmd.linear.z = k_p * self.error_z + k_i * self.sum_z + k_d * self.derivative_z
        #If the velocity command is greater than the maximum velocity, clamp it to the maximum velocity
        cmd.linear.x = max(min(cmd.linear.x, self.max_vel), -self.max_vel)
        cmd.linear.y = max(min(cmd.linear.y, self.max_vel), -self.max_vel)
        cmd.linear.z = max(min(cmd.linear.z, self.max_vel), -self.max_vel)
        #Publishing the velocity command
        self.publisher.publish(cmd)
        #Displaying the velocity commands
        self.get_logger().info(f"Received velocity command: {cmd.linear.x}, {cmd.linear.y}, {cmd.linear.z}")
        #Incrementing the time variable
        self.t+= dt
        #Updating the previous error variables for the next iteration for derivative calculation
        self.error_prev_x = self.error_x
        self.error_prev_y = self.error_y
        self.error_prev_z = self.error_z


def main(args=None):
    rclpy.init(args=args)
    node = PidControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()