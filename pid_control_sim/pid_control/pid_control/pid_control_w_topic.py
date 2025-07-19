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
        super().__init__('pid_control_w_topic')
        #Initializing inner time variable
        self.t = 0.0
        #Initiializing pervious error variables for derivative calculation
        self.error_prev_x = 0.0
        self.error_prev_y = 0.0
        self.error_prev_z = 0.0
        #Initializing sum variables for integral calculation
        self.sum_x = 0
        self.sum_y = 0
        self.sum_z = 0
        #Setting maxium velocity so that drone does not go too fast
        self.max_vel = 1.0
        #Initializing current position variables [Later I can organize this into a vector]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        #Setting input
        self.choice = self.declare_parameter("formation")
        self.made_choice = False
        while not self.made_choice:
            self.choice = input("Enter the formation you would like:"
            "\n1. Circle x-y\n2. Circle x-z \n3. Infinity Loop x-y\n4. Infinity loop x-z\n5. Circle Up and Down\n6. Heart\nEnter Value: ")
            if self.choice in ['1', '2', '3', '4', '5', '6']:
                self.made_choice = True
            else:
                print("Invalid choice. Please enter an integer between 1 and 6.")
        #Odometer Subscriber
        self.subscriber = self.create_subscription(Odometry, '/crazyflie/odom', self.odometry_callback, 10)
        #Velocity Publisher
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        #Calling the velocity callback every 0.03 seconds
        self.timer = self.create_timer(0.03, self.velocity_callback)
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
        dt = 0.03
        #Calculating the functions for the desired position based on the choice made
        if self.choice == '1':  # Circle x-y
            function_x = np.cos(0.1*self.t)
            function_y = np.sin(0.1*self.t)
            function_z = 3.0
        elif self.choice == '2':  # Circle x-z
            function_x = np.cos(0.1*self.t)
            function_y = 0.0
            function_z = np.sin(0.1*self.t) + 3.0
        elif self.choice == '3':  # Infinity Loop x-y
            function_x = np.cos(0.05*self.t)
            function_y = np.sin(0.1*self.t)
            function_z = 3.0
        elif self.choice == '4':  # Infinity Loop x-z
            function_x = np.cos(0.05*self.t)
            function_y = 0.0
            function_z = np.sin(0.1*self.t) + 3.0
        elif self.choice == '5':  # Circle Up and Down
            function_x = np.cos(0.1*self.t)
            function_y = np.sin(0.1*self.t)
            function_z = 3.0 + np.power(np.sin(0.3*self.t),2)
        elif self.choice == '6':  # Heart
            function_x = np.power(np.sin(0.1*self.t), 3)
            function_y = (13.0/16.0)*np.cos(0.1*self.t) - (5.0/16.0)*np.cos(0.2*self.t) - (2.0/16.0)*np.cos(0.3*self.t) - (1.0/16.0)*np.cos(0.4*self.t)
            function_z = 3.0
        #Calculating the error in position by comparing the current position with the desired position
        self.error_x = function_x - self.current_x
        self.error_y = function_y - self.current_y
        self.error_z = function_z - self.current_z
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
        if( abs(cmd.linear.x) > self.max_vel):
            if(cmd.linear.x<0):
                cmd.linear.x = -self.max_vel
            else:
                cmd.linear.x = self.max_vel
        if( abs(cmd.linear.y) > self.max_vel):
            if(cmd.linear.y<0):
                cmd.linear.y = -self.max_vel
            else:
                cmd.linear.y = self.max_vel
        if( abs(cmd.linear.z) > self.max_vel):
            if(cmd.linear.z<0):
                cmd.linear.z = -self.max_vel
            else:
                cmd.linear.z = self.max_vel

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