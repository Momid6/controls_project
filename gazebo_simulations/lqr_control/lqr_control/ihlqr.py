#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.linalg import expm
import numpy as np
from scipy.linalg import expm

def ihlqr(A, B, Q, R, max_iter: int= 1000, tol = 1e-5):
    # get size of x and u from B 
    nx, nu = B.shape
        
    # initialize S with Q
    P = np.copy(Q)
    K = np.zeros((nx, nu))
    # Riccati 
    for riccati_iter in range(max_iter+1): 
        K = np.linalg.solve((R+B.T@P@B),(B.T@P@A))
        P_new = A.T@P@A-(A.T@P@B)@np.linalg.inv(R+B.T@P@B)@(B.T@P@A) + Q
        if np.linalg.norm(P_new-P) <= tol:
            return P, K
        P = P_new


class IhlqrControlNode(Node):
    def __init__(self):
        super().__init__('ihlqr')
        #Initializing inner time variable
        self.t = 0.0
        self.dt = 0.01
        self.tf = 15
        #Setting maxium velocity so that drone does not go too fast
        self.max_vel = 0.7
        #Initializing current position variables [Later I can organize this into a vector]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        #Setting fhlqr stuff
        self.x_goal = np.array([3.0, 3.0, 3.0]) # [x, y, z]
        Ac = np.zeros((3,3))
        Bc = np.identity(3)
        nx, nu = Bc.shape
        M = np.zeros((nx + nu, nx + nu))
        M[0:nx, 0:nx] = Ac
        M[0:nx, nx:nx+nu] = Bc
        Md = expm(self.dt * M)
        A = Md[0:nx, 0:nx]
        B = Md[0:nx, nx:nx+nu]
        nx, nu = B.shape
        Q = np.eye(nx)
        R = np.eye(nu)
        max_iter = 10000
        P, self.K = ihlqr(A, B, Q, R, max_iter, tol = 1e-5)
        #Odometer Subscriber
        self.subscriber = self.create_subscription(Odometry, '/crazyflie/odom', self.odometry_callback, 10)
        #Velocity Publisher
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        #Calling the velocity callback every 0.01 seconds
        self.timer = self.create_timer(0.01, self.velocity_callback)
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
        #Setting cmd to be velocity command
        self.current_pos = np.array([self.current_x, self.current_y, self.current_z])
        cmd = Twist()
        u_lqr = -self.K@(self.current_pos-self.x_goal)
        #Calculating the velocity command using PID control
        cmd.linear.x = u_lqr[0]
        cmd.linear.y = u_lqr[1]
        cmd.linear.z = u_lqr[2]
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
        self.t+= self.dt

def main(args=None):
    rclpy.init(args=args)
    node = IhlqrControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()