#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Accel
from scipy.linalg import expm
import numpy as np
import cvxpy as cp


def convex_trajopt(A,      # A matrix 
                        B,      # B matrix 
                        Q,      # cost weight 
                        R,      # cost weight 
                        Qf,     # term cost weight 
                        N,       # horizon size 
                        u_min, # lower bound on u
                        u_max, # upper bound on u
                        x_ic,   # initial condition
                        x_goal # goal state
                        ):
     
    nx,nu = B.shape

    X = cp.Variable((nx, N+1))
    U = cp.Variable((nu, N))
    
    cost = 0 
    for k in range(N):
        current_x = 0.5*cp.QuadForm(X[:,k]-x_goal,Q)
        current_u = 0.5*cp.QuadForm(U[:,k],R)
        cost += current_x + current_u
    
    cost += 0.5*cp.QuadForm(X[:,N]-x_goal,Qf)
    

    constraints = [X[:, 0] == x_ic]
    for k in range(N):
        constraints += [X[:,(k+1)] == A@X[:,k]+B@U[:,k], u_min <= U[:, k], u_max >= U[:,k]]


    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)
    
    
    X = X.value
    U = U.value
    
    return X, U 

class ConvexControlNode(Node):
    def __init__(self):
        super().__init__('convex_mpc')
        #Initializing inner time variable
        self.t = 0.0
        self.dt = 0.01
        self.tf = 15
        self.initial_poision_initialized = False
        self.i = 0
        #Odometer Subscriber
        self.subscriber = self.create_subscription(Odometry, '/crazyflie/odom', self.odometry_callback, 10)
        self.subscriber_2 = self.create_subscription(Odometry, '/crazyflie/odom', self.initial_position, 10)
        #Velocity Publisher
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        #Calling the velocity callback every 0.01 seconds
        self.timer = self.create_timer(0.01, self.velocity_callback)
        #This is for showing the path of the drone in RViz
        self.path_pub = self.create_publisher(Path, '/drone_path', 10)
        self.path = Path()
        self.path.header.frame_id = 'map'

    def initial_position(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        self.current_pos = np.array([self.current_x, self.current_y, self.current_z])
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
        self.u_min = np.array([-0.7, -0.7, -0.7])
        self.u_max = np.array([0.7, 0.7, 0.7])
        Q = np.eye(nx)
        R = np.eye(nu)
        Qf = 10*Q
        t_vec = np.arange(0, self.tf+self.dt, self.dt)
        self.N = t_vec.size
        Xcvx, self.Ucvx = convex_trajopt(A, B, Q, R, Qf, self.N, self.u_min, self.u_max, np.array([self.current_x, self.current_y, self.current_z]), self.x_goal)
        self.initial_poision_initialized = True
        self.get_logger().info(f"Initial position set: x={self.current_x}, y={self.current_y}, z={self.current_z}")
        self.destroy_subscription(self.subscriber_2)


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
        if not self.initial_poision_initialized:
            self.get_logger().warn("Initial position not set yet. Waiting...")
            return
        cmd = Twist()
        if self.i < self.N-1:
            u_mpc = self.Ucvx[:, self.i]
        else:
            u_mpc = np.zeros(3)
        #Calculating the velocity command using PID control
        cmd.linear.x = u_mpc[0]
        cmd.linear.y = u_mpc[1]
        cmd.linear.z = u_mpc[2]
        #Publishing the velocity command
        self.publisher.publish(cmd)
        #Displaying the velocity commands
        self.get_logger().info(f"Received velocity command: {cmd.linear.x}, {cmd.linear.y}, {cmd.linear.z}")
        #Incrementing the time variable
        self.t+= self.dt
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = ConvexControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()