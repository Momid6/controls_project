#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.linalg import expm
import numpy as np
import cvxpy as cp


def convex_mpc(A, # discrete dynamics matrix A
                    B, # discrete dynamics matrix B
                    c,
                    xic, # current state x 
                    xg, # goal state 
                    u_min, # lower bound on u 
                    u_max, # upper bound on u 
                    N_mpc,  # length of MPC window (horizon)
                    ): # return the first control command of the solved policy 
    
    nx,nu = B.shape

        
    Q = np.eye(nx)
    R = np.eye(nu)
    Qf = 10*Q

    c = cp.Constant(c.flatten())
    X = cp.Variable((nx,N_mpc+1))
    U = cp.Variable((nu,N_mpc))

    obj = 0
    for k in range(N_mpc):
        obj += 0.5*(cp.QuadForm(X[:, k]- xg,Q))+ 0.5*cp.QuadForm(U[:, k], R)
    obj += 0.5*(cp.QuadForm(X[:, N_mpc]- xg, Qf))
   


    constraints = [X[:, 0] == xic]
    for k in range(N_mpc):
        constraints += [X[:,k+1] == A@X[:, k] + B@U[:, k] + c, u_min <= U[:, k], u_max >= U[:,k]]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)
    X = X.value
    U = U.value
    if prob.status != cp.OPTIMAL:
        print(f"[ERROR] MPC failed to solve.")
        return np.zeros(nu)
    return U[:,0]


class ConvexMpcControlNode(Node):
    def __init__(self):
        super().__init__('convex_mpc_advanced')
        #Initializing inner time variable
        self.t = 0.0
        self.dt = 0.01
        self.tf = 15
        #Initializing current position variables [Later I can organize this into a vector]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        #Setting convex_mpc stuff
        self.x_goal = np.array([3.0, 3.0, 3.0, 0, 0, 0]) # [x, y, z, vx, vy, vz]
        g = 9.81
        Ac = np.zeros((6,6))
        Ac[0:3,3:6] = np.identity(3)
        Bc = np.zeros((6, 3))
        Bc[0:3, :] = np.identity(3)
        c_before = np.zeros((6,1))
        c_before[5, 0] = -g 
        nx, nu = Bc.shape
        c_rows, c_cols = c_before.shape
        M = np.zeros((nx + nu + c_cols, nx + nu + c_cols))
        M[0:nx, 0:nx] = Ac
        M[0:nx, nx:nx+nu] = Bc
        M[0:c_rows, nx+nu:nx+nu+c_cols] = c_before
        Md = expm(self.dt * M)
        self.A = Md[0:nx, 0:nx]
        self.B = Md[0:nx, nx:nx+nu]
        self.c_after = Md[0:c_rows, nx+nu:nx+nu+c_cols]
        self.u_min = np.array([-0.7, -0.7, -0.7])
        self.u_max = np.array([0.7, 0.7, 0.7])
        self.N_mpc = 50
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
        cmd = Twist()
        self.current_state = np.array([self.current_x, self.current_y, self.current_z, cmd.linear.x, cmd.linear.y, cmd.linear.z])
        u_mpc = convex_mpc(self.A, self.B, self.c_after, self.current_state, self.x_goal, self.u_min, self.u_max, self.N_mpc)
        print(u_mpc)
        #Calculating the velocity command using PID control
        cmd.linear.x = u_mpc[0]
        cmd.linear.y = u_mpc[1] 
        cmd.linear.z = u_mpc[2]
        #Publishing the velocity command
        self.publisher.publish(cmd)
        #Displaying the velocity commands
        self.get_logger().info(f"Received velocity command: {u_mpc[0]}, {u_mpc[1]}, {u_mpc[2]}")
        #Incrementing the time variable
        self.t+= self.dt

def main(args=None):
    rclpy.init(args=args)
    node = ConvexMpcControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()