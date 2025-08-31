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
                    xic, # current state x 
                    xg, # goal state 
                    u_min, # lower bound on u 
                    u_max, # upper bound on u 
                    N_mpc,  # length of MPC window (horizon)
                    ): # return the first control command of the solved policy 
    
    nx,nu = B.shape

        
    Q = np.diag([1., 1., 1.])
    R = np.eye(nu)
    Qf = 10*Q

    X = cp.Variable((nx,N_mpc+1))
    U = cp.Variable((nu,N_mpc))

    obj = 0
    for k in range(N_mpc):
        obj += 0.5*(cp.QuadForm(X[:, k]- xg[k],Q))+ 0.5*cp.QuadForm(U[:, k], R)
    obj += 0.5*(cp.QuadForm(X[:, N_mpc]- xg[-1], Qf))
   


    constraints = [X[:, 0] == xic]
    for k in range(N_mpc):
        constraints += [X[:,k+1] == A@X[:, k] + B@U[:, k], u_min <= U[:, k], u_max >= U[:,k]]
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
        super().__init__('convex_mpc_with_tracking_v3')
        #Initializing inner time variable
        self.t = 0.0
        self.dt = 0.01
        #Initializing current position variables [Later I can organize this into a vector]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        #Setting convex_mpc stuff
        self.u_min = np.array([-0.7, -0.7, -0.7])
        self.u_max = np.array([0.7, 0.7, 0.7])
        self.current_state = np.array([self.current_x, self.current_y, self.current_z])
        
        R = 1.0
        omega = 1
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
        self.N_mpc = 100
        Ac = np.zeros((3,3))
        Bc = np.identity(3)
        nx, nu = Bc.shape
        M = np.zeros((nx + nu, nx + nu))
        M[0:nx, 0:nx] = Ac
        M[0:nx, nx:nx+nu] = Bc
        Md = expm(self.dt * M)
        self.A = Md[0:nx, 0:nx]
        self.B = Md[0:nx, nx:nx+nu]
            
        nx,nu = self.B.shape
        self.Q = np.diag([1., 1., 1.])
        self.R = np.eye(nu)
        self.Qf = 10*self.Q

        self.X = cp.Variable((nx,self.N_mpc+1))
        self.U = cp.Variable((nu,self.N_mpc))
        self.xg = cp.Parameter((nx, self.N_mpc + 1))
        self.xic = cp.Parameter(nx)
        obj = 0
        for k in range(self.N_mpc):
            obj += 0.5*(cp.QuadForm(self.X[:, k]- self.xg[:, k], self.Q))+ 0.5*cp.QuadForm(self.U[:, k], self.R)
        obj += 0.5*(cp.QuadForm(self.X[:, self.N_mpc]- self.xg[:, -1], self.Qf))
    


        constraints = [self.X[:, 0] == self.xic]
        for k in range(self.N_mpc):
            constraints += [self.X[:,k+1] == self.A@self.X[:, k] + self.B@self.U[:, k], self.u_min <= self.U[:, k], self.u_max >= self.U[:,k]]
        self.prob = cp.Problem(cp.Minimize(obj), constraints)
        #Odometer Subscriber
        self.subscriber = self.create_subscription(Odometry, '/crazyflie/odom', self.odometry_callback, 10)
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        #Velocity Publisher
        #Calling the velocity callback every 0.1 seconds
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
        #Setting cmd to be velocity command
        cmd = Twist()
        T_period = 5.0
        N_total = len(self.x_goal)
        start_idx = int(((self.t % T_period) / T_period) * N_total)
        idxs = [(start_idx + i) % N_total for i in range(self.N_mpc + 1)]
        self.xg.value = self.x_goal[idxs].T
        self.xic.value = np.array([self.current_x, self.current_y, self.current_z])
        self.prob.solve(solver=cp.ECOS)
        U_val = self.U.value
        u_mpc = U_val[:,0]
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