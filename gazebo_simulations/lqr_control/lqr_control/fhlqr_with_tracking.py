#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.linalg import expm
import numpy as np



def fhlqr(A, # A matrix 
               B, # B matrix 
               Q, # cost weight 
               R, # cost weight 
               Qf,# term cost weight 
               N,   # horizon size 
               ):
        
    # check sizes of everything 
    nx,nu = B.shape
        
    # instantiate S and K 

    P = [np.zeros((nx, nx)) for _ in range(N)]
    K = [np.zeros((nu, nx)) for _ in range(N-1)] 
    
    # initialize S[N] with Qf 
    P[N-1] = np.copy(Qf)
    
    # Ricatti 
    for k in range(N-1, 0, -1):
        K[k-1] = np.linalg.solve(R+B.T@P[k]@B, B.T@P[k]@A)
        P[k-1] = A.T@P[k]@A-(A.T@P[k]@B)@np.linalg.inv(R+B.T@P[k]@B)@(B.T@P[k]@A) + Q
    return P, K 

class FlhqrControlNode(Node):
    def __init__(self):
        super().__init__('fhlqr_with_tracking')
        #Initializing inner time variable
        self.t = 0.0
        self.dt = 0.01
        self.i = 0
        self.tf = 15
        #Setting maxium velocity so that drone does not go too fast
        self.max_vel = 0.7
        #Initializing current position variables [Later I can organize this into a vector]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        #Setting fhlqr stuff
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
        Qf = 10*Q
        #Setting the desired trajectory
        Radius = 1.0
        omega = 2
        cycles = 2
        points = 500*cycles
        theta_start = np.pi/2
        theta_end = theta_start + cycles * 2 * np.pi
        theta = np.linspace(theta_start, theta_end, points)
        x = Radius * np.cos(omega*theta)
        y = Radius * np.sin(2*omega*theta)
        z = np.ones(x.size) * 3.0
        x_goal_list = []
        for i in range(200):
            x_goal_list.append([0.0, 0.0, 3.0])
        for i in range(len(x)):
            x_goal_list.append([x[i], y[i], z[i]])
        self.x_goal = np.array(x_goal_list)
        self.N = len(self.x_goal)
        P, self.K = fhlqr(A,B,Q,R,Qf,self.N)
        #Odometer Subscriber
        self.subscriber = self.create_subscription(Odometry, '/crazyflie/odom', self.odometry_callback, 10)
        #Velocity Publisher
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        #Calling the velocity callback every 0.01 seconds
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
        self.current_pos = np.array([self.current_x, self.current_y, self.current_z])
        dt = 0.01
        T_period = 5.0  # seconds for one full cycle
        N_total = len(self.x_goal)
        t_mod = self.t % T_period
        idx = int((t_mod / T_period) * N_total)
        goal = self.x_goal[idx]
        cmd = Twist()
        if self.i < self.N-1:
            u_lqr = -self.K[idx]@(self.current_pos-goal)
        else:
            u_lqr = np.zeros(3)
        #Calculating the velocity command using PID control
        cmd.linear.x = u_lqr[0]
        cmd.linear.y = u_lqr[1]
        cmd.linear.z = u_lqr[2]
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
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = FlhqrControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()