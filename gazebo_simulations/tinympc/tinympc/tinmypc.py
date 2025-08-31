#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.linalg import expm
import numpy as np
import tinympc




class ConvexMpcControlNode(Node):
    def __init__(self):
        super().__init__('tinympc_node')
        #Initializing inner time variable
        self.t = 0.0
        self.dt = 0.01
        self.i = 0
        #Initializing current position variables [Later I can organize this into a vector]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        #Setting convex_mpc stuff
        self.u_min = np.array([-0.7, -0.7, -0.7])
        self.u_max = np.array([0.7, 0.7, 0.7])
        self.current_state = np.array([self.current_x, self.current_y, self.current_z])
        
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
        self.Q = np.diag([1., 1., 1.])
        self.R = np.eye(nu)

        self.prob = tinympc.TinyMPC()
        self.prob.setup(self.A, self.B, self.Q, self.R, self.N_mpc, u_min=self.u_min, u_max=self.u_max)
        
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
        self.current_state = np.array([self.current_x, self.current_y, self.current_z])

        self.prob.set_x0(self.current_state)
        T_period = 5.0
        N_total = len(self.x_goal)
        start_idx = int(((self.t % T_period) / T_period) * N_total)
        idxs = [(start_idx + i) % N_total for i in range(self.N_mpc)] 
        self.xg = self.x_goal[idxs].T 
        self.prob.set_x_ref(self.xg)
        u_mpc = self.prob.solve()
        #Calculating the velocity command using mpc control
        cmd.linear.x = u_mpc['controls'][0]
        cmd.linear.y = u_mpc['controls'][1]
        cmd.linear.z = u_mpc['controls'][2]
        #Publishing the velocity command
        self.publisher.publish(cmd)
        #Displaying the velocity commands
        self.get_logger().info(f"Received velocity command: {u_mpc['controls'][0]}, {u_mpc['controls'][1]}, {u_mpc['controls'][2]}")
        #Incrementing the time variable
        self.t+= self.dt
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = ConvexMpcControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()