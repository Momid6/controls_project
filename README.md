# Welcome to my Controls Project Repository

Note: Since I am focusing more on theory, in order to save time, I used high-level commands to control the drone (Controlling the drone through velocity). I acknowledge that through the complete dynamics of a drone, difference would be more profound, and results may even look completely different.

## Introduction
This repository serves as a collection of my work done learning control theory during the summer of 2025. Through it, I provide insights into how I approached and understood key concepts in control theory. You can view my week to week progress over here.

## Progress
I'll document my progress through phases.

### Phase 1 (19-25 May): Learning ROS2
The major focus of this phase was learning ROS 2 (Robot Operating System 2), which I explored through a combination of official ROS 2 tutorials, YouTube videos, and  experimentation.

During this process, I learned (this list is non-exhaustive and generalized):

- How to create ROS2 packages

- How to write a publisher/subscriber nodes through topcis

- How to write service/client nodes

- How to write action server and client nodes

- The structure and usage of interfaces (msg's, srv's, and actions)

- How to create and work with parameters in ROS2

- How to create and work with launch files

- How to use RVIZ

### Phase 2 (26 May - 2 June): Application
During this phase, I got to learn about and use Gazebo for physics-accurate simulation and Crazyflie. Using my prior knowledge of PID control, I implemented a closed-loop controller to test my understanding of ROS 2 in practice.

What I worked on during this phase:

- Designed a PID controller

- Integrated it with a ROS 2 Python node to control the Crazyflie drone

- Published velocity commands and subscribed to position feedback

- Tuned PID parameters and trajectory angular velocity for stable trajectory tracking

- Visualized drone path using predictions (through RK4 simulation plots) and RViz (real time)

### Phase 3 (3 June - 16 July): Theory
During this phase, I closely followed the Optimal Control (16-745) course from Carnegie Mellon University. I worked through lecture content and completed several homework assignments. This stage deepened my understanding of control theory, optimization, and trajectory planning.

Key topics and implementations (This list again is non-exhaustive but provides an outlook to how I deepend my understanding of the topics discussed below):

Dynamics:
I learned about the different ways dynamics problems are posed in control theory. For example, I explored formulations where the dynamics are expressed explicitly in terms of control inputs (e.g., ẋ = f(x, u)), as well as alternative formulations such as Manipulator Dynamics Equation.

Optimization:
I worked on learning and solving optimization problems. First, I learned how to optomize an unconstrained problem through just taking Newton's method with respect to the gradient. Then, I learned how to optimize equality constrained problems through simple KKT coniditons using the Lagrangian. I passed these KKT conditions to Newton's method to solve for new states and multipliers. Lastly, I learned how to solve generally constrained problems (equality and non-equality) through the interior point method, KKT conditions, and Newton's method. To make my Newton steps more effecient, I used Armijo's rule (line searching) and Hessian regularization.

Discretization: 
I learned about how to discretize dynamics problems (which is useful for later control problems) through two pathways: linear and nonlinear. For linear systems, I used the matrix exponential method to discretize continuous-time dynamics of the form ẋ = Ax + Bu. For nonlinear systems, I used Runge-Kutta 4 integration to approximate the discrete dynamics

Linearization:
I learned how to linearize nonlinear systems around a point. This process involves computing the Jacobian matrices of the system dynamics with respect to the state and control variables. Using these matrices, you can approximate your nonlinear system about a point. A downside to this is that if your state is far from the equilibrium point chosen, then the approximation becomes less accurate. This linearization step was important for applying LQR to nonlinear systems.

Controllability:
I studied the concept of controllability, which basically shows whether you have control over the entire system or not. The concept is simple, you would have to construct the controllability matrix C = [B AB A^2B ...] which can be thought of as moving forward with the dynamics. If the matrix if full rank (meaning its columns are linearly independent), then the system is controllable and you can influence all state variables independently through your controls. If not, then at least one column can be expressed as a linear combination of others, implying that some states cannot be influenced independently. This means you cannot fully control the system.

Linear Quadratic Regulator (LQR):
Studied and implemented both infinite- and finite-horizon LQR for linear systems. Learned how to derive the Riccati recursion through Bellman's principle  of optimality and Dynamic prorgamming. I utilized it to solve real world problems such as space docking and the cartpole problem.

Quadratic Programming (QP):
Formulated constrained optimization problems and solved them using primal-dual interior point methods. Applied this to control scenarios requiring state/input constraints.

Convex Model Predictive Control (MPC):
I explored MPC and solved the space docking problem through MPC. To implement convex MPC, it is crucial that both the cost function and the constraints are convex, ensuring the optimization problem remains convex. To start, I solved problems by using convex solvers, such as ECOS, to solve for a certain horizon (within the full horizon). After that, I only applied the first optimal control sequence to the next step and simulated forward. This method provides the advantage of using an effecient convex solver while maintaining accuracy through resembling a closed-loop control method. 

### Phase 4 (17 July - 31 August): Application Pt. 2
In this phase, I implemented and compared multiple controllers for my ROS 2 Crazyflie drone simulation: PID, Infinite-Horizon LQR (IHLQR), Finite-Horizon LQR (FHLQR), and two forms of Model Predictive Control (MPC): a CVXPY convex MPC and TinyMPC.

Each controller was tested on the same 3D trajectory (a figure-eight pattern at a fixed altitude) using real-time feedback from simulated odometry, and performance was evaluated using tracking error, and control smoothness.


Controllers implemented:

    PID Controller (pid_control_w_tracking.py):

            Classic proportional–integral–derivative loop with velocity clamping

            Tuned gains: k_p = 0.5, k_i = 0.05, k_d = 0.1

            Pros: Simple, very fast; Cons: Less accurate tracking, no explicit handling of constraints, jagged trajectory/not smooth controls

    Finite-Horizon LQR (fhlqr_with_tracking.py):

            Uses backward Riccati recursion for a set number of steps

            Allows time-varying feedback gains for trajectory tracking

            Pros: Improved accuracy over PID, smoother controls, closed-loop, controls update online; Cons: Expensive precomputation required, no handling of constraints (needed to clamp velocity)

    Infinite-Horizon LQR (ihlqr_with_tracking.py):

            Solves the algebraic Riccati equation iteratively until convergence

            Produces constant gain matrix for optimal linear control

            Pros: Improved accuracy over PID, smoother controls, closed-loop, calculations were made offline while the controls update was only online (computaitonally cheap and fast), no need to access different gains; Cons: No handling of constraints (needed to clamp velocity).


    Convex MPC (convex_mpc_with_tracking_v3.py):

            Custom implementation using CVXPY with an ECOS solver

            Formulates trajectory tracking as a constrained quadratic program through solving an optimization problem and taking the first step

            Pros: Handles state/control constraints

    TinyMPC (tinmypc.py):

            Implemented through user friendly tinympc

            Pros: Maintains most of convex MPC’s advantages but with improved solve speed Cons: Worse tracking (I'll look into this more later)

    Key findings from comparison:

            Tracking Accuracy: Convex MPC (CVXPY) had the lowest tracking accuracy, followed by (surprisngly) TinyMPC, IHLQR, FHLQR. PID came last.

            Control Smoothness: MPC and LQR produced smoother velocity profiles compared to the more aggressive corrections from PID.

            Constraint Handling: Only MPC methods explicitly enforced control limits in optimization.

            By the end of this phase, I had a complete comparative framework for evaluating controllers on the same trajectory-tracking task.
