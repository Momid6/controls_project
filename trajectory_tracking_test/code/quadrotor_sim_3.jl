import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
     

using LinearAlgebra
using PythonPlot
using SparseArrays
using ForwardDiff
using ControlSystems
import Convex as cvx 
using OSQP
using ECOS
using ProgressMeter
using Statistics
     

#Model parameters
g = 9.81 #m/s^2
m = 1.0 #kg 
ℓ = 0.3 #meters
J = 0.2*m*ℓ*ℓ

#Thrust limits
umin = [0.2*m*g; 0.2*m*g]
umax = [0.6*m*g; 0.6*m*g]

h = 0.01  
R = 1.0
omega = 1
cycles = 3
points = 500 * cycles


t_start = 0
t_end = (2π * cycles) / omega
t = range(t_start, t_end, length=points)

x = R * cos.(omega .* t)
y = R * sin.(2 .* omega .* t)

x_goal = hcat([ [x[i], y[i]] for i in 1:length(x) ]...)
Nt = size(x_goal, 2)
x_ref_traj = zeros(6, Nt)
x_ref_traj[1:2, :] .= x_goal



#Planar Quadrotor Dynamics
function quad_dynamics(x,u)
    θ = x[3]
    
    ẍ = (1/m)*(u[1] + u[2])*sin(θ)
    ÿ = (1/m)*(u[1] + u[2])*cos(θ) - g
    θ̈ = (1/J)*(ℓ/2)*(u[2] - u[1])
    
    return [x[4:6]; ẍ; ÿ; θ̈]
end

     

function quad_dynamics_rk4(x,u)
    #RK4 integration with zero-order hold on u
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5*h*f1, u)
    f3 = quad_dynamics(x + 0.5*h*f2, u)
    f4 = quad_dynamics(x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end
     

#Linearized dynamics for hovering
x_hover = zeros(6)
u_hover = [0.5*m*g; 0.5*m*g]
Δumin = umin .- u_hover       
Δumax = umax .- u_hover
A = ForwardDiff.jacobian(x->quad_dynamics_rk4(x,u_hover),x_hover);
B = ForwardDiff.jacobian(u->quad_dynamics_rk4(x_hover,u),u_hover);
     

Nx = 6     # number of state
Nu = 2     # number of controls
thist = range(0, step=h, length=Nt)


# Cost weights
Q = Array(Diagonal([3.0, 3.0, 1.0, 1.0, 1.0, 1.0]))
R = Array(Diagonal([0.1, 0.1]))
    

#LQR Hover Controller
P = dare(A,B,Q,R)
K = dlqr(A,B,Q,R)

     

function ihlqr(A::Matrix,       # vector of A matrices 
               B::Matrix,       # vector of B matrices
               Q::Matrix,       # cost matrix Q 
               R::Matrix;       # cost matrix R 
               max_iter = 1000, # max iterations for Riccati
               tol = 1e-5       # convergence tolerance
               )::Tuple{Matrix, Matrix} # return two matrices 
        
    # get size of x and u from B 
    nx, nu = size(B)
        
    # initialize S with Q
    P = deepcopy(Q)
    K = zeros(nx, nu)
    # Riccati 
    for riccati_iter = 1:max_iter 
        K = (R+B'*P*B)\(B'*P*A)
        P_new = A'*P*A-(A'P*B)*inv(R+B'*P*B)*(B'*P*A) + Q
        if norm(P_new-P) <= tol
            return P, K
        end
        P = P_new
    end
end

function solve_ihlqr(A, B, Q, R, N, x_initial, x_goal, u_hover)
    nx, nu = size(B)
    P, K = ihlqr(A,B,Q,R)
    Xsim_lqr = [zeros(nx) for i = 1:N]
    Xsim_lqr[1] = 1*x_initial
    umin = [0.2*m*g; 0.2*m*g]
    umax = [0.6*m*g; 0.6*m*g]
    for i = 1:N-1
        u_lqr = u_hover - K*(Xsim_lqr[i] -x_goal[:, i])
        if(u_lqr[1] > umax[1])
            u_lqr[1] = umax[1]
        end
        if(u_lqr[2] > umax[2])
            u_lqr[2] = umax[2]
        end
        if(u_lqr[1] < umin[1])
            u_lqr[1] = umin[1]
        end
        if(u_lqr[2] < umin[2])
            u_lqr[2] = umin[2]
        end
        Xsim_lqr[i+1] = quad_dynamics_rk4(Xsim_lqr[i], u_lqr)
    end
    return Xsim_lqr
end

function convex_mpc(A::Matrix, # discrete dynamics matrix A
                    B::Matrix, # discrete dynamics matrix B
                    xic::Vector, # current state x 
                    xg::AbstractMatrix, # goal state
                    u_hover,
                    u_min::Vector, # lower bound on u 
                    u_max::Vector, # upper bound on u 
                    N_mpc::Int64,  # length of MPC window (horizon)
                    )::Vector{Float64} # return the first control command of the solved policy 
    
    nx,nu = size(B)
    
    @assert size(A) == (nx, nx)
    @assert length(xic) == nx 
        
    Q = Array(Diagonal([3.0, 3.0, 1.0, 1.0, 1.0, 1.0]))
    R = Array(Diagonal([0.1, 0.1]))
    Qf = 1*Q


    X = cvx.Variable(nx,N_mpc)
    U = cvx.Variable(nu,N_mpc-1)


    obj = 0
    for k = 1:(N_mpc-1)
        obj += 0.5*cvx.quadform(X[:, k] - xg[:, k], Q)+ 0.5*cvx.quadform(U[:, k], R)
    end
    obj += 0.5*(cvx.quadform(X[:, N_mpc]- xg[:, N_mpc], Qf))
   

    prob = cvx.minimize(obj)

    prob.constraints = vcat(X[:, 1] == xic) 
    for k = 1:N_mpc-1
        prob.constraints = vcat(prob.constraints, X[:,k+1] == A*X[:, k] + B*U[:, k], (u_min) <= U[:, k], (u_max) >= U[:,k])
    end
    
    prob.constraints = vcat(prob.constraints)
    cvx.solve!(prob, ECOS.Optimizer; silent = true)
    X = X.value
    U = U.value
    first_Δu = vec(U[:, 1])
    return u_hover .+ first_Δu
end

function solve_convex_mpc(A,B, x0, xg, u_hover, u_min, u_max, dt, N, N_mpc)
    nx, nu = size(B)
    N_sim = N
    t_vec = 0:dt:((N_sim-1)*dt)
    X_sim = [zeros(nx) for i = 1:N_sim]
    X_sim[1] = x0 
    U_sim = [zeros(nu) for i = 1:N_sim-1]
    X_ref_padded = hcat(xg, repeat(xg[:, end:end], 1, N_mpc))

    @showprogress "simulating" for i = 1:N_sim-1 
        X_ref_tilde = X_ref_padded[:, i:i+N_mpc-1]
        u_mpc = convex_mpc(A, B, X_sim[i], X_ref_tilde, u_hover, u_min, u_max, N_mpc)
        U_sim[i] = u_mpc
        X_sim[i+1] = quad_dynamics_rk4(X_sim[i], U_sim[i])
    end
    return X_sim, U_sim
end



     
     

x0 = [0.0; 0.0;0.0; 0.0; 0.0; 0.0]
xhist_lqr = solve_ihlqr(A, B, Q, R, Nt, x0, x_ref_traj, u_hover)
xhist_lqr = hcat(xhist_lqr...)
xhist_mpc, uhist_mpc = solve_convex_mpc(A, B, x0, x_ref_traj, u_hover, Δumin, Δumax, h, Nt, 20)
xhist_mpc = hcat(xhist_mpc...) 
uhist_mpc = hcat(uhist_mpc...)
figure()
plot(x_ref_traj[1,:], x_ref_traj[2,:], "k--", label="Reference")
plot(xhist_lqr[1,:], xhist_lqr[2,:], label="LQR")
plot(xhist_mpc[1,:], xhist_mpc[2,:], label="MPC")
xlabel("x [m]")
ylabel("y [m]")
legend()
title("Figure-8 Tracking")
pygui(true)
dist_error_mpc = [norm(x_ref_traj[1:2,k] - xhist_mpc[1:2,k]) for k in 1:Nt]
mean_error_mpc = mean(dist_error_mpc)
dist_error_lqr = [norm(x_ref_traj[1:2,k] - xhist_lqr[1:2,k]) for k in 1:Nt]
mean_error_lqr = mean(dist_error_lqr)
tracking_error_mpc = norm(x_ref_traj[1:2, :] - xhist_mpc[1:2, :])
tracking_error_lqr = norm(x_ref_traj[1:2, :] - xhist_lqr[1:2, :])
print("Tracking Error for MPC: ", mean_error_mpc)
print("======")
print("Tracking Error for LQR: ", mean_error_lqr)
# Positions
"""
plot(thist, xhist_lqr[1,:], label="x LQR")
plot(thist, xhist_mpc[1,:], label="x MPC")
xlabel("time")
legend()

plot(thist, xhist_lqr[2,:], label="y LQR")
plot(thist, xhist_mpc[2,:], label="y MPC")
xlabel("time")
legend()

plot(thist, xhist_lqr[3,:], label="θ LQR")
plot(thist, xhist_mpc[3,:], label="θ MPC")
xlabel("time")
legend()


# Controls
plot(thist[1:end-1], uhist_lqr[1,:], label="u1 LQR")
plot(thist[1:end-1], uhist_mpc[1,:], label="u1 MPC")
xlabel("Time")
legend()

plot(thist[1:end-1], uhist_lqr[2,:], label="u2 LQR")
plot(thist[1:end-1], uhist_mpc[2,:], label="u2 MPC")
xlabel("Time")
legend()
"""
#Set up visualization
using MeshCat
using RobotZoo: Quadrotor, PlanarQuadrotor
using CoordinateTransformations, Rotations, Colors, StaticArrays, RobotDynamics



function set_mesh!(vis, model::L;
        scaling=1.0, color=colorant"black"
    ) where {L <: Union{Quadrotor, PlanarQuadrotor}} 
    # urdf_folder = joinpath(@__DIR__, "..", "data", "meshes")
    urdf_folder = @__DIR__
    # if scaling != 1.0
    #     quad_scaling = 0.085 * scaling
    obj = joinpath(urdf_folder, "quadrotor_scaled.obj")
    if scaling != 1.0
        error("Scaling not implemented after switching to MeshCat 0.12")
    end
    robot_obj = MeshFileGeometry(obj)
    mat = MeshPhongMaterial(color=color)
    setobject!(vis["robot"]["geom"], robot_obj, mat)
    if hasfield(L, :ned)
        model.ned && settransform!(vis["robot"]["geom"], LinearMap(RotX(pi)))
    end
end

function visualize!(vis, model::PlanarQuadrotor, x::StaticVector)
    px, py = x[1], x[2]   # simulation x, y
    θ = x[3]
    settransform!(vis["robot"], compose(Translation(px, py, 0), LinearMap(RotX(-θ))))
end



function visualize!(vis, model, tf::Real, X)
    fps = Int(round((length(X)-1)/tf))
    anim = MeshCat.Animation(vis; fps)
    for (k,x) in enumerate(X)
        atframe(anim, k) do
            x = X[k]
            visualize!(vis, model, SVector{6}(x)) 
        end
    end
    setanimation!(vis, anim)
end
     

vis = Visualizer()
model = PlanarQuadrotor()
set_mesh!(vis, model)
render(vis)
     

X1 = [SVector{6}(x) for x in eachcol(xhist_lqr)]
X2 = [SVector{6}(x) for x in eachcol(xhist_mpc)]

visualize!(vis, model, thist[end], X1)
