import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
     

using LinearAlgebra
using PythonPlot
using SparseArrays
using ForwardDiff
using ControlSystems
using OSQP
     

#Model parameters
g = 9.81 #m/s^2
m = 1.0 #kg 
ℓ = 0.3 #meters
J = 0.2*m*ℓ*ℓ

#Thrust limits
umin = [0.2*m*g; 0.2*m*g]
umax = [0.6*m*g; 0.6*m*g]

h = 0.01           # timestep
R = 1.0
omega = 1      # angular speed (rad / s)
cycles = 3
points = 500 * cycles

# choose time vector t so that omega * t goes 0/2 -> 2π*cycles
t_start = 0
t_end = (2π * cycles) / omega
t = range(t_start, t_end, length=points)

x = R * cos.(omega .* t)
y = R * sin.(2 .* omega .* t)

x_goal = hcat([ [x[i], y[i]] for i in 1:length(x) ]...)
Nt = size(x_goal, 2)
x_ref_traj = zeros(6, Nt)
x_ref_traj[1:2, :] .= x_goal
# velocities and yaw remain zero


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
A = ForwardDiff.jacobian(x->quad_dynamics_rk4(x,u_hover),x_hover);
B = ForwardDiff.jacobian(u->quad_dynamics_rk4(x_hover,u),u_hover);
quad_dynamics_rk4(x_hover, u_hover)
     

Nx = 6     # number of state
Nu = 2     # number of controls
Tfinal = 10.0 # final time
Nt = size(x_ref_traj, 2)     
thist = range(0, step=h, length=Nt)
     

# Cost weights
Q = Array(Diagonal([3.0, 3.0, 1.0, 1.0, 1.0, 1.0]))
R = Array(Diagonal([0.1, 0.1]))
Qn = 1*Q
     

#Cost function
function cost(xhist,uhist, x_ref)
    cost = 0.5*(xhist[:,end]-x_ref[:,end])'*Qn*(xhist[:,end]-x_ref[:,end])
    for k = 1:(size(xhist,2)-1)
        cost = cost + 0.5*(xhist[:,k]-x_ref[:, k])'*Q*(xhist[:,k]-x_ref[:, k]) + 0.5*(uhist[k]'*R*uhist[k])[1]
    end
    return cost
end
     

#LQR Hover Controller
P = dare(A,B,Q,R)
K = dlqr(A,B,Q,R)

function lqr_controller_traj(tidx, x, K, x_ref_traj)
    xref = x_ref_traj[:, tidx]
    return u_hover - K*(x - xref)
end

     

#Build QP matrices for OSQP
Nh = 20 #one second horizon at 20Hz
Nx = 6
Nu = 2
U = kron(Diagonal(I,Nh), [I zeros(Nu,Nx)]) #Matrix that picks out all u
Θ = kron(Diagonal(I,Nh), [0 0 0 0 1 0 0 0]) #Matrix that picks out all x3 (θ)
H = sparse([kron(Diagonal(I,Nh-1),[R zeros(Nu,Nx); zeros(Nx,Nu) Q]) zeros((Nx+Nu)*(Nh-1), Nx+Nu); zeros(Nx+Nu,(Nx+Nu)*(Nh-1)) [R zeros(Nu,Nx); zeros(Nx,Nu) P]])
b = zeros(Nh*(Nx+Nu))
C = sparse([[B -I zeros(Nx,(Nh-1)*(Nu+Nx))]; zeros(Nx*(Nh-1),Nu) [kron(Diagonal(I,Nh-1), [A B]) zeros((Nh-1)*Nx,Nx)] + [zeros((Nh-1)*Nx,Nx) kron(Diagonal(I,Nh-1),[zeros(Nx,Nu) Diagonal(-I,Nx)])]])

#Dynamics + Thrust limit constraints
D = [C; U]
lb = [zeros(Nx*Nh); kron(ones(Nh),umin-u_hover)]
ub = [zeros(Nx*Nh); kron(ones(Nh),umax-u_hover)]

#Dynamics + thrust limit + bound constraint on θ to keep the system within small-angle approximation
#D = [C; U; Θ]
#lb = [zeros(Nx*Nh); kron(ones(Nh),umin-u_hover); -0.2*ones(Nh)]
#ub = [zeros(Nx*Nh); kron(ones(Nh),umax-u_hover); 0.2*ones(Nh)]

prob = OSQP.Model()
OSQP.setup!(prob; P=H, q=b, A=D, l=lb, u=ub, verbose=false, eps_abs=1e-8, eps_rel=1e-8, polish=1);
     

#MPC Controller
function mpc_controller_traj(tidx, x, x_ref_traj)
    xref = x_ref_traj[:, tidx]
    
    # Update QP problem with the current reference
    lb[1:6] .= -A*x
    ub[1:6] .= -A*x
    
    for j = 1:(Nh-1)
        b[(Nu+(j-1)*(Nx+Nu)).+(1:Nx)] .= -Q*xref
    end
    b[(Nu+(Nh-1)*(Nx+Nu)).+(1:Nx)] .= -P*xref
    
    OSQP.update!(prob, q=b, l=lb, u=ub)
    results = OSQP.solve!(prob)
    Δu = results.x[1:Nu]
    return u_hover + Δu
end

     

function closed_loop_traj(x0, controller, N, x_ref_traj)
    xhist = zeros(length(x0), N)
    uhist = zeros(2, N-1)
    xhist[:,1] .= x0
    uhist[:,1] .= controller(1, x0, x_ref_traj)
    for k = 1:(N-1)
        uk = controller(k, xhist[:,k], x_ref_traj)
        uhist[:,k] .= max.(min.(umax, uk), umin)
        xhist[:,k+1] .= quad_dynamics_rk4(xhist[:,k], uhist[:,k])
    end
    return xhist, uhist
end

     

x0 = [0.0; 0.0;0.0; 0.0; 0.0; 0.0]
xhist_lqr, uhist_lqr = closed_loop_traj(x0, (tidx,x,xr)->lqr_controller_traj(tidx,x,K,xr), Nt, x_ref_traj)
xhist_mpc, uhist_mpc = closed_loop_traj(x0, (tidx,x,xr)->mpc_controller_traj(tidx,x,xr), Nt, x_ref_traj)
figure()
plot(x_ref_traj[1,:], x_ref_traj[2,:], "k--", label="Reference")
plot(xhist_lqr[1,:], xhist_lqr[2,:], label="LQR")
plot(xhist_mpc[1,:], xhist_mpc[2,:], label="MPC")
xlabel("x [m]")
ylabel("y [m]")
legend()
title("Figure-8 Tracking")
pygui(true)
tracking_error_mpc = norm(x_ref_traj[1:2, :] - xhist_mpc[1:2, :])
tracking_error_lqr = norm(x_ref_traj[1:2, :] - xhist_lqr[1:2, :])
error_x_mpc = sqrt(mean((x_ref_traj[1,:] - xhist_mpc[1,:]).^2))
error_y_mpc = sqrt(mean((x_ref_traj[2,:] - xhist_mpc[2,:]).^2))
dist_error_mpc = [norm(x_ref_traj[1:2,k] - xhist_mpc[1:2,k]) for k in 1:Nt]
mean_error_mpc = mean(dist_error_mpc)
max_error_mpc = maximum(dist_error_mpc)

error_x_lqr = sqrt(mean((x_ref_traj[1,:] - xhist_lqr[1,:]).^2))
error_y_lqr = sqrt(mean((x_ref_traj[2,:] - xhist_lqr[2,:]).^2))
dist_error_lqr = [norm(x_ref_traj[1:2,k] - xhist_lqr[1:2,k]) for k in 1:Nt]
mean_error_lqr = mean(dist_error_lqr)
max_error_lqr = maximum(dist_error_lqr)
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

visualize!(vis, model, thist[end], X2)
