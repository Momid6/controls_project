import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
import MeshCat as mc 
using JLD2
using Test
using Random
include(joinpath(@__DIR__,"utils/cartpole_animation.jl"))
include(joinpath(@__DIR__,"utils/basin_of_attraction.jl"))

function dynamics(params::NamedTuple, x::Vector, u)
    # cartpole ODE, parametrized by params. 

    # cartpole physical parameters 
    mc, mp, l = params.mc, params.mp, params.l
    g = 9.81
    
    q = x[1:2]
    qd = x[3:4]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp*g*l*s]
    B = [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd;qdd]

end

# true nonlinear dynamics of the system
# if I want to simulate, this is what I do
function rk4(params::NamedTuple, x::Vector,u,dt::Float64)
    # vanilla RK4
    k1 = dt*dynamics(params, x, u)
    k2 = dt*dynamics(params, x + k1/2, u)
    k3 = dt*dynamics(params, x + k2/2, u)
    k4 = dt*dynamics(params, x + k3, u)
    return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end
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

@testset "LQR about eq" begin
    
    # states and control sizes 
    nx = 4 
    nu = 1 
    
    # desired x and g (linearize about these)
    xgoal = [0, pi, 0, 0]
    ugoal = [0]
    
    # initial condition (slightly off of our linearization point)
    x0 = [0, pi, 0, 0] + [1.5, deg2rad(-20), .3, 0]
    
    # simulation size 
    dt = 0.1 
    tf = 5.0 
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(nx) for i = 1:N]
    X[1] = x0 
    
    # estimated parameters (design our controller with these)
    params_est = (mc = 1.0, mp = 0.2, l = 0.5)
    
    # real paremeters (simulate our system with these)
    params_real = (mc = 1.2, mp = 0.16, l = 0.55)
    
    # cost terms 
    Q = diagm([1,1,.05,.1])
    R = 0.1*diagm(ones(nu))
    A = FD.jacobian(dx -> rk4(params_est, dx, ugoal, dt), xgoal)
    B = FD.jacobian(du -> rk4(params_est, xgoal, du, dt), ugoal)

    _, Kinf = ihlqr(A, B, Q, R)
    for k = 1:N-1
        u = ugoal -Kinf*(X[k]-xgoal)
        X[k+1] = rk4(params_real, X[k], u, dt)
    end

    
    # ---------------tests and plots/animations---------------
    @test X[1] == x0 
    @test norm(X[end])>0
    @test norm(X[end] - xgoal) < 0.1 
    
    Xm = hcat(X...)
    display(plot(t_vec,Xm',title = "cartpole",
                 xlabel = "time(s)", ylabel = "x",
                 label = ["p" "θ" "ṗ" "θ̇"]))
    
    # animation stuff
    display(animate_cartpole(X, dt))
    # ---------------tests and plots/animations---------------
end