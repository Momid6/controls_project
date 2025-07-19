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

"""
continuous time dynamics for a cartpole, the state is 
x = [p, θ, ṗ, θ̇]
where p is the horizontal position, and θ is the angle
where θ = 0 has the pole hanging down, and θ = 180 is up.

The cartpole is parametrized by a cart mass `mc`, pole 
mass `mp`, and pole length `l`. These parameters are loaded 
into a `params::NamedTuple`. We are going to design the
controller for a estimated `params_est`, and simulate with 
`params_real`. 
"""
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
    
    # TODO: solve for the infinite horizon LQR gain Kinf
    
    # cost terms 
    Q = diagm([1,1,.05,.1])
    R = 0.1*diagm(ones(nu))
    A = FD.jacobian(dx -> rk4(params_est, dx, ugoal, dt), xgoal)
    B = FD.jacobian(du -> rk4(params_est, xgoal, du, dt), ugoal)

    _, Kinf = ihlqr(A, B, Q, R)
    # TODO: simulate this controlled system with rk4(params_real, ...)
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

function create_initial_conditions()
    # create a span of initial configurations 
    M=20
    ps = LinRange(-7, 7, M)
    thetas = LinRange(deg2rad(180-60), deg2rad(180+60), M)
    
    initial_conditions = []
    
    for p in ps 
        for theta in thetas 
            push!(initial_conditions, [p, theta, 0, 0.0])
        end
    end
    
    return initial_conditions, ps, thetas
end

function check_simulation_convergence(params_real, initial_condition, Kinf, xgoal, N, dt)
    """
    args
        params_real: named tuple with model dynamics parametesr 
        initial_condition: X0, length 4 vector 
        Kinf: IHLQR feedback gain 
        xgoal: desired state, length 4 vector 
        N: number of simulation steps
        dt: time between steps 
    
    return
        is_controlled: bool 
    """

    x0= 1 * initial_condition 

    is_controlled = false
    
    # TODO: simulate the closed-loop (controlled) cartpole starting at the initial condition 
    
    # for some of the unstable initial conditions, the integrator will "blow up", in order to 
    # catch these errors, you can stop the sim and return is_controlled = false if norm(x) > 100 
    
    # you should consider the simulation to have been successfuly controlled if the 
    # L2 norm of |xfinal - xgoal| < 0.1. (norm(xfinal-xgoal) < 0.1 in Julia)
    x = x0
    for k=1:N-1
        u = -Kinf*(x-xgoal)
        x = rk4(params_real, x, u, dt)
        if norm(x)>100
            is_controlled=false
            return is_controlled
        end
    end
    if norm(x-xgoal) < 0.1
        is_controlled = true
    else
        is_controlled = false
    end
    

    return is_controlled 
end

let 
    
    nx = 4 
    nu = 1 
    xgoal = [0, pi, 0, 0]
    ugoal = [0]
    dt = 0.1 
    tf = 5.0 
    t_vec = 0:dt:tf
    N = length(t_vec)
    
    
    # estimated parameters (design our controller with these)
    params_est = (mc = 1.0, mp = 0.2, l = 0.5)
    
    # real paremeters (simulate our system with these)
    params_real = (mc = 1.2, mp = 0.16, l = 0.55)
    
    # TODO: solve for the infinite horizon LQR gain Kinf
    # this is the same controller as part B
    # cost terms
    Q = diagm([1,1,.05,.1])
    R = 0.1*diagm(ones(nu))
    A = FD.jacobian(dx -> rk4(params_est, dx, ugoal, dt), xgoal)
    B = FD.jacobian(du -> rk4(params_est, xgoal, du, dt ), ugoal)
    
    _, Kinf = ihlqr(A,B,Q,R)
    
    # create the set of initial conditions we want to test for convergence
    initial_conditions, ps, thetas = create_initial_conditions()
    
    convergence_list = [] 
    
    for initial_condition in initial_conditions

        convergence = check_simulation_convergence(params_real,
                                                   initial_condition,
                                                   Kinf, xgoal, N, dt)
        
        push!(convergence_list, convergence)
    end
    
    plot_basin_of_attraction(initial_conditions, convergence_list, ps, rad2deg.(thetas))
    

    # -------------tests------------------
    @test sum(convergence_list) < 190 
    @test sum(convergence_list) > 180 
    @test length(convergence_list) == 400 
    @test length(initial_conditions) == 400
        
end


@testset "LQR about eq" begin
    
    # states and control sizes 
    nx = 4 
    nu = 1 
    
    # desired x and g (linearize about these)
    xgoal = [0, pi, 0, 0]
    ugoal = [0]
    
    # initial condition (slightly off of our linearization point)
    x0 = [0, pi, 0, 0] + [0.5, deg2rad(-10), .3, 0]
    
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
    
    # TODO: solve for the infinite horizon LQR gain Kinf
    
    # cost terms
    Q = diagm([1,1,.01,.01])
    R = 1*diagm(ones(nu))
    A = FD.jacobian(dx -> rk4(params_est, dx, ugoal, dt), xgoal)
    B = FD.jacobian(du -> rk4(params_est, xgoal, du, dt), ugoal)
    _, Kinf = ihlqr(A, B, Q, R) 
    
    # vector of length 1 vectors for our control
    U = [[0.0] for i = 1:N-1]
    
    # TODO: simulate this controlled system with rk4(params_real, ...)
    # TODO: make sure you clamp the control input with clamp.(U[i], -3.0, 3.0)
    for k = 1:N-1
        U[k] = clamp.(-Kinf*(X[k]-xgoal), -3, 3)
        X[k+1] = rk4(params_real, X[k], U[k], dt)
    end
  
      
    
    
    # ---------------tests and plots/animations---------------
    @test X[1] == x0  # initial condition is used
    @test norm(X[end])>0 # end is nonzero
    @test norm(X[end] - xgoal) < 0.1 # within 0.1 of the goal 
    @test norm(vcat(U...), Inf) <= 3.0 # actuator limits are respected
    
    Xm = hcat(X...)
    display(plot(t_vec,Xm',title = "cartpole",
                 xlabel = "time(s)", ylabel = "x",
                 label = ["p" "θ" "ṗ" "θ̇"]))
    
    # animation stuff
    display(animate_cartpole(X, dt))
    # ---------------tests and plots/animations---------------
    
    
end