import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
import MeshCat as mc 
using Test
using Random
import Convex as cvx 
import ECOS      # the solver we use in this hw 
# import Hypatia # other solvers you can try
# import COSMO   # other solvers you can try 
using ProgressMeter
include(joinpath(@__DIR__,"utils/rendezvous.jl"))


# utilities for converting to and from vector of vectors <-> matrix 
function mat_from_vec(X::Vector{Vector{Float64}})::Matrix
    # convert a vector of vectors to a matrix 
    Xm = hcat(X...)
    return Xm 
end
function vec_from_mat(Xm::Matrix)::Vector{Vector{Float64}}
    # convert a matrix into a vector of vectors 
    X = [Xm[:,i] for i = 1:size(Xm,2)]
    return X 
end

function create_dynamics(dt::Real)::Tuple{Matrix,Matrix}
    mu = 3.986004418e14 # standard gravitational parameter
    a = 6971100.0       # semi-major axis of ISS
    n = sqrt(mu/a^3)    # mean motion
    # continuous time dynamics ẋ = Ax + Bu
    A = [0     0  0    1   0   0; 
         0     0  0    0   1   0;
         0     0  0    0   0   1;
         3*n^2 0  0    0   2*n 0;
         0     0  0   -2*n 0   0;
         0     0 -n^2  0   0   0]
         
    B = Matrix([zeros(3,3);0.1*I(3)])
    nx, nu = size(B)
    # TODO: convert to discrete time X_{k+1} = Ad*x_k + Bd*u_k
    M = zeros(nx+nu, nx+nu)
    M[1:nx, 1:nx] = A
    M[1:nx, nx+1:end] = B
    Md = exp(M*dt)
    Ad = Md[1:nx, 1:nx]
    Bd = Md[1:nx, nx+1:end]

    return Ad, Bd
end


function fhlqr(A::Matrix, # A matrix 
               B::Matrix, # B matrix 
               Q::Matrix, # cost weight 
               R::Matrix, # cost weight 
               Qf::Matrix,# term cost weight 
               N::Int64   # horizon size 
               )::Tuple{Vector{Matrix{Float64}}, Vector{Matrix{Float64}}} # return two matrices 
        
    # check sizes of everything 
    nx,nu = size(B)
    @assert size(A) == (nx, nx)
    @assert size(Q) == (nx, nx)
    @assert size(R) == (nu, nu)
    @assert size(Qf) == (nx, nx)
        
    # instantiate S and K 
    P = [zeros(nx,nx) for i = 1:N]
    K = [zeros(nu,nx) for i = 1:N-1]
    
    # initialize S[N] with Qf 
    P[N] = deepcopy(Qf)
    
    # Ricatti 
    for k = N:-1:2
        K[k-1] = (R+B'*P[k]*B)\(B'*P[k]*A)
        P[k-1] = A'*P[k]*A-(A'P[k]*B)*inv(R+B'*P[k]*B)*(B'*P[k]*A) + Q
    end
    
    return P, K 
end

"""
@testset "discrete dynamics" begin 
    A,B = create_dynamics(1.0)
    
    x = [1,3,-.3,.2,.4,-.5]
    u = [-.1,.5,.3]
    
    # test these matrices 
    @test isapprox(A*x + B*u, [1.195453, 3.424786, -0.78499972, 0.190925, 0.4495759, -0.4699993], atol = 1e-3)
    @test isapprox(det(A), 1, atol = 1e-8)
    @test isapprox(norm(B,Inf), 0.0999999803, atol = 1e-5)
    
end

@testset "LQR rendezvous" begin 

    # create our discrete time model 
    dt = 1.0
    A,B = create_dynamics(dt)

    # get our sizes for state and control
    nx,nu = size(B)

    # initial and goal states
    x0 = [-2;-4;2;0;0;.0]
    xg = [0,-.68,3.05,0,0,0]

    # bounds on U
    u_max = 0.4*ones(3)
    u_min = -u_max

    # problem size and reference trajectory 
    N = 120
    t_vec = 0:dt:((N-1)*dt)
    X_ref = desired_trajectory_long(x0,xg,200,dt)[1:N]
    
    # TODO: FHLQR 
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 10*Q
    # TODO get K's from fhlqr
    _, K = fhlqr(A, B, Q, R, Qf, N)
    
    
    # simulation 
    X_sim = [zeros(nx) for i = 1:N]
    U_sim = [zeros(nu) for i = 1:N-1]
    X_sim[1] = x0 
    for i = 1:(N-1) 
        # TODO: put LQR control law here 
        u = -K[i]*(X_sim[i]-X_ref[i])
        # make sure to clamp 
        U_sim[i] = clamp.(u, u_min,u_max)
        
        # simulate 1 step 
        X_sim[i+1] = A*X_sim[i] + B*U_sim[i]
    end

    # -------------plotting/animation---------------------------
    Xm = mat_from_vec(X_sim)
    Um = mat_from_vec(U_sim)
    display(plot(t_vec,Xm[1:3,:]',title = "Positions (LQR)",
                 xlabel = "time (s)", ylabel = "position (m)",
                 label = ["x" "y" "z"]))
    display(plot(t_vec,Xm[4:6,:]',title = "Velocities (LQR)",
            xlabel = "time (s)", ylabel = "velocity (m/s)",
                 label = ["x" "y" "z"]))
    display(plot(t_vec[1:end-1],Um',title = "Control (LQR)",
            xlabel = "time (s)", ylabel = "thrust (N)",
                 label = ["x" "y" "z"]))

    # feel free to toggle `show_reference`
    display(animate_rendezvous(X_sim, X_ref, dt;show_reference = false))
    # -------------plotting/animation---------------------------

    
    # testing 
    xs=[x[1] for x in X_sim]
    ys=[x[2] for x in X_sim]
    zs=[x[3] for x in X_sim]
    @test norm(X_sim[end] - xg) < .01 # goal 
    @test (xg[2] + .1) < maximum(ys) < 0 # we should have hit the ISS 
    @test maximum(zs) >= 4 # check to see if you did the circle 
    @test minimum(zs) <= 2 # check to see if you did the circle 
    @test maximum(xs) >= 1 # check to see if you did the circle 
    @test maximum(norm.(U_sim,Inf)) <= 0.4 # control constraints satisfied 

end
"""


"""
Xcvx,Ucvx = convex_trajopt(A,B,X_ref,x0,xg,u_min,u_max,N)

setup and solve the above optimization problem, returning 
the solutions X and U, after first converting them to 
vectors of vectors with vec_from_mat(X.value)
"""
function convex_trajopt(A::Matrix, # discrete dynamics A 
                        B::Matrix, # discrete dynamics B 
                        X_ref::Vector{Vector{Float64}}, # reference trajectory 
                        x0::Vector, # initial condition 
                        xg::Vector, # goal state 
                        u_min::Vector, # lower bound on u 
                        u_max::Vector, # upper bound on u
                        N::Int64, # length of trajectory 
                        )::Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}} # return Xcvx,Ucvx
    
    # get our sizes for state and control
    nx,nu = size(B)
    
    @assert size(A) == (nx, nx)
    @assert length(x0) == nx 
    @assert length(xg) == nx 
        
    # LQR cost
    Q = diagm(ones(nx))
    R = diagm(ones(nu))

    # variables we are solving for
    X = cvx.Variable(nx,N)
    U = cvx.Variable(nu,N-1)
    # TODO: implement cost
    obj = 0
    for k = 1:(N-1)
        obj += 0.5*(cvx.quadform(X[:, k]- X_ref[k],Q))+ 0.5*cvx.quadform(U[:, k], R)
    end
    # create problem with objective
    prob = cvx.minimize(obj)

    # TODO: add constraints with prob.constraints = vcat(prob.constraints, ...)
    prob.constraints = vcat(X[:, 1] == x0, X[:, N]==xg)
    for k = 1:N-1
        prob.constraints = vcat(prob.constraints,  X[:, k+1] == A*X[:, k]+B*U[:, k], u_min <= U[:, k], u_max >= U[:, k], X[2, k]<=xg[2])
    end
    prob.constraints = vcat(prob.constraints, X[2, N]<=xg[2])

    cvx.solve!(prob, ECOS.Optimizer; silent = true)

    X = X.value
    U = U.value
    
    Xcvx = vec_from_mat(X)
    Ucvx = vec_from_mat(U)
    
    return Xcvx, Ucvx
end
    
@testset "convex trajopt" begin 

    # create our discrete time model 
    dt = 1.0
    A,B = create_dynamics(dt)

    # get our sizes for state and control
    nx,nu = size(B)

    # initial and goal states
    x0 = [-2;-4;2;0;0;.0]
    xg = [0,-.68,3.05,0,0,0]

    # bounds on U
    u_max = 0.4*ones(3)
    u_min = -u_max

    # problem size and reference trajectory 
    N = 100 
    t_vec = 0:dt:((N-1)*dt)
    X_ref = desired_trajectory(x0,xg,N,dt) 
    
    # solve convex trajectory optimization problem 
    X_cvx, U_cvx = convex_trajopt(A,B,X_ref, x0,xg,u_min,u_max,N)
    
    X_sim = [zeros(nx) for i = 1:N]
    X_sim[1] = x0 
    for i = 1:N-1 
        X_sim[i+1] = A*X_sim[i] + B*U_cvx[i]
    end

    # -------------plotting/animation---------------------------
    Xm = mat_from_vec(X_sim)
    Um = mat_from_vec(U_cvx)
    display(plot(t_vec,Xm[1:3,:]',title = "Positions",
                 xlabel = "time (s)", ylabel = "position (m)",
                 label = ["x" "y" "z"]))
    display(plot(t_vec,Xm[4:6,:]',title = "Velocities",
            xlabel = "time (s)", ylabel = "velocity (m/s)",
                 label = ["x" "y" "z"]))
    display(plot(t_vec[1:end-1],Um',title = "Control",
            xlabel = "time (s)", ylabel = "thrust (N)",
                 label = ["x" "y" "z"]))

    
    display(animate_rendezvous(X_sim, X_ref, dt;show_reference = false))
    # -------------plotting/animation---------------------------


    @test maximum(norm.( X_sim .- X_cvx, Inf)) < 1e-3 
    @test norm(X_sim[end] - xg) < 1e-3 # goal 
    xs=[x[1] for x in X_sim]
    ys=[x[2] for x in X_sim]
    zs=[x[3] for x in X_sim]
    @test maximum(ys) <= (xg[2] + 1e-3)
    @test maximum(zs) >= 4 # check to see if you did the circle 
    @test minimum(zs) <= 2 # check to see if you did the circle 
    @test maximum(xs) >= 1 # check to see if you did the circle 
    @test maximum(norm.(U_cvx,Inf)) <= 0.4 + 1e-3 # control constraints satisfied 

end

"""
`u = convex_mpc(A,B,X_ref_window,xic,xg,u_min,u_max,N_mpc)`

setup and solve the above optimization problem, returning the 
first control u_1 from the solution (should be a length nu 
Vector{Float64}).  
"""
function convex_mpc(A::Matrix, # discrete dynamics matrix A
                    B::Matrix, # discrete dynamics matrix B
                    X_ref_window::Vector{Vector{Float64}}, # reference trajectory for this window 
                    xic::Vector, # current state x 
                    xg::Vector, # goal state 
                    u_min::Vector, # lower bound on u 
                    u_max::Vector, # upper bound on u 
                    N_mpc::Int64,  # length of MPC window (horizon)
                    )::Vector{Float64} # return the first control command of the solved policy 
    
    # get our sizes for state and control
    nx,nu = size(B)
    
    # check sizes 
    @assert size(A) == (nx, nx)
    @assert length(xic) == nx 
    @assert length(xg) == nx 
    @assert length(X_ref_window) == N_mpc 
        
    # LQR cost
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 10*Q

    # variables we are solving for
    X = cvx.Variable(nx,N_mpc)
    U = cvx.Variable(nu,N_mpc-1)

    # TODO: implement cost function
    obj = 0
    for k = 1:(N_mpc-1)
        obj += 0.5*(cvx.quadform(X[:, k]- X_ref_window[k],Q))+ 0.5*cvx.quadform(U[:, k], R)
    end
    obj += 0.5*(cvx.quadform(X[:, N_mpc]- X_ref_window[N_mpc], Qf))
   

    # create problem with objective
    prob = cvx.minimize(obj)

    # TODO: add constraints with prob.constraints = vcat(prob.constraints, ...)
    prob.constraints = vcat(X[:, 1] == xic) 
    for k = 1:N_mpc-1
        prob.constraints = vcat(prob.constraints, X[:,k+1] == A*X[:, k] + B*U[:, k], u_min <= U[:, k], u_max >= U[:,k], X[2, k] <= xg[2])
    end
    prob.constraints = vcat(prob.constraints, X[2, N_mpc]<=xg[2])
    # solve problem 
    cvx.solve!(prob, ECOS.Optimizer; silent = true)

    # get X and U solutions 
    X = X.value
    U = U.value
    
    # return first control U 
    return U[:,1]
end
        
@testset "convex mpc" begin 

    # create our discrete time model 
    dt = 1.0
    A,B = create_dynamics(dt)

    # get our sizes for state and control
    nx,nu = size(B)

    # initial and goal states
    x0 = [-2;-4;2;0;0;.0]
    xg = [0,-.68,3.05,0,0,0]

    # bounds on U
    u_max = 0.4*ones(3)
    u_min = -u_max

    # problem size and reference trajectory 
    N = 100 
    t_vec = 0:dt:((N-1)*dt)
    X_ref = [desired_trajectory(x0,xg,N,dt)...,[xg for i = 1:N]...] 
    
    # MPC window size 
    N_mpc = 20 
    
    # sim size and setup 
    N_sim = N + 20 
    t_vec = 0:dt:((N_sim-1)*dt)
    X_sim = [zeros(nx) for i = 1:N_sim]
    X_sim[1] = x0 
    U_sim = [zeros(nu) for i = 1:N_sim-1]
    
    # simulate 
    @showprogress "simulating" for i = 1:N_sim-1 
        
        # get state estimate
        xi_estimate = state_estimate(X_sim[i], xg)
        
        # TODO: given a window of N_mpc timesteps, get current reference trajectory
        X_ref_tilde = X_ref[i:N_mpc+i-1]
        # TODO: call convex mpc controller with state estimate 
        u_mpc = convex_mpc(A, B, X_ref_tilde, xi_estimate, xg, u_min, u_max, N_mpc)
        
        # commanded control goes into thruster model where it gets modified 
        U_sim[i] = thruster_model(X_sim[i], xg, u_mpc)
        
        # simulate one step 
        X_sim[i+1] = A*X_sim[i] + B*U_sim[i]
    end
    
    

    # -------------plotting/animation---------------------------
    Xm = mat_from_vec(X_sim)
    Um = mat_from_vec(U_sim)
    display(plot(t_vec,Xm[1:3,:]',title = "Positions",
                 xlabel = "time (s)", ylabel = "position (m)",
                 label = ["x" "y" "z"]))
    display(plot(t_vec,Xm[4:6,:]',title = "Velocities",
            xlabel = "time (s)", ylabel = "velocity (m/s)",
                 label = ["x" "y" "z"]))
    display(plot(t_vec[1:end-1],Um',title = "Control",
            xlabel = "time (s)", ylabel = "thrust (N)",
                 label = ["x" "y" "z"]))

    
    display(animate_rendezvous(X_sim, X_ref, dt;show_reference = false))
    # -------------plotting/animation---------------------------

    # tests 
    @test norm(X_sim[end] - xg) < 1e-3 # goal 
    xs=[x[1] for x in X_sim]
    ys=[x[2] for x in X_sim]
    zs=[x[3] for x in X_sim]
    @test maximum(ys) <= (xg[2] + 1e-3)
    @test maximum(zs) >= 4 # check to see if you did the circle 
    @test minimum(zs) <= 2 # check to see if you did the circle 
    @test maximum(xs) >= 1 # check to see if you did the circle 
    @test maximum(norm.(U_sim,Inf)) <= 0.4 + 1e-3 # control constraints satisfied 

end