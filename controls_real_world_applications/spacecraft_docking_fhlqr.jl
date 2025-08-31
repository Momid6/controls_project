import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
import MeshCat as mc 
using Test
using Random
import Convex as cvx 
import ECOS 
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