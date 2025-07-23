import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
using Test
import Convex as cvx 
import ECOS
using Random
using MathOptInterface

function convex_trajopt(A::Matrix,      # A matrix 
                        B::Matrix,      # B matrix 
                        Q::Matrix,      # cost weight 
                        R::Matrix,      # cost weight 
                        Qf::Matrix,     # term cost weight 
                        N::Int64,       # horizon size 
                        x_ic::Vector;   # initial condition
                        verbose = false
                        )::Tuple{Vector{Vector{Float64}},Vector{Vector{Float64}}}
     
    nx,nu = size(B)
    @assert size(A) == (nx, nx)
    @assert size(Q) == (nx, nx)
    @assert size(R) == (nu, nu)
    @assert size(Qf) == (nx, nx)
    @assert length(x_ic) == nx
    
    X = cvx.Variable(nx, N)
    U = cvx.Variable(nu, N - 1)
    
    cost = 0 
    for k = 1:(N-1)
        current_x = 0.5*cvx.quadform(X[:,k],Q)
        current_u = 0.5*cvx.quadform(U[:,k],R)
        cost += current_x + current_u
    end
    
    cost += 0.5*cvx.quadform(X[:,N],Qf)
    
    prob = cvx.minimize(cost)

    prob.constraints = [X[:, 1] == x_ic]
    for k = 1:(N-1)
        prob.constraints= vcat(prob.constraints, (X[:,(k+1)] == A*X[:,k]+B*U[:,k]))
    end

    cvx.solve!(prob, ECOS.Optimizer; silent = !verbose) 
    
    if prob.status != MathOptInterface.OPTIMAL
        error("Convex.jl problem failed to solve for some reason")
    end 
    
    X = vec_from_mat(X.value) 
    U = vec_from_mat(U.value) 
    
    return X, U 
end

function solve_lqr_convex(A, B, Q, R, Qf, dt, tf, x_initial)
    t_vec = 0:dt:tf 
    N = length(t_vec)
    nx, nu = size(B)
    
    
    Xcvx,Ucvx = convex_trajopt(A,B,Q,R,Qf,N,x_initial; verbose = false) 
    

    Xsim = [zeros(nx) for i = 1:N]
    Xsim[1] = 1*x_initial

    for i = 1:N-1
        Xsim[i+1] = A*Xsim[i] + B*Ucvx[i]
    end
    return Xsim
end
