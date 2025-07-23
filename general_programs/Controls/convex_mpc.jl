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


function convex_mpc(A::Matrix, # discrete dynamics matrix A
                    B::Matrix, # discrete dynamics matrix B
                    xic::Vector, # current state x 
                    xg::Vector, # goal state 
                    u_min::Vector, # lower bound on u 
                    u_max::Vector, # upper bound on u 
                    N_mpc::Int64,  # length of MPC window (horizon)
                    )::Vector{Float64} # return the first control command of the solved policy 
    
    nx,nu = size(B)
    
    @assert size(A) == (nx, nx)
    @assert length(xic) == nx 
    @assert length(xg) == nx 
        
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 10*Q


    X = cvx.Variable(nx,N_mpc)
    U = cvx.Variable(nu,N_mpc-1)


    obj = 0
    for k = 1:(N_mpc-1)
        obj += 0.5*(cvx.quadform(X[:, k]- xg,Q))+ 0.5*cvx.quadform(U[:, k], R)
    end
    obj += 0.5*(cvx.quadform(X[:, N_mpc]- xg, Qf))
   

    prob = cvx.minimize(obj)

    prob.constraints = vcat(X[:, 1] == xic) 
    for k = 1:N_mpc-1
        prob.constraints = vcat(prob.constraints, X[:,k+1] == A*X[:, k] + B*U[:, k], u_min <= U[:, k], u_max >= U[:,k], X[1:3, k] <= xg[1:3])
    end
    
    prob.constraints = vcat(prob.constraints, X[1:3, N_mpc]<=xg[1:3])
    cvx.solve!(prob, ECOS.Optimizer; silent = true)
    X = X.value
    U = U.value
    
    return U[:,1]
end

function solve_convex_mpc(A,B, x0, xg, u_min, u_max, dt, N, N_mpc)
    nx, nu = size(B)
    N_sim = N + N_mpc
    t_vec = 0:dt:((N_sim-1)*dt)
    X_sim = [zeros(nx) for i = 1:N_sim]
    X_sim[1] = x0 
    U_sim = [zeros(nu) for i = 1:N_sim-1]
    
    @showprogress "simulating" for i = 1:N_sim-1 
        
        u_mpc = convex_mpc(A, B, X_sim[i], xg, u_min, u_max, N_mpc)
        U_sim[i] = u_mpc
        X_sim[i+1] = A*X_sim[i] + B*U_sim[i]
    end
    return X_sim, U_sim
end