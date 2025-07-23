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
        P[k-1] = A'*P[k]*A-(A'*P[k]*B)*inv(R+B'*P[k]*B)*(B'*P[k]*A) + Q
    end
    
    return P, K 
end

function solve_fhlqr(A, B, Q, R, Qf, dt, tf, x_initial, x_goal)
    t_vec = 0:dt:tf 
    N = length(t_vec)
    nx, nu = size(B)
    P, K = fhlqr(A,B,Q,R,Qf,N)
    Xsim_lqr = [zeros(nx) for i = 1:N]
    Xsim_lqr[1] = 1*x_initial
    for i = 1:N-1
        u_lqr = -K[i]*(Xsim_lqr[i]-x_goal)
        Xsim_lqr[i+1] = A*Xsim_lqr[i] + B*u_lqr
    end
    return Xsim_lqr
end

