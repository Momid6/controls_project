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

function solve_ihlqr(A, B, Q, R, dt, tf, x_initial, x_goal)
    t_vec = 0:dt:tf 
    N = length(t_vec)
    nx, nu = size(B)
    P, K = ihlqr(A,B,Q,R)
    Xsim_lqr = [zeros(nx) for i = 1:N]
    Xsim_lqr[1] = 1*x_initial
    for i = 1:N-1
        u_lqr = -K*(Xsim_lqr[i]-x_goal)
        Xsim_lqr[i+1] = A*Xsim_lqr[i] + B*u_lqr
    end
    return Xsim_lqr
end
