import numpy as np
import cvxpy as cp
from scipy.linalg import expm
import matplotlib.pyplot as plt

def convex_trajopt(A,      # A matrix 
                        B,      # B matrix 
                        Q,      # cost weight 
                        R,      # cost weight 
                        Qf,     # term cost weight 
                        N,       # horizon size 
                        x_ic   # initial condition
                        ):
     
    nx,nu = B.shape

    X = cp.Variable((nx, N+1))
    U = cp.Variable((nu, N))
    
    cost = 0 
    for k in range(N):
        current_x = 0.5*cp.QuadForm(X[:,k],Q)
        current_u = 0.5*cp.QuadForm(U[:,k],R)
        cost += current_x + current_u
    
    cost += 0.5*cp.QuadForm(X[:,N],Qf)
    

    constraints = [X[:, 0] == x_ic]
    for k in range(N):
        constraints += [X[:,(k+1)] == A@X[:,k]+B@U[:,k]]


    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)
    
    
    X = X.value
    U = U.value
    
    return X, U 


def solve_lqr_convex(A, B, Q, R, Qf, dt, tf, x_initial):
    t_vec = np.arange(0, tf, dt)
    N = t_vec.size
    nx, nu = B.shape
    
    
    Xcvx,Ucvx = convex_trajopt(A,B,Q,R,Qf,N,x_initial) 
    

    Xsim = [np.zeros(nx) for i in range(N+1)]
    Xsim[0] = np.copy(x_initial)

    for i in range(N):
        Xsim[i+1] = A@Xsim[i] + B@Ucvx[:, i]
    return Xsim