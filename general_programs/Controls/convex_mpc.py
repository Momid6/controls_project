import numpy as np
import cvxpy as cp


def convex_mpc(A, # discrete dynamics matrix A
                    B, # discrete dynamics matrix B
                    xic, # current state x 
                    xg, # goal state 
                    u_min, # lower bound on u 
                    u_max, # upper bound on u 
                    N_mpc,  # length of MPC window (horizon)
                    ): # return the first control command of the solved policy 
    
    nx,nu = B.shape

        
    Q = np.eye(nx)
    R = np.eye(nu)
    Qf = 10*Q


    X = cp.Variable((nx,N_mpc+1))
    U = cp.Variable((nu,N_mpc))

    obj = 0
    for k in range(N_mpc):
        obj += 0.5*(cp.QuadForm(X[:, k]- xg,Q))+ 0.5*cp.QuadForm(U[:, k], R)
    obj += 0.5*(cp.QuadForm(X[:, N_mpc]- xg, Qf))
   


    constraints = [X[:, 0] == xic]
    for k in range(N_mpc):
        constraints += [X[:,k+1] == A@X[:, k] + B@U[:, k], u_min <= U[:, k], u_max >= U[:,k], X[0:2, k] <= xg[0:2]]
    constraints += [X[0:2, N_mpc]<=xg[0:2]]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)
    X = X.value
    U = U.value
    return U[:,0]

def solve_convex_mpc(A,B, x0, xg, u_min, u_max, dt, N, N_mpc):
    nx, nu = B.shape
    N_sim = N + N_mpc
    t_vec = np.arange(0,N_sim*dt, dt)
    X_sim = [np.zeros(nx) for i in range(N_sim+1)]
    X_sim[0] = x0 
    U_sim = [np.zeros(nu) for i in range(N_sim)]
    for i in range(N_sim):
        u_mpc = convex_mpc(A, B, X_sim[i], xg, u_min, u_max, N_mpc)
        U_sim[i] = u_mpc
        X_sim[i+1] = A@X_sim[i] + B@U_sim[i]
    return X_sim, U_sim