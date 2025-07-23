import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def ihlqr(A, B, Q, R, max_iter: int= 1000, tol = 1e-5):
    # get size of x and u from B 
    nx, nu = B.shape
        
    # initialize S with Q
    P = np.copy(Q)
    K = np.zeros((nx, nu))
    # Riccati 
    for riccati_iter in range(max_iter+1): 
        K = np.linalg.solve((R+B.T@P@B),(B.T@P@A))
        P_new = A.T@P@A-(A.T@P@B)@np.linalg.inv(R+B.T@P@B)@(B.T@P@A) + Q
        if np.linalg.norm(P_new-P) <= tol:
            return P, K
        P = P_new

def solve_ihlqr(A, B, Q, R, dt, tf, x_initial, x_goal):
    t_vec = np.arange(0, dt+tf, dt) 
    N = t_vec.size
    nx, nu = B.shape
    P, K = ihlqr(A,B,Q,R)
    Xsim_lqr = [np.zeros(nx) for i in  range(N+1)]
    Xsim_lqr[0] = np.copy(x_initial)
    for i in range(N):
        u_lqr = -K@(Xsim_lqr[i]-x_goal)
        Xsim_lqr[i+1] = A@Xsim_lqr[i] + B @ u_lqr
    return Xsim_lqr
