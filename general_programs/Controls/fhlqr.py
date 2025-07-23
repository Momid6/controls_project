import numpy as np

def fhlqr(A, # A matrix 
               B, # B matrix 
               Q, # cost weight 
               R, # cost weight 
               Qf,# term cost weight 
               N,   # horizon size 
               ):
        
    # check sizes of everything 
    nx,nu = B.shape
        
    # instantiate S and K 

    P = [np.zeros((nx, nx)) for _ in range(N)]
    K = [np.zeros((nu, nx)) for _ in range(N-1)] 
    
    # initialize S[N] with Qf 
    P[N-1] = np.copy(Qf)
    
    # Ricatti 
    for k in range(N-1, 0, -1):
        K[k-1] = np.linalg.solve(R+B.T@P[k]@B, B.T@P[k]@A)
        P[k-1] = A.T@P[k]@A-(A.T@P[k]@B)@np.linalg.inv(R+B.T@P[k]@B)@(B.T@P[k]@A) + Q
        print(k)
    
    return P, K 

def solve_fhlqr(A, B, Q, R, Qf, dt, tf, x_initial, x_goal):
    t_vec = np.arange(0, tf + dt, dt)
    N = t_vec.size
    nx, nu = B.shape
    P, K = fhlqr(A,B,Q,R,Qf,N)
    Xsim_lqr = [np.zeros(nx) for i in range(N+1)]
    Xsim_lqr[0] = np.copy(x_initial)
    for i in range(0,N-1):
        u_lqr = -K[i]@(Xsim_lqr[i]-x_goal)
        Xsim_lqr[i+1] = A@Xsim_lqr[i] + B@u_lqr
    return Xsim_lqr