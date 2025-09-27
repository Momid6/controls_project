import numpy as np

def fhlqr(A, # A matrix 
               B, # B matrix 
               Q, # cost weight
               q, # cost weight 
               R, # cost weight
               r, # cost weight 
               Qf,# final term cost weight
               qf, # final term cost weight
               N,   # horizon size 
               ):
        
    # check sizes of everything 
    nx,nu = B.shape
        
    # instantiate S and K 

    P = [np.zeros((nx, nx)) for _ in range(N)]
    p = [np.zeros((nx, 1)) for _ in range(N)]
    K = [np.zeros((nu, nx)) for _ in range(N-1)] 
    d = [np.zeros((nu, 1)) for _ in range(N-1)] 
    # initialize S[N] with Qf 
    P[N-1] = np.copy(Qf)
    p[N-1] = np.copy(qf)
    
    # Ricatti 
    for k in range(N-1, 0, -1):
        K[k-1] = np.linalg.solve(R+B.T@P[k]@B, B.T@P[k]@A)
        d[k-1] = np.linalg.solve(R+B.T@P[k]@B, B.T@p[k] + r[k-1])
        P[k-1] = K[k-1].T@R@K[k-1] + (A-B@K[k-1]).T@P[k]@((A-B@K[k-1])) + Q 
        p[k-1] = (A-B@K[k-1]).T@(p[k] - P[k]@B@d[k-1]) + K[k-1].T@(R@d[k-1]-r)+ q
    
    return P, K, p, d

def solve_fhlqr(A, B, Q, q, R, r, Qf, qf, dt, tf, x_initial, x_goal):
    t_vec = np.arange(0, tf + dt, dt)
    N = t_vec.size
    nx, nu = B.shape
    P, K, p , d = fhlqr(A,B,Q, q, R, r, Qf, qf, N)
    Xsim_lqr = [np.zeros(nx) for i in range(N+1)]
    Xsim_lqr[0] = np.copy(x_initial)
    for i in range(0,N-1):
        u_lqr = -K[i]@(Xsim_lqr[i]-x_goal) - d[i]
        Xsim_lqr[i+1] = A@Xsim_lqr[i] + B@u_lqr
    return Xsim_lqr