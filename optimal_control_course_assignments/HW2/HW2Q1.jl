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

# double integrator dynamics 
function double_integrator_AB(dt)::Tuple{Matrix,Matrix}
    Ac = [0 0 1 0;
          0 0 0 1;
          0 0 0 0;
          0 0 0 0.]
    Bc = [0 0;
          0 0;
          1 0;
          0 1]
    nx, nu = size(Bc)
        
    # TODO: discretize this linear system using the Matrix Exponential
    M = zeros(nx + nu, nx + nu)
    M[1:nx, 1:nx] .= Ac
    M[1:nx,nx+1:end] .= Bc
    Md = exp(dt * M)
    
    A = Md[1:nx, 1:nx] # TODO 
    B = Md[1:nx,nx+1:end] # TODO 
    
    @assert size(A) == (nx,nx)
    @assert size(B) == (nx,nu)
    
    return A, B 
end
    
@testset "discrete time dynamics" begin 
    dt = 0.1 
    A,B = double_integrator_AB(dt)
    
    x = [1,2,3,4.]
    u = [-1,-3.]
    
    @test isapprox((A*x + B*u),[1.295, 2.385, 2.9, 3.7];atol = 1e-10) 
    
end

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
"""
X,U = convex_trajopt(A,B,Q,R,Qf,N,x_ic; verbose = false)

This function takes in a dynamics model x_{k+1} = A*x_k + B*u_k
and LQR cost Q,R,Qf, with a horizon size N, and initial condition 
x_ic, and returns the optimal X and U's from the above optimization 
problem. You should use the `vec_from_mat` function to convert the 
solution matrices from cvx into vectors of vectors (vec_from_mat(X.value))
"""
function convex_trajopt(A::Matrix,      # A matrix 
                        B::Matrix,      # B matrix 
                        Q::Matrix,      # cost weight 
                        R::Matrix,      # cost weight 
                        Qf::Matrix,     # term cost weight 
                        N::Int64,       # horizon size 
                        x_ic::Vector;   # initial condition
                        verbose = false
                        )::Tuple{Vector{Vector{Float64}},Vector{Vector{Float64}}}
    
    # check sizes of everything 
    nx,nu = size(B)
    @assert size(A) == (nx, nx)
    @assert size(Q) == (nx, nx)
    @assert size(R) == (nu, nu)
    @assert size(Qf) == (nx, nx)
    @assert length(x_ic) == nx
    
    # TODO: 
    # create cvx variables where each column is a time step
    X = cvx.Variable(nx, N)
    U = cvx.Variable(nu, N - 1)
    
    # create cost 
    # hint: you can't do x'*Q*x in Convex.jl, you must do cvx.quadform(x,Q)
    # hint: add all of your cost terms to `cost`  
    # hint: x_k = X[:,k], u_k = U[:,k]
    cost = 0 
    for k = 1:(N-1)
        current_x = 0.5*cvx.quadform(X[:,k],Q)
        current_u = 0.5*cvx.quadform(U[:,k],R)
        # add stagewise cost
        cost += current_x + current_u
    end
    
    # add terminal cost
    cost += 0.5*cvx.quadform(X[:,N],Qf)
    
    # initialize cvx problem 
    prob = cvx.minimize(cost)

    # TODO: initial condition constraint 
    # hint: you can add constraints to our problem like this:
    # prob.constraints = vcat(prob.constraints, (Gz == h)) 
    prob.constraints = [X[:, 1] == x_ic]
    for k = 1:(N-1)
        # dynamics constraints 
        prob.constraints= vcat(prob.constraints, (X[:,(k+1)] == A*X[:,k]+B*U[:,k]))
    end
    
    # solve problem (silent solver tells us the output)
    cvx.solve!(prob, ECOS.Optimizer; silent = !verbose) 
    
    if prob.status != MathOptInterface.OPTIMAL
        error("Convex.jl problem failed to solve for some reason")
    end 
    
    # convert the solution matrices into vectors of vectors 
    X = vec_from_mat(X.value) 
    U = vec_from_mat(U.value) 
    
    return X, U 
end

@testset "LQR via Convex.jl" begin 
    
    # problem setup stuff 
    dt = 0.1 
    tf = 5.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    A,B = double_integrator_AB(dt)
    nx,nu = size(B)
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 5*Q 
    
    # initial condition 
    x_ic = [5,7,2,-1.4]
    
    # setup and solve our convex optimization problem (verbose = true for submission)
    Xcvx,Ucvx = convex_trajopt(A,B,Q,R,Qf,N,x_ic; verbose = false) 
    
    # TODO: simulate with the dynamics with control Ucvx, storing the 
    # state in Xsim
    # initial condition 
    Xsim = [zeros(nx) for i = 1:N]
    Xsim[1] = 1*x_ic 

    # TODO dynamics simulation 
    for i = 1:N-1
        Xsim[i+1] = A*Xsim[i] + B*Ucvx[i]
    end
    
    
    @test length(Xsim) == N 
    @test norm(Xsim[end])>1e-13 
    #----------------------plotting-----------------------
    Xsim_m = mat_from_vec(Xsim)
    
    # plot state history 
    display(plot(t_vec,Xsim_m',label = ["x₁" "x₂" "ẋ₁" "ẋ₂"],
                 title = "State History",
                 xlabel = "time (s)", ylabel = "x"))
    
    # plot trajectory in x1 x2 space 
    display(plot(Xsim_m[1,:],Xsim_m[2,:],
                 title = "Trajectory in State Space",
                 ylabel = "x₂", xlabel = "x₁", label = ""))
    #----------------------plotting-----------------------
    
    # tests 
    @test 1e-14 < maximum(norm.(Xsim .- Xcvx,Inf)) < 1e-3
    @test isapprox(Ucvx[1], [-7.8532442316767, -4.127120137234], atol = 1e-3)
    @test isapprox(Xcvx[end], [-0.02285990, -0.07140241, -0.21259, -0.1540299], atol = 1e-3)
    @test 1e-14 < norm(Xcvx[end] - Xsim[end]) < 1e-3
end

@testset "Bellman's Principle of Optimality" begin 
    
    # problem setup 
    dt = 0.1 
    tf = 5.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    A,B = double_integrator_AB(dt)
    nx,nu = size(B)
    x0 = [5,7,2,-1.4] # initial condition 
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 5*Q 
    
    # solve for X_{1:N}, U_{1:N-1} with convex optimization
    Xcvx1,Ucvx1 = convex_trajopt(A,B,Q,R,Qf,N,x0; verbose = false)
        
    # now let's solve a subsection of this trajectory 
    L = 18 
    N_2 = N - L + 1
    
    # here is our updated initial condition from the first problem 
    x0_2 = Xcvx1[L]
    Xcvx2,Ucvx2 = convex_trajopt(A,B,Q,R,Qf,N_2,x0_2; verbose = false)
        
    # test if these trajectories match for the times they share 
    U_error = Ucvx1[L:end] .- Ucvx2
    X_error = Xcvx1[L:end] .- Xcvx2
    @test 1e-14 < maximum(norm.(U_error)) < 1e-3
    @test 1e-14 < maximum(norm.(X_error)) < 1e-3


    # ---------------------------plotting ------------------------------
    X1m = mat_from_vec(Xcvx1)
    X2m = mat_from_vec(Xcvx2)
    plot(X2m[1,:],X2m[2,:], label = "optimal subtrajectory", lw = 5, ls = :dot)
    display(plot!(X1m[1,:],X1m[2,:],
                 title = "Trajectory in State Space",
                 ylabel = "x₂", xlabel = "x₁", label = "full trajectory"))
    # ---------------------------plotting ------------------------------
    
    @test isapprox(Xcvx1[end], [-0.02285990, -0.07140241, -0.21259, -0.1540299], rtol = 1e-3)
    @test 1e-14 < norm(Xcvx1[end] - Xcvx2[end],Inf) < 1e-3
end

"""
use the Riccati recursion to calculate the cost to go quadratic matrix P and 
optimal control gain K at every time step. Return these as a vector of matrices, 
where P_k = P[k], and K_k = K[k]
"""
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

@testset "Convex trajopt vs LQR" begin 
    
    # problem stuff
    dt = 0.1 
    tf = 5.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    A,B = double_integrator_AB(dt)
    nx,nu = size(B)
    x0 = [5,7,2,-1.4] # initial condition 
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 5*Q 
    
    # solve for X_{1:N}, U_{1:N-1} with convex optimization
    Xcvx,Ucvx = convex_trajopt(A,B,Q,R,Qf,N,x0; verbose = false)
    P, K = fhlqr(A,B,Q,R,Qf,N)
    # now let's simulate using Ucvx 
    Xsim_cvx = [zeros(nx) for i = 1:N]
    Xsim_cvx[1] = 1*x0 
    Xsim_lqr = [zeros(nx) for i = 1:N]
    Xsim_lqr[1] = 1*x0 
    for i = 1:N-1
        # simulate cvx control 
        Xsim_cvx[i+1] = A*Xsim_cvx[i] + B*Ucvx[i]
        
        # TODO: use your FHLQR control gains K to calculate u_lqr
        # simulate lqr control
        u_lqr = -K[i]*Xsim_lqr[i]
        Xsim_lqr[i+1] = A*Xsim_lqr[i] + B*u_lqr
    end
    

    @test isapprox(Xsim_lqr[end], [-0.02286201, -0.0714058, -0.21259, -0.154030], rtol = 1e-3)
    @test 1e-13 < norm(Xsim_lqr[end] - Xsim_cvx[end]) < 1e-3
    @test 1e-13 < maximum(norm.(Xsim_lqr - Xsim_cvx)) < 1e-3

    
    # ------------------------plotting--------------------------
    X1m = mat_from_vec(Xsim_cvx)
    X2m = mat_from_vec(Xsim_lqr)
    # plot trajectory in x1 x2 space 
    plot(X1m[1,:],X1m[2,:], label = "cvx trajectory", lw = 4, ls = :dot)
    display(plot!(X2m[1,:],X2m[2,:],
                 title = "Trajectory in State Space",
                 ylabel = "x₂", xlabel = "x₁", lw = 2, label = "lqr trajectory"))
    # ------------------------plotting--------------------------

end


@testset "Why LQR is great reason 1" begin 
    
    # problem stuff
    dt = 0.1 
    tf = 7.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    A,B = double_integrator_AB(dt)
    nx,nu = size(B)
    x0 = [5,7,2,-1.4] # initial condition 
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 10*Q 
    
    # solve for X_{1:N}, U_{1:N-1} with convex optimization
    Xcvx,Ucvx = convex_trajopt(A,B,Q,R,Qf,N,x0; verbose = false)
    P, K = fhlqr(A,B,Q,R,Qf,N)
    # now let's simulate using Ucvx 
    Xsim_cvx = [zeros(nx) for i = 1:N]
    Xsim_cvx[1] = 1*x0 
    Xsim_lqr = [zeros(nx) for i = 1:N]
    Xsim_lqr[1] = 1*x0 
    for i = 1:N-1
        # sampled noise to be added after each step 
        noise = [.005*randn(2);.1*randn(2)]
        
        # simulate cvx control 
        Xsim_cvx[i+1] = A*Xsim_cvx[i] + B*Ucvx[i] + noise
        
        # TODO: use your FHLQR control gains K to calculate u_lqr
        # simulate lqr control
        u_lqr = -K[i]*Xsim_lqr[i]
        Xsim_lqr[i+1] = A*Xsim_lqr[i] + B*u_lqr + noise
    end
    
    # make sure our LQR achieved the goal 
    @test norm(Xsim_cvx[end]) > norm(Xsim_lqr[end])
    @test norm(Xsim_lqr[end]) < .7
    @test norm(Xsim_cvx[end]) > 2.0
    
    
    # ------------------------plotting--------------------------
    X1m = mat_from_vec(Xsim_cvx)
    X2m = mat_from_vec(Xsim_lqr)
    # plot trajectory in x1 x2 space 
    plot(X1m[1,:],X1m[2,:], label = "CVX Trajectory (no replanning)", lw = 4, ls = :dot)
    display(plot!(X2m[1,:],X2m[2,:],
                 title = "Trajectory in State Space (Noisy Dynamics)",
                 ylabel = "x₂", xlabel = "x₁", lw = 2, label = "LQR Trajectory"))
    ecvx = [norm(x[1:2]) for x in Xsim_cvx]
    elqr = [norm(x[1:2]) for x in Xsim_lqr]
    plot(t_vec, elqr, label = "LQR Trajectory",ylabel = "|x - xgoal|",
         xlabel = "time (s)", title = "Error for CVX vs LQR (Noisy Dynamics)")
    display(plot!(t_vec, ecvx, label = "CVX Trajectory (no replanning)"))
    # ------------------------plotting--------------------------

end

@testset "Why LQR is great reason 2" begin 
    
    # problem stuff
    dt = 0.1 
    tf = 20.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    A,B = double_integrator_AB(dt)
    nx,nu = size(B)
    x0 = [5,7,2,-1.4] # initial condition 
    Q = diagm(ones(nx))
    R = diagm(ones(nu))
    Qf = 10*Q 
    
    P, K = fhlqr(A,B,Q,R,Qf,N)
    
    # TODO: specify any goal state with 0 velocity within a 5m radius of 0 
    xgoal = [2,4,0,0]
    @test norm(xgoal[1:2])< 5
    @test norm(xgoal[3:4])<1e-13 # ensure 0 velocity

    Xsim_lqr = [zeros(nx) for i = 1:N]
    Xsim_lqr[1] = 1*x0 
    
    for i = 1:N-1
        # TODO: use your FHLQR control gains K to calculate u_lqr
        # simulate lqr control 
        u_lqr = -K[i]*(Xsim_lqr[i]-xgoal)
        Xsim_lqr[i+1] = A*Xsim_lqr[i] + B*u_lqr
    end
    
    @test norm(Xsim_lqr[end][1:2] - xgoal[1:2]) < .1 
    
    # ------------------------plotting--------------------------
    Xm = mat_from_vec(Xsim_lqr)
    plot(xgoal[1:1],xgoal[2:2],seriestype = :scatter, label = "goal state")
    display(plot!(Xm[1,:],Xm[2,:],
                 title = "Trajectory in State Space",
                 ylabel = "x₂", xlabel = "x₁", lw = 2, label = "LQR Trajectory"))


end

# half vectorization of a matrix 
function vech(A)
    return A[tril(trues(size(A)))]
end
@testset "P and K time analysis" begin 
    
    # problem stuff
    dt = 0.1 
    tf = 10.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    A,B = double_integrator_AB(dt)
    nx,nu = size(B)
    
    # cost terms 
    Q = diagm(ones(nx))
    R = .5*diagm(ones(nu))
    Qf = randn(nx,nx); Qf = Qf'*Qf + I;
    
    P, K = fhlqr(A,B,Q,R,Qf,N)
    
    Pm = hcat(vech.(P)...)
    Km = hcat(vec.(K)...)
    
    # make sure these things converged 
    @test 1e-13 < norm(P[1] - P[2]) < 1e-3 
    @test 1e-13 < norm(K[1] - K[2]) < 1e-3
    
    display(plot(t_vec, Pm', label = "",title = "Cost-to-go Matrix (P)", xlabel = "time(s)"))
    display(plot(t_vec[1:end-1], Km', label = "",title = "Gain Matrix (K)", xlabel = "time(s)"))

    
end

"""
P,K = ihlqr(A,B,Q,R)

TODO: complete this infinite horizon LQR function where 
you do the Riccati recursion until the cost to go matrix 
P converges to a steady value |P_k - P_{k+1}| ≤ tol 
"""
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
@testset "ihlqr test" begin 
    # problem stuff
    dt = 0.1 
    A,B = double_integrator_AB(dt)
    nx,nu = size(B)
    
    # we're just going to modify the system a little bit 
    # so the following graphs are still interesting

    Q = diagm(ones(nx))
    R = .5*diagm(ones(nu))
    P, K = ihlqr(A,B,Q,R)
    
    # check this P is in fact a solution to the Riccati equation
    @test typeof(P) == Matrix{Float64} 
    @test typeof(K) == Matrix{Float64}
    @test 1e-13 < norm(Q + K'*R*K + (A - B*K)'P*(A - B*K) - P) < 1e-3
    
end