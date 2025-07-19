import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
using Printf
using JLD2
using Test

function c_eq(qp::NamedTuple, x::Vector)::Vector
    qp.A*x - qp.b 
end
function h_ineq(qp::NamedTuple, x::Vector)::Vector
    qp.G*x - qp.h
end
function kkt_conditions(qp::NamedTuple, z::Vector, ρ::Float64)::Vector
    x, μ, σ = z[qp.xi], z[qp.μi], z[qp.σi]

    # TODO compute λ from σ and ρ
    λ = sqrt(ρ)*exp.(-σ)

    # TODO compute and return KKT conditions
    stationarity = qp.Q*x + qp.q + qp.A'* μ - qp.G'* λ
    primal_feasability_1 = qp.A * x - qp.b
    primal_feasability_2 = min.(qp.G*x-qp.h, 0)
    dual_feasibility = min.(λ, 0)
    complementarity = λ .* (qp.h - qp.G * x) 
    return vcat(stationarity, primal_feasability_1, primal_feasability_2, dual_feasibility, complementarity)
end


function ip_kkt_conditions(qp::NamedTuple, z::Vector, ρ::Float64)::Vector
    x, μ, σ = z[qp.xi], z[qp.μi], z[qp.σi]

    # TODO compute λ and s from σ and ρ
    λ = sqrt(ρ)*exp.(-σ)
    s = sqrt(ρ)*exp.(σ)
    # TODO compute and return IP KKT conditions
    stationarity = qp.Q*x + qp.q + qp.A'* μ - qp.G'* λ
    primal_feasability_1 = qp.A * x - qp.b
    primal_feasability_2 = qp.G*x- qp.h - s

    return vcat(stationarity, primal_feasability_1, primal_feasability_2)
end

""
function ip_kkt_jac(qp::NamedTuple, z::Vector, ρ::Float64)::Matrix
    x, μ, σ = z[qp.xi], z[qp.μi], z[qp.σi]
    λ = sqrt(ρ)*exp.(-σ)
    s = sqrt(ρ)*exp.(σ)
    n = length(x)
    m = length(μ)
    p = length(σ)
    # TODO: return full Newton jacobian (don't use ForwardDiff)
    return [qp.Q qp.A' qp.G'*Diagonal(λ);
        qp.A zeros(m,m) zeros(m,p);
        qp.G zeros(p,m) -Diagonal(s)
        ]
end


function logging(qp::NamedTuple, main_iter::Int, z::Vector, ρ::Real, α::Real)
    x, μ, σ = z[qp.xi], z[qp.μi], z[qp.σi]

    # TODO: compute λ
    λ = sqrt(ρ)*exp.(-σ)

    # TODO: stationarity norm
    stationarity_norm =  norm(qp.Q*x + qp.q + qp.A'* μ - qp.G'* λ)

    @printf("%3d  % 7.2e  % 7.2e  % 7.2e  % 7.2e  %5.0e  %5.0e\n",
          main_iter, stationarity_norm, minimum(h_ineq(qp,x)),
          norm(c_eq(qp,x),Inf), abs(dot(λ,h_ineq(qp,x))), ρ, α)
end


function solve_qp(qp; verbose = true, max_iters = 100, tol = 1e-8)
    # Init solution vector z = [x; μ; σ]
    z = zeros(length(qp.q) + length(qp.b) + length(qp.h))

    if verbose
        @printf "iter   |∇Lₓ|      min(h)       |c|       compl     ρ      α\n"
        @printf "----------------------------------------------------------------\n"
    end

    # TODO: implement your solver according to the above pseudocode
    ρ = 0.1
    for main_iter = 1:max_iters 

        # TODO: make sure to save the step length (α) from your linesearch for logging
        ip_res = ip_kkt_conditions(qp, z, ρ)
        ip_jac = ip_kkt_jac(qp, z, ρ)
        Δz = -ip_jac\ip_res
        α = 1
        for i = 1:10
            if norm(ip_kkt_conditions(qp, z +  α*Δz, ρ)) < norm(ip_res)
                break
            end
            α = α/2
        end
        z = z + α*Δz
        
        if verbose
            logging(qp, main_iter, z, ρ, α) 
        end

        if norm(kkt_conditions(qp, z, ρ), Inf) < tol
            x, μ, λ = z[qp.xi], z[qp.μi], sqrt(ρ).*exp.(-z[qp.σi])
            return x, μ, λ
        elseif norm(ip_kkt_conditions(qp, z, ρ), Inf) < tol
            ρ = ρ * 0.1
        end

    end
end

function brick_simulation_qp(pos, v; mass = 1.0, Δt = 0.01)
    
    # TODO: fill in the QP problem data for a simulation step 
    # fill in Q, q, G, h, but leave A, b the same 
    # this is because there are no equality constraints in this qp 
    g = [0; 9.81]
    J = [0 1]
    qp = (
        Q = Matrix(mass*I(2)), 
        q = mass*I(2)*(Δt*g-v),
        A = zeros(0,2), # don't edit this
        b = zeros(0),   # don't edit this 
        G = J*Δt,
        h = -J*pos,
        xi = 1:2,       # don't edit this
        μi = [],        # don't edit this
        σi = 3:3        # don't edit this
    )
    
    return qp 
end


@testset "brick qp" begin 
    
    q = [1,3.0]
    v = [2,-3.0]
    
    qp = brick_simulation_qp(q,v)
    
    # check all the types to make sure they're right
    qp.Q::Matrix{Float64}
    qp.q::Vector{Float64}
    qp.A::Matrix{Float64}
    qp.b::Vector{Float64}
    qp.G::Matrix{Float64}
    qp.h::Vector{Float64}
    
    @test size(qp.Q) == (2,2)
    @test size(qp.q) == (2,)
    @test size(qp.A) == (0,2)
    @test size(qp.b) == (0,)
    @test size(qp.G) == (1,2)
    @test size(qp.h) == (1,)
    
    @test abs(tr(qp.Q) - 2) < 1e-10
    @test norm(qp.q - [-2.0, 3.0981]) < 1e-10 
    @test norm(qp.G - [0 .01]) < 1e-10 
    @test abs(qp.h[1] - -3) < 1e-10
    
end

include(joinpath(@__DIR__, "animate_brick.jl"))
let 
    
    dt = 0.01 
    T = 3.0 
    
    t_vec = 0:dt:T
    N = length(t_vec)
    
    qs = [zeros(2) for i = 1:N]
    vs = [zeros(2) for i = 1:N]
    
    qs[1] = [0, 1.0]
    vs[1] = [1, 4.5]
    
    # TODO: simulate the brick by forming and solving a qp 
    # at each timestep. Your QP should solve for vs[k+1], and
    # you should use this to update qs[k+1]
    for k=1:N-1
        qp = brick_simulation_qp(qs[k], vs[k])
        vs[k+1], mu, lambda = solve_qp(qp, verbose=false, tol=1e-4)
        qs[k+1] = qs[k] + dt*vs[k+1]
    end

    
    xs = [q[1] for q in qs]
    ys = [q[2] for q in qs]
    
    @show @test abs(maximum(ys)-2)<1e-1
    @show @test minimum(ys) > -1e-2
    @show @test abs(xs[end] - 3) < 1e-2
    
    xdot = diff(xs)/dt
    @show @test maximum(xdot) < 1.0001
    @show @test minimum(xdot) > 0.9999
    @show @test ys[110] > 1e-2
    @show @test abs(ys[111]) < 1e-2
    @show @test abs(ys[112]) < 1e-2
    
    display(plot(xs, ys, ylabel = "y (m)", xlabel = "x (m)"))
    
    animate_brick(qs)
    
    
    
end