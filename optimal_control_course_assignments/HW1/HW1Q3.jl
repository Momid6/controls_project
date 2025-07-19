import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
using Printf
using JLD2


# TODO: read below
# NOTE: DO NOT USE A WHILE LOOP ANYWHERE
"""
The data for the QP is stored in `qp` the following way:
    @load joinpath(@__DIR__, "qp_data.jld2") qp 

which is a NamedTuple, where
    Q, q, A, b, G, h, xi, μi, σi = qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h

contains all of the problem data you will need for the QP.

Your job is to make the following functions where z = [x; μ; σ], λ = sqrt(ρ).*exp.(-σ), and s = sqrt(ρ).*exp.(σ)
    
    kkt_res = kkt_conditions(qp, z, ρ)
    ip_res = ip_kkt_conditions(qp, z)
    ip_jac = ip_kkt_jacobian(qp, z)
    x, μ, λ = solve_qp(qp; verbose = true, max_iters = 100, tol = 1e-8)

"""

# Helper functions (you can use or not use these)
function c_eq(qp::NamedTuple, x::Vector)::Vector
    qp.A*x - qp.b 
end
function h_ineq(qp::NamedTuple, x::Vector)::Vector
    qp.G*x - qp.h
end

"""
    kkt_res = kkt_conditions(qp, z, ρ)

Return the KKT residual from part A as a vector (make sure to clamp the inequalities!)
In Julia, use the following for elementwise min.
elementwise_min = min.(a, b) # This is elementwise min
scalar_elementwise_min = min.(a, 0) # You can also take an elementwise min with a scalar
"""
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

"""
    ip_res = ip_kkt_conditions(qp, z)

Return the interior point KKT residual from part C as a vector
"""
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

"""
    ip_jac = ip_jacobian(qp, z, ρ)

Return the full Newton jacobian of the interior point KKT conditions (part C) with respect to z
Construct it analytically (don't use auto differentiation)
"""
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

"""
    x, μ, λ = solve_qp(qp; verbose = true, max_iters = 100, tol = 1e-8)

Solve the QP using the method defined in the pseudocode above, where z = [x; μ; σ], λ = sqrt(ρ).*exp.(-σ), and s = sqrt(ρ).*exp.(σ)
"""
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
     

# 10 points 
using Test 
@testset "qp solver" begin 
    @load joinpath(@__DIR__, "qp_data.jld2") qp 
    x, λ, μ = solve_qp(qp; verbose = true, max_iters = 100, tol = 1e-6)
    
    @load joinpath(@__DIR__, "qp_solutions.jld2") qp_solutions
    @test norm(kkt_conditions(qp, qp_solutions.z, qp_solutions.ρ))<1e-3;
    @test norm(ip_kkt_conditions(qp, qp_solutions.z, qp_solutions.ρ))<1e-3;
    @test norm(ip_kkt_jac(qp, qp_solutions.z, qp_solutions.ρ) - FD.jacobian(dz -> ip_kkt_conditions(qp, dz, qp_solutions.ρ), qp_solutions.z), Inf) < 1e-3
    @test norm(x - qp_solutions.x,Inf)<1e-3;
    @test norm(λ - qp_solutions.λ,Inf)<1e-3;
    @test norm(μ - qp_solutions.μ,Inf)<1e-3;
end
     