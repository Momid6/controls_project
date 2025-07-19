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

function qp_data()
    
    # TODO: fill in the QP problem data for a simulation step 
    # fill in Q, q, G, h, but leave A, b the same 
    # this is because there are no equality constraints in this qp 
    qp = (
        Q = [1 .3 0 0;
             0.3 1 0 0;
             0 0 2 0;
             0 0 0 4 ], 
        q = [-2, 3.4, 2, 4],
        A = [0 0 1 1;
             -1 2.3 1 -2], # don't edit this
        b = [1; 3],   # don't edit this 
        G = [-diagm(ones(4)); diagm(ones(4))],
        h = [-1; -1; -1; -1; -1; -0.5; -0.5; -1],
        xi = 1:4,       # don't edit this
        μi = 5:6,        # don't edit this
        σi = 7:14       # don't edit this
    )
    
    return qp 
end


@testset "part D" begin

    y = randn(2)
    a = randn()
    b = randn()
    
    #TODO: Create your qp and solve it. Don't forget the indices (xi, μi, and σi)
    x, mu, sigma = solve_qp(qp_data(), verbose=true, tol = 1e-6)
    y = x[1:2]
    a = x[3]
    b = x[4]
    
    @test norm(y - [-0.080823; 0.834424]) < 1e-3 
    @test abs(a - 1) < 1e-3 
    @test abs(b) < 1e-3 
end