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
    λ = sqrt(ρ)*exp.(-σ)

    stationarity = qp.Q*x + qp.q + qp.A'* μ - qp.G'* λ
    primal_feasability_1 = qp.A * x - qp.b
    primal_feasability_2 = min.(qp.G*x-qp.h, 0)
    dual_feasibility = min.(λ, 0)
    complementarity = λ .* (qp.h - qp.G * x) 
    return vcat(stationarity, primal_feasability_1, primal_feasability_2, dual_feasibility, complementarity)
end


function ip_kkt_conditions(qp::NamedTuple, z::Vector, ρ::Float64)::Vector
    x, μ, σ = z[qp.xi], z[qp.μi], z[qp.σi]

    λ = sqrt(ρ)*exp.(-σ)
    s = sqrt(ρ)*exp.(σ)
    stationarity = qp.Q*x + qp.q + qp.A'* μ - qp.G'* λ
    primal_feasability_1 = qp.A * x - qp.b
    primal_feasability_2 = qp.G*x- qp.h - s

    return vcat(stationarity, primal_feasability_1, primal_feasability_2)
end

function ip_kkt_jac(qp::NamedTuple, z::Vector, ρ::Float64)::Matrix
    x, μ, σ = z[qp.xi], z[qp.μi], z[qp.σi]
    λ = sqrt(ρ)*exp.(-σ)
    s = sqrt(ρ)*exp.(σ)
    n = length(x)
    m = length(μ)
    p = length(σ)
    return [qp.Q qp.A' qp.G'*Diagonal(λ);
        qp.A zeros(m,m) zeros(m,p);
        qp.G zeros(p,m) -Diagonal(s)
        ]
end


function solve_qp(qp; verbose = false, max_iters = 100, tol = 1e-8)
    z = zeros(length(qp.q) + length(qp.b) + length(qp.h))
    
    ρ = 0.1
    for main_iter = 1:max_iters 

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
        

        if norm(kkt_conditions(qp, z, ρ), Inf) < tol
            x, μ, λ = z[qp.xi], z[qp.μi], sqrt(ρ).*exp.(-z[qp.σi])
            return x, μ, λ
        elseif norm(ip_kkt_conditions(qp, z, ρ), Inf) < tol
            ρ = ρ * 0.1
        end

    end
end

function qp_data_create(state_variables, eq_constraint_size, inequality_const_size)
    qp = (
        Q = [1 .3 0 0;
             0.3 1 0 0;
             0 0 2 0;
             0 0 0 4 ], 
        q = [-2, 3.4, 2, 4],
        A = [0 0 1 1;
             -1 2.3 1 -2], 
        b = [1; 3], 
        G = [-diagm(ones(4)); diagm(ones(4))],
        h = [-1; -1; -1; -1; -1; -0.5; -0.5; -1],
        xi = 1:state_variables,     
        μi = state_variables+1:state_variables+eq_constraint_size,   
        σi = state_variables+eq_constraint_size+1:state_variables+eq_constraint_size+inequality_const_size
    )
    
    return qp 
end

function qp_data()
    state_variables = 4
    eq_constraint_size = 2
    inequality_constraint_size = 8
    return qp_data_create(state_variables, eq_constraint_size, inequality_constraint_size)
end
