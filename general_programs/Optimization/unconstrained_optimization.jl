import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
using Printf
using JLD2



function constraint_jacobian(x::Vector, constraint_fx::Function)::Matrix
    J = FD.jacobian(constraint_fx, x)
    @assert size(J) == (length(constraint_fx(x)), length(x))
    return J 
end

function kkt_conditions(z::Vector, decision_vars::Int64, eq_constraint_vars::Int64, ineq_constraint_vars::Int64, cost_fx::Function, eq_constraint_fx::Function, ineq_constraint_fx::Function, ρ)::Vector
    n = decision_vars
    m = eq_constraint_vars
    p = ineq_constraint_vars
    x = z[1:n]
    μ = z[n+1:n+m]
    σ = z[n+m+1:n+m+p]
    ∇f = FD.gradient(cost_fx, x)
    ∇c = constraint_jacobian(x, eq_constraint_fx)
    ∇d = constraint_jacobian(x, ineq_constraint_fx)
    λ = sqrt(ρ)*exp.(-σ)

    stationarity = ∇f + ∇c' * μ - ∇d'* λ
    primal_feasability_1 = eq_constraint_fx(x)
    primal_feasability_2 = min.(ineq_constraint_fx(x), 0)
    dual_feasibility = min.(λ, 0)
    complementarity = λ .* (-ineq_constraint_fx(x)) 
    return vcat(stationarity, primal_feasability_1, primal_feasability_2, dual_feasibility, complementarity)
end

function ip_kkt_conditions(z::Vector, decision_vars::Int64, eq_constraint_vars::Int64, ineq_constraint_vars::Int64, cost_fx::Function, eq_constraint_fx::Function, ineq_constraint_fx::Function, ρ)::Vector
    n = decision_vars
    m = eq_constraint_vars
    p = ineq_constraint_vars
    x = z[1:n]
    μ = z[n+1:n+m]
    σ = z[n+m+1:n+m+p]
    ∇f = FD.gradient(cost_fx, x)
    ∇c = constraint_jacobian(x, eq_constraint_fx)
    ∇d = constraint_jacobian(x, ineq_constraint_fx)

    λ = sqrt(ρ)*exp.(-σ)
    s = sqrt(ρ)*exp.(σ)
 
    stationarity = ∇f + ∇c' * μ - ∇d'* λ
    primal_feasability_1 = eq_constraint_fx(x)
    primal_feasability_2 = ineq_constraint_fx(x) - s

    return vcat(stationarity, primal_feasability_1, primal_feasability_2)
end

function ip_kkt_jac(z::Vector, decision_vars::Int64, eq_constraint_vars::Int64, ineq_constraint_vars::Int64, cost_fx::Function, eq_constraint_fx::Function, ineq_constraint_fx::Function, ρ)::Matrix
    J = FD.jacobian(dz -> ip_kkt_conditions(dz, decision_vars, eq_constraint_vars, ineq_constraint_vars, cost_fx, eq_constraint_fx, ineq_constraint_fx, ρ), z)
    return J
end


function interior_point_method(decision_vars::Int64, eq_constraint_vars::Int64, ineq_constraint_vars::Int64, cost_fx::Function, eq_constraint_fx::Function, ineq_constraint_fx; verbose = true, max_iters = 100, tol = 1e-8)
    n = decision_vars
    m = eq_constraint_vars
    p = ineq_constraint_vars
    z = zeros(n+m+p)

    ρ = 0.1
    for main_iter = 1:max_iters 

        ip_res = ip_kkt_conditions(z, decision_vars, eq_constraint_vars, ineq_constraint_vars, cost_fx, eq_constraint_fx, ineq_constraint_fx, ρ)
        ip_jac = ip_kkt_jac(z, decision_vars, eq_constraint_vars, ineq_constraint_vars, cost_fx, eq_constraint_fx, ineq_constraint_fx, ρ)
        Δz = -ip_jac\ip_res
        α = 1
        for i = 1:10
            if norm(ip_kkt_conditions(z +  α*Δz, decision_vars, eq_constraint_vars, ineq_constraint_vars, cost_fx, eq_constraint_fx, ineq_constraint_fx, ρ)) < norm(ip_res)
                break
            end
            α = α/2
        end
        z = z + α*Δz
        

        if norm(kkt_conditions(z, decision_vars, eq_constraint_vars, ineq_constraint_vars, cost_fx, eq_constraint_fx, ineq_constraint_fx, ρ), Inf) < tol
            x = z[1:n]
            μ = z[n+1:n+m]
            σ = z[n+m+1:n+m+p]
            λ = sqrt(ρ)*exp.(-σ)
            return x, μ, λ
        elseif norm(ip_kkt_conditions(z, decision_vars, eq_constraint_vars, ineq_constraint_vars, cost_fx, eq_constraint_fx, ineq_constraint_fx, ρ), Inf) < tol
            ρ = ρ * 0.1
        end

    end
end

