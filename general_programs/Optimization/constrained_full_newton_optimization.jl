import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
using MeshCat
using Test
using Plots


function linesearch(z::Vector, Δz::Vector, merit_fx::Function;max_ls_iters = 10)::Float64 # optional argument with a default
    α = 1 
    m0 = merit_fx(z)
    for i = 1:max_ls_iters
        if merit_fx(z + α*Δz) < m0
            return α
        end
        α = α/2
    end
    return α
end

function newtons_method(z0::Vector, res_fx::Function, res_jac_fx::Function, merit_fx::Function;tol = 1e-10, max_iters = 50, verbose = false)::Vector{Vector{Float64}}
    Z = Vector{Vector{Float64}}()
    z_current = z0
    push!(Z, z_current)
    for i = 1:max_iters
        Δz = -res_jac_fx(z_current)\res_fx(z_current)
        α = linesearch(z_current, Δz, merit_fx)
        z_current = z_current + α*Δz
        push!(Z, z_current)
        if norm(res_fx(z_current)) < tol
            return Z
        end
    end
    return Z
end

function constraint_jacobian(x::Vector, constraint_fx::Function)::Matrix
    J = FD.jacobian(constraint_fx, x)
    @assert size(J) == (length(constraint_fx(x)), length(x))
    return J 
end

function kkt_conditions(z::Vector, decision_vars::Int64, cost_fx::Function, constraint_fx::Function)::Vector
    n = decision_vars
    x = z[1:n]
    λ = z[n+1:end]
    ∇f = FD.gradient(cost_fx, x)
    ∇c = FD.jacobian(dx -> constraint_fx(dx), x)
    c_val = constraint_fx(x)
    return vcat(∇f + ∇c' * λ, c_val)
end

function fn_kkt_jac(z::Vector, decision_vars::Int64, constraint_vars::Int64, cost_fx::Function, constraint_fx::Function)::Matrix
    n = decision_vars
    m = constraint_vars
    x = z[1:n]
    λ = z[n+1:end]
    J = FD.jacobian(dz -> kkt_conditions(dz, decision_vars, cost_fx, constraint_fx), z)
    J[n+1:end, n+1:end] += 1e-3*I(m)
    return J 
end

function full_newton(z0::Vector, cost_fx::Function, decision_vars::Int64, constraint_fx::Function, constraint_vars::Int64, merit_fx::Function)
    kkt(_z) = kkt_conditions(_z, decision_vars, cost_fx, constraint_fx)
    kkt_jac(_z) = fn_kkt_jac(_z, decision_vars, constraint_vars, cost_fx, constraint_fx)
    Z = newtons_method(z0, kkt, kkt_jac, merit_fx; tol = 1e-6, max_iters = 100, verbose = true)
    return Z
end

