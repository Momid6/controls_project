import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
using MeshCat
using Test
using Plots



# include the functions from quadruped.jl
include(joinpath(@__DIR__, "utils/quadruped.jl"))

# this loads in our continuous time dynamics function xdot = dynamics(model, x, u)

# --------these three are global variables------------
model = UnitreeA1() # contains all the model properties for the quadruped
mvis = initialize_visualizer(model) # visualizer 
const x_guess = initial_state(model) # our guess state for balancing
# ----------------------------------------------------

set_configuration!(mvis, x_guess[1:state_dim(model)÷2])
render(mvis)

# initial guess 
const x_guess = initial_state(model)

# indexing stuff 
const idx_x = 1:30 
const idx_u = 31:42
const idx_c = 43:72

#y = [x;u]
# Newton's method will solve for z = [x;u;λ], or z = [y;λ]

function quadruped_cost(y::Vector)
    # cost function 
    @assert length(y) == 42
    x = y[idx_x]
    u = y[idx_u]
    return 0.5*(x-x_guess)'*(x-x_guess) + 0.5*(1e-3)*u'*u
end
function quadruped_constraint(y::Vector)::Vector
    # constraint function 
    @assert length(y) == 42
    x = y[idx_x]
    u = y[idx_u]
    return dynamics(model, x, u)
end
function quadruped_kkt(z::Vector)::Vector
    @assert length(z) == 72 
    x = z[idx_x]
    u = z[idx_u]
    λ = z[idx_c]
    
    y = [x;u]
    ∇f = FD.gradient(quadruped_cost, y)
    ∇c = FD.jacobian(quadruped_constraint, y)
    c_val = quadruped_constraint(y)
    return vcat(∇f + ∇c' * λ , c_val)
end

function quadruped_kkt_jac(z::Vector)::Matrix
    @assert length(z) == 72 
    x = z[idx_x]
    u = z[idx_u]
    λ = z[idx_c]
    
    y = [x;u]
    H = FD.hessian(quadruped_cost, y)
    g = FD.jacobian(quadruped_constraint, y)
    reg = 1e-3*I(30)
    kkt_jac = [H g'; g Matrix(reg)]
    return kkt_jac
end

function quadruped_merit(z)
    # merit function for the quadruped problem 
    @assert length(z) == 72 
    r = quadruped_kkt(z)
    return norm(r[1:42]) + 1e4*norm(r[43:end])
end


function linesearch(z::Vector, Δz::Vector, merit_fx::Function;max_ls_iters = 10)::Float64 # optional argument with a default
    m0 = merit_fx(z)
    for i = 1:max_ls_iter
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

@testset "quadruped standing" begin
    
    z0 = [x_guess; zeros(12); zeros(30)]
    Z = newtons_method(z0, quadruped_kkt, quadruped_kkt_jac, quadruped_merit; tol = 1e-6, max_iters = 50, verbose = true)
    set_configuration!(mvis, Z[end][1:state_dim(model)÷2])
    R = norm.(quadruped_kkt.(Z))
    
    display(plot(1:length(R), R, yaxis=:log,xlabel = "iteration", ylabel = "|r|"))
    
    @test R[end] < 1e-6
    @test length(Z) < 25
    
    x,u = Z[end][idx_x], Z[end][idx_u]
    
    @test norm(dynamics(model, x, u)) < 1e-6
    
end


let
    
    # let's visualize the balancing position we found
    
    z0 = [x_guess; zeros(12); zeros(30)]
    Z = newtons_method(z0, quadruped_kkt, quadruped_kkt_jac, quadruped_merit; tol = 1e-6, verbose = false, max_iters = 50)
    # visualizer 
    mvis = initialize_visualizer(model)
    set_configuration!(mvis, Z[end][1:state_dim(model)÷2])
    render(mvis)
    
    
end

