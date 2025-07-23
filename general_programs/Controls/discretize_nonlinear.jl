import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra
import ForwardDiff as FD


function rk4(dynamics::Function, params::NamedTuple, x::Vector,u,dt::Float64)
    # vanilla RK4
    k1 = dt*dynamics(params, x, u)
    k2 = dt*dynamics(params, x + k1/2, u)
    k3 = dt*dynamics(params, x + k2/2, u)
    k4 = dt*dynamics(params, x + k3, u)
    return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end

function discretize_nonlinear(dynamics::Function, params::NamedTuple, x::Vector, u::Vector, dt::Float64)
    A = FD.jacobian(dx -> rk4(dynamics, params, dx, u, dt), x)
    B = FD.jacobian(du -> rk4(dynamics, params, x, du, dt), u)
    return A, B
end
