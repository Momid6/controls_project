import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra
import ForwardDiff as FD 


#Doesn't save memory
function newtons_method(x0::Vector{Float64}, residual_function::Function; max_iters = 50, tol=1e-13)::Vector{Float64}

    x_current = x0
    for i = 1:max_iters-1
        
        Δx = -FD.jacobian(dx -> residual_function(dx), x_current)\residual_function(x_current)
        x_current = Δx + x_current
        if norm(residual_function(x_current)) < tol
            return x_current
        end
    end
        return error("Newton did not converge")
end

function backward_euler(params::NamedTuple, dynamics::Function, x1::Vector, x2::Vector, dt::Real)::Vector{Float64}
   return x1 + dt*dynamics(params, x2) - x2
end

function implicit_integrator_solve(params::NamedTuple, dynamics::Function, x1::Vector, dt::Real;tol = 1e-13, max_iters = 10)::Vector{Float64}
    # initialize guess
    x2 = 1*x1
    residual_function = x2 -> backward_euler(params, dynamics, x1, x2, dt)
    x2 = newtons_method(x2, residual_function; max_iters, tol)
    return x2
end    

function simulate_backward_euler(params::NamedTuple,dynamics::Function,x0::Vector,dt::Real,tf::Real; tol = 1e-13)
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    for k = 1:N-1
    X[k+1] = implicit_integrator_solve(params, dynamics, X[k], dt;tol)
    end
    return X
end