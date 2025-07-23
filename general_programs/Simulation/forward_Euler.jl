import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


function forward_euler(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    return x + dt*dynamics(params, x)
end


function simulate_forward_euler(params::NamedTuple,dynamics::Function,x0::Vector,dt::Real,tf::Real)
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    for k = 1:N-1
        X[k+1] = forward_euler(params, dynamics, X[k], dt)
    end
    return X
end