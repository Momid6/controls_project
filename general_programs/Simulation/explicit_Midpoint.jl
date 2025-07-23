import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


function midpoint(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    x_m = x + (dt*dynamics(params, x))/2
    x_new = x + dt*dynamics(params, x_m)
    return x_new
end


function simulate_midpoint_explicit(params::NamedTuple,dynamics::Function,x0::Vector,dt::Real,tf::Real)
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    
    for k = 1:N-1
        X[k+1] = midpoint(params, dynamics, X[k], dt)
    end
    return X
end
