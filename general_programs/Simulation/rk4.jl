import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


function rk4(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    # TODO: implement RK4
    k1 = dt * dynamics(params,x)
    k2 = dt * dynamics(params, x+k1/2)
    k3 = dt * dynamics(params, x+k2/2)
    k4 = dt* dynamics(params, x+k3)
    x_new = x + (1/6)*(k1+2*k2+2*k3+k4)
    return x_new
end


function simulate_rk4(params::NamedTuple,dynamics::Function,x0::Vector,dt::Real,tf::Real)
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    for k = 1:N-1
        X[k+1] = rk4(params, dynamics, X[k], dt)
    end
    return X
end