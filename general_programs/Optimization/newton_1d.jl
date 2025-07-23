import Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate();

using Test, LinearAlgebra
import ForwardDiff as FD 

function newtons_method_1d(x0::Float64, residual_function::Function; max_iters = 50)::Vector{Float64}
    X = zeros(max_iters)
    X[1] = x0 
    for i = 1:max_iters-1 
        Δx = -residual_function(X[i]) / FD.derivative(dx -> residual_function(dx), X[i])
        X[i+1] = Δx + X[i]
        if abs(residual_function(X[i+1])) < 1e-10
            return X[1:i+1]
        end
    end
        return error("Newton did not converge")
end