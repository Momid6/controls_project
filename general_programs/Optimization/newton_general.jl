import Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate();

using Test, LinearAlgebra
import ForwardDiff as FD 

#Doesn't save memory
function newtons_method_(x0::Vector{Float64}, residual_function::Function; max_iters = 50, tol=1e-10)::Vector{Float64}

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

#Saves memory
function newtons_method(x0::Vector{Float64}, residual_function::Function; max_iters = 50, tol=1e-10)::Vector{Vector{Float64}}
    X = [zeros(length(x0)) for i = 1:max_iters]
    X[1] = x0 
    
    for i = 1:max_iters-1 
        Δx = -FD.jacobian(dx -> residual_function(dx), X[i])\residual_function(X[i])
        X[i+1] = Δx + X[i]
        if norm(residual_function(X[i+1])) < tol
            return X[1:i+1]
        end
    end
        return error("Newton did not converge")
end