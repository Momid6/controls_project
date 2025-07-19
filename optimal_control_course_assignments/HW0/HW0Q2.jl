import Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate();

using Test, LinearAlgebra
import ForwardDiff as FD 
import FiniteDiff as FD2
using Plots


#Q2a
function newtons_method_1d(x0::Float64, residual_function::Function; max_iters = 10)::Vector{Float64}
    # return the history of iterates as a 1d vector (Vector{Float64})
    # consider convergence to be when abs(residual_function(X[i])) < 1e-10 
    # at this point, trim X to be X = X[1:i], and return X 

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


     

#Q2b



function newtons_method(x0::Vector{Float64}, residual_function::Function; max_iters = 10)::Vector{Vector{Float64}}
    # return the history of iterates as a vector of vectors (Vector{Vector{Float64}})
    # consider convergence to be when norm(residual_function(X[i])) < 1e-10 
    # at this point, trim X to be X = X[1:i], and return X 

    X = [zeros(length(x0)) for i = 1:max_iters]
    X[1] = x0 
    
    for i = 1:max_iters-1 
        Δx = -FD.jacobian(dx -> residual_function(dx), X[i])\residual_function(X[i])
        X[i+1] = Δx + X[i]
        if norm(residual_function(X[i+1])) < 1e-10
            return X[1:i+1]
        end
    end
        return error("Newton did not converge")
end
     


@testset "2a" begin
    # residual function 
    residual_fx(_x) = sin(_x)*_x^2
    
    x0 = 2.8 
    X = newtons_method_1d(x0, residual_fx; max_iters = 10)
    R = residual_fx.(X) # the . evaluates the function at each element of the array
    
    @test abs(R[end]) < 1e-10
    
    # plotting
    display(plot(abs.(R),yaxis=:log,ylabel = "|r|",xlabel = "iteration",
         yticks= [1.0*10.0^(-x) for x = float(15:-1:-2)],
         title = "Convergence of Newton's Method (1D case)",label = ""))
    
end

@testset "2b" begin 
    # residual function 
    r(x) = [sin(x[3] + 0.3)*cos(x[2]- 0.2) - 0.3*x[1];
            cos(x[1]) + sin(x[2]) + tan(x[3]);
            3*x[1] + 0.1*x[2]^3]
    
    x0 = [.1;.1;0.1]
    X = newtons_method(x0, r; max_iters = 10)
    R = r.(X) # the . evaluates the function at each element of the array

    Rp = [[abs(R[i][ii]) for i = 1:length(R)] for ii = 1:3] # this gets abs of each term at each iteration
    
    # tests 
    @test norm(R[end])<1e-10 
    
    # convergence plotting 
    plot(Rp[1],yaxis=:log,ylabel = "|r|",xlabel = "iteration",
         yticks= [1.0*10.0^(-x) for x = float(15:-1:-2)],
         title = "Convergence of Newton's Method (3D case)",label = "|r_1|")
    plot!(Rp[2],label = "|r_2|")
    display(plot!(Rp[3],label = "|r_3|"))

end


@testset "2c" begin 
    Q = [1.65539  2.89376; 2.89376  6.51521];
    q = [2;-3]
    f(x) = 0.5*x'*Q*x + q'*x + exp(-1.3*x[1] + 0.3*x[2]^2)
    
    function kkt_conditions(x)
        return FD.gradient(dx -> f(dx), x)
    end
    
    residual_fx(_x) = kkt_conditions(_x)

    x0 = [-0.9512129986081451, 0.8061342694354091]
    X = newtons_method(x0, residual_fx; max_iters = 10)
    R = residual_fx.(X) # the . evaluates the function at each element of the array

    Rp = [[abs(R[i][ii]) for i = 1:length(R)] for ii = 1:length(R[1])] # this gets abs of each term at each iteration
    
    # tests 
    @test norm(R[end])<1e-10; 

    plot(Rp[1],yaxis=:log,ylabel = "|r|",xlabel = "iteration",
         yticks= [1.0*10.0^(-x) for x = float(15:-1:-2)],
         title = "Convergence of Newton's Method on KKT Conditions",label = "|r_1|")
    display(plot!(Rp[2],label = "|r_2|"))
    
end