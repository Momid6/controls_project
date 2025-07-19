
import Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate();
using LinearAlgebra
using Test
Pkg.add("ForwardDiff")
import ForwardDiff as FD 

#Q1a
function foo4(x)
    Q = diagm([1;2;3.0]) # this creates a diagonal matrix from a vector
    return 0.5*x'*Q*x/x[1] - log(x[1])*exp(x[2])^x[3] 
end

function foo4_expansion(x)
    ∇²foo4 = FD.hessian(foo4, x)
    ∇foo4 = FD.gradient(foo4, x)

    g = zeros(length(x))
    g = ∇foo4

    H = zeros(length(x),length(x))
    H = ∇²foo4
    
    return g, H
end
     


#Q1b
function eulers(x,u,J)
    # dynamics when x is angular velocity and u is an input torque
    ẋ = J\(u - cross(x,J*x))
    return ẋ
end

function eulers_jacobians(x,u,J)
    # given x, u, and J, calculate the following two jacobians 
    
    # TODO: fill in the following two jacobians
    
    # ∂ẋ/∂x
    A = zeros(3,3)
    A = FD.jacobian(dx -> eulers(dx,u,J), x)
    
    # ∂ẋ/∂u
    B = zeros(3,3)
    B = FD.jacobian(du -> eulers(x,du,J), u)
    
    return A, B
end


#Q1c

function f2(x)
    return x*sin(x)/2
end
function g2(x)
    return cos(x)^2 - tan(x)^3
end

function composite_derivs(x)
    # TODO: return ∂y/∂x where y = g2(f2(x)) 
    # (hint: this is 1D input and 1D output, so it's ForwardDiff.derivative)
    return FD.derivative(dx -> g2(f2(dx)), x)
end    

#Q1D

# TODO: fix this error when trying to diff through this function
# hint: you can use promote_type(eltype(x),eltype(u)) to return the correct type if either x or u is a ForwardDiff.Dual (option 1)

function dynamics(x,u)
    xdot = zeros(promote_type(eltype(x),eltype(u)), length(x))
    xdot[1] = x[1]*sin(u[1])
    xdot[2] = x[2]*cos(u[2])
    return xdot
end
     

@testset "1a" begin                        
    x = [.2;.4;.5]
    g,H = foo4_expansion(x)
    @test isapprox(g,[-18.98201379080085, 4.982885952667278, 8.286308762133823],atol = 1e-8)        
    @test norm(H -[164.2850689540042 -23.053506895400425 -39.942805516320334;
                             -23.053506895400425 10.491442976333639 2.3589262864014673;
                             -39.94280551632034 2.3589262864014673 15.314523504853529]) < 1e-8 
end

@testset "1b" begin                                                
    
    x = [.2;-7;.2]
    u = [.1;-.2;.343]
    J = diagm([1.03;4;3.45])
    
    A,B = eulers_jacobians(x,u,J)

    skew(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    @test isapprox(A,-J\(skew(x)*J - skew(J*x)), atol = 1e-8)  

    @test norm(B - inv(J)) < 1e-8                

end

@testset "1c" begin                                           
    x = 1.34 
    deriv = composite_derivs(x)

    @test isapprox(deriv,-2.390628273373545,atol = 1e-8)  
end

@testset "1d" begin                                     
    x = [.1;.4]
    u = [.2;-.3]
    A = FD.jacobian(_x -> dynamics(_x,u),x) 
    B = FD.jacobian(_u -> dynamics(x,_u),u) 
    @test typeof(A) == Matrix{Float64}                  
    @test typeof(B) == Matrix{Float64}                  
end