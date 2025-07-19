import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
import MeshCat as mc
using Test

# these two functions are given, no TODO's here 
function double_pendulum_dynamics(params::NamedTuple, x::Vector)
    # continuous time dynamics for a double pendulum given state x,
    # also known as the "equations of motion". 
    # returns the time derivative of the state, ẋ (dx/dt)

    # the state is the following:
    θ1,θ̇1,θ2,θ̇2 = x

    # system parameters
    m1, m2, L1, L2, g = params.m1, params.m2, params.L1, params.L2, params.g

    # dynamics
    c = cos(θ1-θ2)
    s = sin(θ1-θ2)

    ẋ = [
        θ̇1;
        ( m2*g*sin(θ2)*c - m2*s*(L1*c*θ̇1^2 + L2*θ̇2^2) - (m1+m2)*g*sin(θ1) ) / ( L1 *(m1+m2*s^2) );
        θ̇2;
        ((m1+m2)*(L1*θ̇1^2*s - g*sin(θ2) + g*sin(θ1)*c) + m2*L2*θ̇2^2*s*c) / (L2 * (m1 + m2*s^2));
        ]

    return ẋ
end
function double_pendulum_energy(params::NamedTuple, x::Vector)::Real
    # calculate the total energy (kinetic + potential) of a double pendulum given a state x 


    # the state is the following:
    θ1,θ̇1,θ2,θ̇2 = x

    # system parameters
    m1, m2, L1, L2, g = params.m1, params.m2, params.L1, params.L2, params.g

    # cartesian positions/velocities of the masses
    r1 = [L1*sin(θ1), 0, -params.L1*cos(θ1) + 2]
    r2 = r1 + [params.L2*sin(θ2), 0, -params.L2*cos(θ2)]
    v1 = [L1*θ̇1*cos(θ1), 0, L1*θ̇1*sin(θ1)]
    v2 = v1 + [L2*θ̇2*cos(θ2), 0, L2*θ̇2*sin(θ2)]

    # energy calculation
    kinetic = 0.5*(m1*v1'*v1 + m2*v2'*v2)
    potential = m1*g*r1[3] + m2*g*r2[3]
    return kinetic + potential
end

#Forward_Euler


function forward_euler(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    return x + dt*dynamics(params, x)
end


function midpoint(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    x_m = x + (dt*dynamics(params, x))/2
    x_new = x + dt*dynamics(params, x_m)
    return x_new
end
function rk4(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    # TODO: implement RK4
    k1 = dt * dynamics(params,x)
    k2 = dt * dynamics(params, x+k1/2)
    k3 = dt * dynamics(params, x+k2/2)
    k4 = dt* dynamics(params, x+k3)
    x_new = x + (1/6)*(k1+2*k2+2*k3+k4)
    return x_new
end


function simulate_explicit(params::NamedTuple,dynamics::Function,integrator::Function,x0::Vector,dt::Real,tf::Real)
    # TOOD: update this function to simulate dynamics forward
    # with the given explicit integrator 
    
    # take in 
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    
    # TODO: simulate X forward
    for k = 1:N-1
        X[k+1] = integrator(params, dynamics, X[k], dt)
    end
    # return state history X and energy E 
    E = [double_pendulum_energy(params,x) for x in X]
    return X, E
end


include(joinpath(@__DIR__, "animation.jl"))

@testset "forward euler" begin

    # parameters for the simulation
    params = (
        m1 = 1.0,
        m2 = 1.0,
        L1 = 1.0,
        L2 = 1.0,
        g = 9.8
    )

    # initial condition
    x0 = [pi/1.6; 0; pi/1.8; 0]

    # time step size (s)
    dt = 0.01
    tf = 30.0 
    t_vec = 0:dt:tf
    N = length(t_vec)
    # store the trajectory in a vector of vectors
    X = [zeros(4) for i = 1:N]
    X[1] = 1*x0 

    # TODO: simulate the double pendulum with `forward_euler` 
    # X[k] = `x_k`, so X[k+1] = forward_euler(params, double_pendulum_dynamics, X[k], dt)
    for k=1:N-1
       X[k+1] = forward_euler(params, double_pendulum_dynamics, X[k], dt)
    end
    
    # calculate energy 
    E = [double_pendulum_energy(params,x) for x in X]

    @test norm(X[end]) > 1e-10   # make sure all X's were updated
    @test 2 < (E[end]/E[1]) < 3  # energy should be increasing

    # plot state history, energy history, and animate it
    display(plot(t_vec, hcat(X...)',xlabel = "time (s)", label = ["θ₁" "θ̇₁ dot" "θ₂" "θ̇₂ dot"]))
    display(plot(t_vec, E, xlabel = "time (s)", ylabel = "energy (J)"))
    meshcat_animate(params,X,dt,N)
    
    
end

@testset "RK4" begin

    # parameters for the simulation
    params = (
        m1 = 1.0,
        m2 = 1.0,
        L1 = 1.0,
        L2 = 1.0,
        g = 9.8
    )

    # initial condition
    x0 = [pi/1.6; 0; pi/1.8; 0]

    # time step size (s)
    dt = 0.01
    tf = 30.0
    t_vec = 0:dt:tf
    N = length(t_vec)

    X, E = simulate_explicit(params, double_pendulum_dynamics, rk4, x0,dt,tf)

    @test norm(X[end]) > 1e-10   # make sure all X's were updated
    @test abs(E[end]-E[1]) < 1e-2  # energy should be conserved

    # plot state history, energy history, and animate it
    display(plot(t_vec, hcat(X...)',xlabel = "time (s)", label = ["θ₁" "θ̇₁ dot" "θ₂" "θ̇₂ dot"]))
    display(plot(t_vec, E, xlabel = "time (s)", ylabel = "energy (J)"))
    meshcat_animate(params,X,dt,N)
    
    
end

@testset "Midpoint" begin

    # parameters for the simulation
    params = (
        m1 = 1.0,
        m2 = 1.0,
        L1 = 1.0,
        L2 = 1.0,
        g = 9.8
    )

    # initial condition
    x0 = [pi/1.6; 0; pi/1.8; 0]

    # time step size (s)
    dt = 0.01
    tf = 30.0
    t_vec = 0:dt:tf
    N = length(t_vec)

    X, E = simulate_explicit(params, double_pendulum_dynamics, midpoint, x0,dt,tf)

    @test norm(X[end]) > 1e-10   # make sure all X's were updated
    @test abs(E[end]-E[1]) < 1  # energy should be conserved

    # plot state history, energy history, and animate it
    display(plot(t_vec, hcat(X...)',xlabel = "time (s)", label = ["θ₁" "θ̇₁ dot" "θ₂" "θ̇₂ dot"]))
    display(plot(t_vec, E, xlabel = "time (s)", ylabel = "energy (J)"))
    meshcat_animate(params,X,dt,N)
    
    
end