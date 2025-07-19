import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
import MeshCat as mc
using Test
import ForwardDiff as FD 


# initial condition
const x0 = [pi/1.6; 0; pi/1.8; 0]

const params = (
    m1 = 1.0,
    m2 = 1.0,
    L1 = 1.0,
    L2 = 1.0,
    g = 9.8
)

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


# since these are explicit integrators, these function will return the residuals described above
# NOTE: we are NOT solving anything here, simply return the residuals 
function backward_euler(params::NamedTuple, dynamics::Function, x1::Vector, x2::Vector, dt::Real)::Vector
   return x1 + dt*dynamics(params, x2) - x2
end
function implicit_midpoint(params::NamedTuple, dynamics::Function, x1::Vector, x2::Vector, dt::Real)::Vector
    x_half =  0.5*(x1+x2)
    return x1 + dt*dynamics(params, x_half) - x2
end
function hermite_simpson(params::NamedTuple, dynamics::Function, x1::Vector, x2::Vector, dt::Real)::Vector
    x_step = (1/2)*(x1+x2) + (dt/8)*(dynamics(params, x1) - dynamics(params, x2))
    return x1 + (dt/6)*(dynamics(params, x1) + 4*dynamics(params, x_step) + dynamics(params, x2)) - x2
end

# TODO
# this function takes in a dynamics function, implicit integrator function, and x1 
# and uses Newton's method to solve for an x2 that satsifies the implicit integration equations
# that we wrote about in the functions above
function implicit_integrator_solve(params::NamedTuple, dynamics::Function, implicit_integrator::Function, x1::Vector, dt::Real;tol = 1e-13, max_iters = 10)::Vector
    # initialize guess
    x2 = 1*x1
    residual_function = x2 -> implicit_integrator(params, dynamics, x1, x2, dt)
    x2 = newtons_method(x2, residual_function; max_iters, tol)
    return x2
end    
     

function simulate_implicit(params::NamedTuple,dynamics::Function,implicit_integrator::Function,x0::Vector,dt::Real,tf::Real; tol = 1e-13)
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    
    # TODO: do a forward simulation with the selected implicit integrator 
    # hint: use your `implicit_integrator_solve` function
    for k = 1:N-1
    X[k+1] = implicit_integrator_solve(params, double_pendulum_dynamics, implicit_integrator, X[k], dt; tol)
    end

    E = [double_pendulum_energy(params,x) for x in X]
    @assert length(X)==N
    @assert length(E)==N
    return X, E
end


function max_err_E(E)
    E0 = E[1]
    err = abs.(E .- E0)
    return maximum(err)
end
function get_explicit_energy_error(integrator::Function, dts::Vector)
    [max_err_E(simulate_explicit(params,double_pendulum_dynamics,integrator,x0,dt,tf)[2]) for dt in dts]
end
function get_implicit_energy_error(integrator::Function, dts::Vector)
    [max_err_E(simulate_implicit(params,double_pendulum_dynamics,integrator,x0,dt,tf)[2]) for dt in dts]
end


const tf = 2.0
let 
    # here we compare everything 
    dts = [1e-3,1e-2,1e-1]
    explicit_integrators = [forward_euler, midpoint, rk4]
    implicit_integrators = [backward_euler, implicit_midpoint, hermite_simpson]

    explicit_data = [get_explicit_energy_error(integrator, dts) for integrator in explicit_integrators]
    implicit_data = [get_implicit_energy_error(integrator, dts) for integrator in implicit_integrators]

    plot(dts, hcat(explicit_data...),label = ["Forward Euler" "Midpoint" "RK4"],xaxis=:log10,yaxis=:log10, xlabel = "Δt (s)", ylabel = "Energy Error")
    plot!(dts, hcat(implicit_data...),ls = :dash, label = ["Backward Euler" "Implicit Midpoint" "Hermite Simpson"])
    plot!(legend=:bottomright)
end

@testset "energy behavior" begin 
    
    # simulate with all integrators
    dt = 0.01
    t_vec = 0:dt:tf
    E1 = simulate_explicit(params,double_pendulum_dynamics,forward_euler,x0,dt,tf)[2]
    E2 = simulate_implicit(params,double_pendulum_dynamics,backward_euler,x0,dt,tf)[2]
    E3 = simulate_implicit(params,double_pendulum_dynamics,implicit_midpoint,x0,dt,tf)[2]
    E4 = simulate_implicit(params,double_pendulum_dynamics,hermite_simpson,x0,dt,tf)[2]
    E5 = simulate_explicit(params,double_pendulum_dynamics,midpoint,x0,dt,tf)[2]
    E6 = simulate_explicit(params,double_pendulum_dynamics,rk4,x0,dt,tf)[2]
    
    # plot forward/backward euler and implicit midpoint 
    plot(t_vec,E1, label = "Forward Euler (explicit)")
    plot!(t_vec,E2, label = "Backward Euler (implicit)")
    display(plot!(t_vec,E3, label = "Implicit Midpoint",xlabel = "Time (s)", ylabel="Energy (J)"))

    # test energy behavior
    E0 = E1[1]
    
    @test 2.5  < (E1[end] - E0) < 3.0
    @test -3.0 < (E2[end] - E0) < -2.5  
    @test abs(E3[end] - E0) < 1e-2
    @test abs(E0 - E4[end]) < 1e-4
    @test abs(E0 - E5[end]) < 1e-1 
    @test abs(E0 - E6[end]) < 1e-4
end