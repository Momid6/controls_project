import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra

function is_controllable(A::Matrix, B::Matrix)
    nx, nu = size(B)
    U = B
    for k=1:nx-1
        U = hcat(U, (A^k)*B)
    end
    return rank(U) == nx
end