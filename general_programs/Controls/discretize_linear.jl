import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


function discretize(Ac, Bc, dt)::Tuple{Matrix,Matrix}
    nx, nu = size(Bc)
    M = zeros(nx + nu, nx + nu)
    M[1:nx, 1:nx] .= Ac
    M[1:nx,nx+1:end] .= Bc
    Md = exp(dt * M)
    
    A = Md[1:nx, 1:nx]
    B = Md[1:nx,nx+1:end] 
    
    @assert size(A) == (nx,nx)
    @assert size(B) == (nx,nu)
    return A, B 
end
