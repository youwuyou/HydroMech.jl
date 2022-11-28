# Testing

The implemented solvers of `JustRelax.jl` can be easily used given the following environment setup.


```julia
# Example script for running the 2D Stokes solvers
using JustRelax
using Printf, LinearAlgebra, CairoMakie

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# choose benchmark
benchmark = :solcx

# model resolution (number of gridpoints)
nx, ny = 128, 128

# :single for a single run model with nx, ny resolution
# :multiple for grid sensitivy error plot
runtype = :single

if benchmark == :solcx
    # include plotting and error related functions
    include("solcx/SolCx.jl") # need to call this again if we switch from gpu <-> cpu
    
    # viscosity contrast
    ∆η = 1e6
    
    if runtype == :single
        # run model
        geometry, stokes, iters, ρ = solCx(∆η; nx=nx, ny=ny)
    
        # plot model output and error
        f = plot_solCx_error(geometry, stokes, ∆η; cmap=:romaO)
    elseif runtype == :multiple
        f = multiple_solCx(; ∆η=∆η, nrange=6:10) # nx = ny = 2^(nrange)-1
    end
elseif ...
(...)
end
```