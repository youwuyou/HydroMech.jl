"""
Main module for HydroMech.jl
    
A collection of hydro-mechanical solvers for incompressible and compressible 2-phase flow.
"""
module HydroMech

# define constant for redirection of the files
# const PROJECT_ROOT = pkgdir(HydroMech)

using Reexport
@reexport using ParallelStencil
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA
using MPI

include("MetaHydroMech.jl")

export PS_Setup, environment!, ps_reset!


# Alphabetical include of submodules, except computation-submodules (below)
include("../scripts/compressible/HydroMech2D.jl")
include("../scripts/incompressible/HydroMech2D.jl")

# export types/functions that we desire to display on the HydroMech.jl API page
export HydroMech2D_compressible, HydroMech2D_incompressible


end # module HydroMech
