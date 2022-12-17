"""
Main module for HydroMech.jl
    
A collection of hydro-mechanical solvers for incompressible and compressible 2-phase flow.
"""
module HydroMech

# define constant for redirection of the files
# const PROJECT_ROOT = pkgdir(HydroMech)
const DO_VIZ    = true
const SAVE_TEST = false

export DO_VIZ, SAVE_TEST


using Reexport
@reexport using ParallelStencil
# @reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA
using MPI

# using an intermediate script to include methods dependent on ParallelStencil.jl
include("MetaHydroMech.jl")
export PS_Setup, environment!, ps_reset!


# export types/functions that we desire to display on the HydroMech.jl API page
include("API.jl")
export CompressibleTPF, IncompressibleTPF


end # module HydroMech
