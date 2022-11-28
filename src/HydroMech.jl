"""
Main module for HydroMech.jl
    
A collection of hydro-mechanical solvers for incompressible and compressible 2-phase flow.
"""
module HydroMech


#   Alphabetical include of submodules, except computation-submodules (below)
include("compressible/HydroMech2D.jl")
include("incompressible/HydroMech2D.jl")


# export types/functions that we desire to display on the HydroMech.jl API page
export HydroMech2D_compressible, HydroMech2D_incompressible


end # module HydroMech
