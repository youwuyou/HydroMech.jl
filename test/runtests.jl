using HydroMech
using Test, ReferenceTests, BSON

# make sure to turn off GPU usage, at least for Github Actions
include("test_incompressible.jl")
include("test_compressible.jl")
