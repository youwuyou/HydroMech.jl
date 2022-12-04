using ReferenceTests

# make sure to turn off GPU usage, at least for Github Actions
include("test_2D.jl")
include("test_3D.jl")
