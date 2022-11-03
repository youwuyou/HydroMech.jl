push!(LOAD_PATH,"../src/")
using HydroMech

using Documenter

makedocs(
         sitename = "HydroMech.jl",
         modules  = [HydroMech],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/youwuyou/HydroMech.jl",
)
