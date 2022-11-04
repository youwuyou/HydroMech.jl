push!(LOAD_PATH,"../src/")
using HydroMech

using Documenter

makedocs(
         sitename = "HydroMech.jl",
         modules  = [HydroMech],
         pages=[
                "Home" => "index.md",
                "Getting started" => 
                    [
                            "Overview" => "overview.md",
                            "Source code" => [ 
                                                "Modules" => "modules.md",
                                                "Types"   => "types.md"
                                             ],
                              
                    ],
                "Solvers" =>
                    [
                        "2D-Hydro-mechanical solver" => "2D-hydro-mechanical-solver.md",
                        "3D-Hydro-mechanical solver" => "3D-hydro-mechanical-solver.md"
                    ],
                "Concepts" => 
                    [
                    "Pseudo Transient Method" => "pseudo-transient-method.md"
                    ],
                        
                "Benchmarks" => "benchmarks.md",
                "Visualization" => "visualization.md",          
                "Troubleshooting" => "troubleshooting.md",
                
                "Reference" => 
                    [
                        "Computational earthquake physics" => "computational-earthquake-physics.md",
                        "PTsolvers/JustRelax.jl" => 
                                            [ "Workflow"    =>  "justrelax-workflow.md",
                                              "Source code" => [
                                                                    "Modules" => "justrelax-modules.md",
                                                                    "Types"   => "justrelax-types.md"
                                                                ],
                                              "Miniapps"    => "justrelax-Miniapps.md",
                                              "Testing"     => "justrelax-Testing.md"
                                            ],
                        "ParallelStencil" => "parallelstencil.md",
                        "JuliaGeodynamics" => "juliageodynamics.md"
                    ],
 
                "License" => "license.md"
                ]
        
       )
deploydocs(;
    repo="github.com/youwuyou/HydroMech.jl",
)
