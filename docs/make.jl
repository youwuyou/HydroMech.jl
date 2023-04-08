push!(LOAD_PATH,"../src/")
using HydroMech, ParallelStencil, GeoParams

using Documenter

# 'modules' argument contains also other modules for displaying their API in the website
makedocs(
         sitename = "HydroMech.jl",
         modules  = [HydroMech, ParallelStencil, GeoParams],
         doctest  = false,
         pages=[
                "Home" => "index.md",
                "Getting started" => 
                    [
                            "Overview" => "overview.md",
                            # "Methodology" => "methodology.md",
                            "Tutorial" => "tutorial.md"
                    ],
                "Solvers" =>
                    [
                        "2D-Hydro-mechanical solver" => "2D-hydro-mechanical-solver.md",
                        "3D-Hydro-mechanical solver" => "3D-hydro-mechanical-solver.md"
                    ],
                "Concepts" => 
                    [
                        "Pseudo Transient Method" => "pseudo-transient-method.md",
                        "Iteration Parameters" => "iteration-parameters.md",
                        "Computational Earthquake Physics" => "computational-earthquake-physics.md"
                        # "Stiffness of PDEs" => "stiffness-of-pdes.md",
                        # "Eigenvalue Problem" => "eigenvalue-problem.md",
                        # "Dispersion Analysis" => "dispersion-analysis.md",
                        # "Von Neumann Stability Analysis" => "von-neumann-stability-analysis.md",
                    ],

                "Performance" => "performance.md",
                "Benchmarks" => 
                    [
                        "Porosity Wave" => "benchmark-porosity-wave.md",
                        "Fluid injection" => "benchmark-fluid-injection.md"
                    ],
                "Visualization" => "visualization.md",          
                "Development" =>
                    [
                        "Roadmap" => "roadmap.md",
                        "Time line" => "timeline.md",
                        "Troubleshooting" => "troubleshooting.md"

                    ],                
                "Reference" => 
                    [
                        "HydroMech.jl" => "hydromech.md",

                        "PTsolvers/JustRelax.jl" => 
                                            [ "Overview"    =>  "justrelax-overview.md",
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
