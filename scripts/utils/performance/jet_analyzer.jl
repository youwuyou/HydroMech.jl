using Suppressor   # for I/O redirection
using JET          # util script for analyzing program using JET

# file to be analyzed
include("../incompressible/HydroMech2D.jl")

# @report_opt HydroMech2D_incompressible(;t_tot_=0.02)
output = @capture_out print(@report_opt HydroMech2D_incompressible(;t_tot_=0.02))


# file I/O
open("analyzer.txt", "w") do io
    write(io, output)
end