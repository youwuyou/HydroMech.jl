# Testing part 1 - 2D
using Test, JLD, StatsBase

include("../scripts/incompressible/HydroMech2D.jl")

# incompressible solver
@testset "Reference test: HydroMech2D_incompressible" begin
    # Testing information
    printstyled("Time stepping: t_tot = 0.0005, dt = 1e-5\n"; bold=true, color=:white)
    
    Peff_ref = load("2D/incom_Peff.jld")["data"]
    Peff     = HydroMech2D_incompressible(;t_tot_=0.0005)

    I = sample(1:length(Peff_ref), 5, replace=false)

    @testset "randomly chosen entries $i" for i in I
        @test Peff_ref[i] ≈ Peff[i]
    end
end;


# include("../scripts/compressible/HydroMech2D.jl")

# compressible solver
# solving incompressible case by setting parameter 
# TODO: check which param is the switch
# @testset "Reference test: HydroMech2D_compressible" begin
#     # Testing information
#     printstyled("Compressible solver solving incompressible case \n
#                 Time stepping: t_tot = 0.0005, dt = 1e-5\n"; bold=true, color=:white)
    
#     Peff_ref = load("2D/com_Peff.jld")["data"]
#     Peff     = HydroMech2D_compressible(;t_tot_=0.0005)

#     I = sample(1:length(Peff_ref), 5, replace=false)

#     @testset "randomly chosen entries $i" for i in I
#         @test Peff_ref[i] ≈ Peff[i]
#     end
# end;