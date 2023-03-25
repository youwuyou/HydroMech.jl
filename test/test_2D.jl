# Testing part 1 - 2D
using Test, JLD, StatsBase

include("../benchmark/porosity_wave/PorosityWave2D.jl")

# incompressible solver
@testset "Reference test: PorosityWave2D_incompressible" begin
    # Testing information
    printstyled("Time stepping: t_tot = 0.0005, Δt = 1e-5\n"; bold=true, color=:white)
    
    Peff_ref = load("2D/incom_Peff.jld")["data"]
    Peff     = PorosityWave2D(;t_tot_=0.0005)

    I = sample(1:length(Peff_ref), 5, replace=false)

    @testset "randomly chosen entries $i" for i in I
        @test Peff_ref[i] ≈ Peff[i]
    end
end;

