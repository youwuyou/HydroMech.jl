"""Rheology.jl
contains types specifically required for describing the physics. And
object of type that belongs to the AbstractRheology struct contains
the following information

    i). rheology (viscous, visco-elastic, visco-plasto-elastic)
    ii). creep law (describing viscosity in dependence of other parameters)
    iii). other empirical laws
"""


abstract type AbstractEmpiricalLaw{T} end

abstract type AbstractConstitutiveLaw{T} <: AbstractEmpiricalLaw{T} end

abstract type AbstractRheology{T} end



struct ViscousRheology{T} <: AbstractRheology{T}

    #====viscous constitutive law====#
    μˢ::T
    µᶠ::T

    # porosity-dependent viscosity - for computing 𝞰ɸ
    # η_ϕ = η_c ⋅ ɸ0/ɸ (1+ 1/2(1/R − 1)(1+tanh(−Pₑ/λₚ)))
    # ηc = μs/C/φ0
    C::T
    R::T
    λp::T        

    # Carman-Kozeny relationship for permeability
    # for computing 𝗞ɸ_µᶠ   
    #       k_ɸ = k0 (ɸ/ɸ0)^nₖ    
    k0::T
    ɸ0::T
    nₖ::T

    # relaxation factors for nonlinear terms
    # for computing 𝗞ɸ_µᶠ
    θ_e::T
    θ_k::T

end
