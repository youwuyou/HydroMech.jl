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
    μˢ::T    # solid shear viscosity
    µᶠ::T

    # porosity-dependent viscosity - for computing 𝞰ɸ
    # η_ϕ = η_c ⋅ ɸ0/ɸ (1+ 1/2(1/R − 1)(1+tanh(−Pₑ/λₚ)))
    # ηc = μs/C/φ0
    C::T     # bulk/shear viscosity ratio
    R::T     # Compaction/decompaction strength ratio for bulk rheology
    λp::T    # effective pressure transition zone

    # Carman-Kozeny relationship for permeability
    # for computing 𝐤ɸ_µᶠ   
    #       k_ɸ = k0 (ɸ/ɸ0)^nₖ    
    k0::T    # reference permeability
    ɸ0::T
    nₖ::T    # Carman-Kozeny exponent    

    # relaxation factors for nonlinear terms
    # for computing 𝐤ɸ_µᶠ
    θ_e::T   # relaxation factor for non-linear viscosity
    θ_k::T   # relaxation factor for non-linear permeability

    function ViscousRheology(μˢ::T,
                             µᶠ::T,
                             C::T,
                             R::T,
                             k0::T,
                             ɸ0::T;
                             nₖ  = 3.0,
                             λp  = 0.01,  # not used if R set to 1
                             θ_e = 9e-1, 
                             θ_k = 1e-1) where {T}

        return new{T}(μˢ,µᶠ,C,R,λp,k0,ɸ0,nₖ,θ_e,θ_k)
    end

end
