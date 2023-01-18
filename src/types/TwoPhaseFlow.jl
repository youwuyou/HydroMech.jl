"""  TwoPhaseFlowTypes.jl
contains types specifically required for the two-phase flow equations,
we organized the equations in a way that reflects its mathematical formulations.
The residuals used for PT solve! computation is also contained within the TwoPhaseFlow2D struct.
"""

abstract type AbstractResidual{nVar} end


function make_twophase_residual_struct!(ndim; name::Symbol=:TwoPhaseResidual)
    dims = (:Pt, :Pf, :Vx, :Vy, :Vz)
    fields = [:($(dims[i])::T) for i in 1:ndim+2]
    @eval begin
        struct $(name){T} <: AbstractResidual{4}
            $(fields...)

            function $(name)(ni::NTuple{4,T}) where {T}
                return new{$PTArray}(@zeros(ni[1]...), 
                                     @zeros(ni[2]...),
                                     @zeros(ni[3]...),
                                     @zeros(ni[4]...)
                                     )
            end


            # TODO: add the term for 3D

        end
    end
end








"""
AbstractFlow{NDIMS, NVARS}
An abstract supertype of specific equations such as the compressible Euler equations.
The type parameters encode the number of spatial dimensions (`nDim`) and the
number of primary variables (`nVar`) of the physics model.

NOTE: formulation borrowed from Trixi.jl, changed from AbstractEquations
"""
abstract type AbstractFlow{nDim, nVar} end
abstract type AbstractTwoPhaseFlow{nDim, nVar} <:  AbstractFlow{nDim, nVar} end



function make_twophase_struct!()

    @eval begin
        mutable struct TwoPhaseFlow2D{T} <: AbstractTwoPhaseFlow{2,6}  # nDim, nVar
            # six unknowns
            Pf::PTArray
            Pt::PTArray
            V::PTVector
            qD::PTVector
            
            # used for computing residual
            𝞃::PTSymmetricTensor

            𝝫::PTArray
            𝞰ɸ::PTArray
            𝗞ɸ_µᶠ::PTArray    # k^ϕ/μᶠ

            𝞀g::PTArray
        
            # for computing 𝞀g and qD.y
            ρfg::T
            ρsg::T 
            ρgBG::T   # for computing Rhog and qDy


            # divergence field and old arrays used in update routine
            ∇qD::PTArray
            ∇V::PTArray
            ∇V_o::PTArray 
            𝝫_o::PTArray

            # residuals
            R::TwoPhaseResidual
            
            # constructor
            function TwoPhaseFlow2D(mesh::PTMesh, ρg::NTuple{3,T}) where {T}
                ni  = mesh.ni  # this is used for later

                Pf  = @zeros(ni...)
                Pt  = @zeros(ni...)
        
                V   = PTVector(((ni[1] + 1, ni[2]), (ni[1], ni[2] + 1)))
                qD  = PTVector(((ni[1] + 1, ni[2]), (ni[1], ni[2] + 1)))
        
                𝞃   = PTSymmetricTensor(((ni[1], ni[2]), (ni[1]-1, ni[2]-1), (ni[1], ni[2])))

                𝝫   = @zeros(ni...)
                𝞰ɸ   = @zeros(ni...)
                𝗞ɸ_µᶠ= @zeros(ni...)    # k^ϕ/μᶠ
                
                𝞀g  = @zeros(ni...)
        
                ρfg  = ρg[1]
                ρsg  = ρg[2] 
                ρgBG = ρg[3]

                ∇qD  = @zeros(ni...)
                ∇V   = @zeros(ni...)
                ∇V_o = @zeros(ni...)
                𝝫_o  = @zeros(ni...)

                R    = TwoPhaseResidual(((ni[1], ni[2]), (ni[1],ni[2]), (ni[1]-1,ni[2]-2), (ni[1]-2, ni[2]-1) ))

        
                return new{T}(
                    Pf,
                    Pt,
                    V,
                    qD,
                    𝞃,
                    𝝫,
                    𝞰ɸ,
                    𝗞ɸ_µᶠ,    # k^ϕ/μᶠ
                    𝞀g,
                    ρfg, 
                    ρsg, 
                    ρgBG,
                    ∇qD,
                    ∇V,
                    ∇V_o, 
                    𝝫_o,
                    R
                )
            end    # end of the constructor
        end  # end of the struct

    end # end of the eval


end



# Compressibility
function make_compressibility_struct!()
    
    @eval begin

        mutable struct Compressibility{T}
            Pt_o::PTArray
            Pf_o::PTArray
            𝗞d::PTArray
            𝗞ɸ::PTArray
            𝝰::PTArray
            𝝱d::PTArray
            𝗕::PTArray

            µ::T 
            Ks::T 
            βs::T 
            βf::T

            function Compressibility(mesh::PTMesh, µ::T, Ks::T, βs::T, βf::T) where {T}
                nx, ny   = mesh.ni  # this is used for later

                Pt_o     = @zeros(nx, ny)
                Pf_o     = @zeros(nx, ny)

                𝗞d       = @zeros(nx, ny)
                𝗞ɸ       = @zeros(nx, ny)
                𝝰        = @zeros(nx, ny)
                𝝱d       = @zeros(nx, ny)
                𝗕        = @zeros(nx, ny)

                return new{T}(    
                Pt_o, 
                Pf_o,
                𝗞d, 
                𝗞ɸ, 
                𝝰, 
                𝝱d, 
                𝗕,
                µ,
                Ks, 
                βs, 
                βf 
                )

            end # end of constructor

        end # end of struct

    end # end of the eval

end




# this needs to be exported! Used for type dispatch
abstract type OriginalDamping end


function make_pt_struct!()

    @eval begin 
        struct PTCoeff{T}
            dτPf::PTArray
            dτPt::T
            ## PT Pseudo-terms
            dτV::T
            dVxdτ::PTArray
            dVydτ::PTArray
            
            ## PT damping coefficients
            βₚₜ::T
            dampX::T
            dampY::T
            Pfdmp::T
            Pfsc::T
            Ptsc::T

            function PTCoeff(model::Type{OriginalDamping},
                            mesh::PTMesh, 
                            μˢ::T,
                            Vsc::T,
                            βₚₜ::T,
                            dampX::T,
                            dampY::T,
                            Pfdmp::T,
                            Pfsc::T,
                            Ptsc::T
                            ) where {T}

                nx, ny   = mesh.ni  # used for computing dτV
                dx, dy   = mesh.di

                dτV      = min(dx,dy)^2/μˢ/(1.0+βₚₜ)/4.1/Vsc     # PT time step for velocity

                dτPf     = @zeros(nx, ny)
                dτPt     = 4.1*μˢ*(1.0+βₚₜ)/max(nx,ny)/Ptsc
                dVxdτ    = @zeros(nx-1,ny-2)
                dVydτ    = @zeros(nx-2,ny-1)
                        
                return new{T}(dτPf,
                              dτPt,
                              dτV,
                              dVxdτ,
                              dVydτ,
                              βₚₜ,
                              dampX,
                              dampY,
                              Pfdmp,
                              Pfsc,
                              Ptsc)
            end

        
        end # end of PT struct
    end # end of eval

end # end of function