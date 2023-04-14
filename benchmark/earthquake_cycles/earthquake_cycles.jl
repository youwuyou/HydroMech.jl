using HydroMech

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)


# NOTE: despite of using the package we initialize here again because 
# we need to use the type Data.Array, Data.Number for argument types
const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
@static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
else
        @init_parallel_stencil(Threads, Float64, 2)
end


using Statistics, Printf, LinearAlgebra, SpecialFunctions
using Plots, Plots.Measures



const STORE_PRESSURE            = false   # set to true for stroing fluid pressure along fault
const INERTIA                   = true
const VISCOUS_ELASTO_PLASTICITY = true
const RATE_AND_STATE_FRICTION   = true


@static if STORE_PRESSURE
    using JLD
end



###################### PHYSICS #####################
#=
Physics contained in injection script

- Two phase flow + Stokes
- Viscous rheology
- compressibility

Added physics here:

[x] use the kernel with inertia in momentum equations

[x] add the elasticity + plasticity

[x] add the rate- and state-dependent friction to
the visco-elasto-plastic hydro-mechanical code

[x] add the new parameters needed for earthquake cycle simulation

[x] add the adaptive time stepping for earthquake cycles

[x] change the boundary conditions

[x] change visualization to plot slip rate against time (yr)


=#



###################### GPU KERNELS ######################

# Boundary conditions
# eg. pf = pt - peff
@inline @inbounds @parallel_indices (ix) function injection_constant_effective_pressure_x!(A::Data.Array, B::Data.Array, val::Data.Number)
    A[ix, 1]   = B[ix, 1] - val
    A[ix, end] = B[ix, end] - val
    return nothing
end


@inline @inbounds @parallel_indices (iy) function injection_dirichlet_x!(A::Data.Array, val_top::Data.Number, val_bottom::Data.Number)
    A[1, iy]   = val_top
    A[end, iy] = val_bottom 
    return nothing
end

@inline @inbounds @parallel_indices (ix) function injection_dirichlet_y!(A::Data.Array, val_left::Data.Number, val_right::Data.Number)
    A[ix, 1]   = val_left
    A[ix, end] = val_right
    return nothing
end


# FREE SLIP
@inline @inbounds @parallel_indices (iy) function injection_free_slip_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return nothing
end

@inline @inbounds @parallel_indices (ix) function injection_free_slip_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return nothing
end



# CONSTANT FLUX ∂V/∂x = C for some constant

# apply constant flux condition along x-axis
@inline @inbounds @parallel_indices (ix) function injection_constant_flux_y!(A::Data.Array, val_top::Data.Number, val_bottom::Data.Number)

    A[ix, 1]   = 2 * val_bottom - A[ix, 2]         # constant flux at top
    A[ix, end] = 2 * val_top - A[ix, end-1]  # constant flux at bottom

    return nothing
end





# Precomputation with constant porosity
@inbounds @parallel function injection_compute_poro_elasticity!(𝗞d::Data.Array, 𝗞ɸ::Data.Array, ɸ0::Data.Number, 𝝰::Data.Array, 𝝱d::Data.Array, B::Data.Array, _Ks::Data.Number, µ::Data.Number, ν::Data.Number, βs::Data.Number, βf::Data.Number)

    # i) Kd
    # Kɸ = 2m/(1+m)µ*/ɸ =  µ/(1-ν)/ɸ (m=1)
    @all(𝗞ɸ) = µ / (1.0 - ν) / ɸ0                                  # compute effective bulk modulus for the pores, µ shear modulus

    # Kd = (1-ɸ)(1/Kɸ + 1/Ks)⁻¹
    @all(𝗞d) = (1.0 - ɸ0) / (1.0 /@all(𝗞ɸ) + _Ks)                  # compute drained bulk modulus

    # ii). ɑ
    # 𝝱d = (1+ βs·Kɸ)/(Kɸ-Kɸ·ɸ) = (1+ βs·Kɸ)/Kɸ/(1-ɸ)
    @all(𝝱d) = (1.0 + βs * @all(𝗞ɸ)) / @all(𝗞ɸ) / (1.0 - ɸ0)       # compute solid skeleton compressibility
    @all(𝝰)  = 1.0 - βs / @all(𝝱d)                                       # compute Biot Willis coefficient

    # iii). B
    # B = (𝝱d - βs)/(𝝱d - βs + ɸ(βf - βs))
    @all(B) = (@all(𝝱d) - βs) / (@all(𝝱d) - βs + ɸ0 * (βf - βs))    # compute skempton coefficient


    return nothing
end


@inbounds @parallel function injection_compute_pt_steps!(Δτₚᶠ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, Pfᵣ::Data.Array, min_dxy2::Data.Number)
    @inn(Δτₚᶠ) = min_dxy2/4.1/@maxloc(𝐤ɸ_µᶠ)/@inn(Pfᵣ)
    return nothing
end



# without constant porosity
@inbounds @parallel function injection_assign!(∇V_o::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, ∇V::Data.Array,  Pt::Data.Array, Pf::Data.Array)
    @all(∇V_o)  = @all(∇V)

    # use the value from last physical iteration throughout PT iterations
    @all(Pt_o)  = @all(Pt)
    @all(Pf_o)  = @all(Pf)
    return nothing
end


# with inertia
@inbounds @parallel function injection_assign_inertia!(∇V_o::Data.Array, ∇V::Data.Array,  Pt_o::Data.Array, Pt::Data.Array, Pf_o::Data.Array, Pf::Data.Array, Vx_o::Data.Array, Vx::Data.Array, Vy_o::Data.Array, Vy::Data.Array, Vfx_o::Data.Array, Vfx::Data.Array, Vfy_o::Data.Array, Vfy::Data.Array)
    @all(∇V_o)  = @all(∇V)

    # use the value from last physical iteration throughout PT iterations
    @all(Pt_o)  = @all(Pt)
    @all(Pf_o)  = @all(Pf)

    # solid momentum
    @all(Vx_o)     = @all(Vx)
    @all(Vy_o)     = @all(Vy)

    # fluid momentum
    @all(Vfx_o)    = @all(Vfx)
    @all(Vfy_o)    = @all(Vfy)

    return nothing
end



@inbounds @parallel function injection_compute_∇!(∇V::Data.Array, ∇qD::Data.Array, Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, _dx::Data.Number, _dy::Data.Number)
    # compute gradient 2D
    @all(∇V)    = @d_xa(Vx)* _dx  + @d_ya(Vy)* _dy
    @all(∇qD)   = @d_xa(qDx)* _dx + @d_ya(qDy)* _dy

    return nothing
end



@inbounds @parallel function injection_compute_residual_mass_law!(fᴾᵗ::Data.Array, fᴾᶠ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, 𝞰ɸ::Data.Array, ɸ0::Data.Number, 𝗞d::Data.Array, 𝝰::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, 𝗕::Data.Array, dampPf::Data.Number, min_dxy2::Data.Number, _Δt::Data.Number)
    
    # residual f_pt for compressible solid mass
   #  + @all(𝝰) ... and + 1/@all(B) here to avoid subtraction operation due to performance
    @all(fᴾᵗ)  =  - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0- ɸ0)) -
                        1.0 /@all(𝗞d)*_Δt * (@all(Pt)- @all(Pt_o) + @all(𝝰)* (@all(Pf_o) - @all(Pf)))

    #  residual f_pf for compressible fluid mass
    @all(fᴾᶠ)  = @all(fᴾᶠ) * dampPf - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0- ɸ0)) + 
                       @all(𝝰)/@all(𝗞d)*_Δt * (@all(Pt) - @all(Pt_o) + 1.0/@all(𝗕)* (@all(Pf_o) - @all(Pf)))

    return nothing
end




################## NEW DAMPING - STOKES #######################

# compute residual for fluid and solid mass conservation eq
@inbounds @parallel function injection_compute_pressure_newdamping!(Pt::Data.Array, Pf::Data.Array, fᴾᵗ::Data.Array, fᴾᶠ::Data.Array, Δτₚᶠ::Data.Array, GΔτₚᵗ::Data.Number, r::Data.Number)

    # solid mass, total pressure update
    # ptⁿ = ptⁿ⁻¹ + Δτ_pt f_ptⁿ
    # @all(Pt)  = @all(Pt) + Δτₚᵗ * @all(fᴾᵗ)
    @all(Pt)  = @all(Pt) + r*GΔτₚᵗ*@all(fᴾᵗ)
 
    # fluid mass, fluid pressure update
    # pfⁿ = pfⁿ⁻¹ + Δτ_pf f_pfⁿ
    @all(Pf)  = @all(Pf) + @all(Δτₚᶠ)*@all(fᴾᶠ)
    
    return nothing
end


# viscous
@inbounds @parallel function injection_compute_tensor_newdamping!(σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, fᴾᵗ::Data.Array, GΔτₚᵗ::Data.Number, μˢ::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # viscous
    # ɛ̇xx =  @d_xa(Vx)* _dx
    # ɛ̇yy =  @d_ya(Vy)* _dy
    # ɛ̇xy = 0.5*(@d_yi(Vx)* _dy + @d_xi(Vy)* _dx)


    # General formula for viscous creep shear rheology
    # μˢ <-> solid shear viscosity
    @all(σxxʼ) = (@all(σxxʼ) + 2.0 * GΔτₚᵗ*@d_xa(Vx)* _dx) / (GΔτₚᵗ/μˢ + 1.0)
    @all(σyyʼ) = (@all(σyyʼ) + 2.0 * GΔτₚᵗ*@d_ya(Vy)* _dy) / (GΔτₚᵗ/μˢ + 1.0)
    @all(σxyʼ) = (@all(σxyʼ) + 2.0 * GΔτₚᵗ*(0.5*(@d_yi(Vx)* _dy + @d_xi(Vy)* _dx)))/(GΔτₚᵗ/μˢ + 1.0)

    return nothing
end


# FIXME: visco-elasto-plastic
@inbounds @parallel function stokesvep_compute_tensor_newdamping!(σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, σII::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, fᴾᵗ::Data.Array, Z::Data.Array, ηvp::Data.Array, µ::Data.Number, GΔτₚᵗ::Data.Number, _dx::Data.Number, _dy::Data.Number, Δt::Data.Number)


    # using the viscous-like vep reformulation as in Gerya's script

    # compute Z
    @all(Z)    = µ * Δt / (µ * Δt + @all(ηvp))

    # viscous-elasto-plastic
    # ɛ̇xx =  0.5* ( @d_xa(Vx)* _dx - @d_ya(Vy)* _dy)
    # ɛ̇yy =  0.5* ( @d_ya(Vy)* _dy - @d_xa(Vx)* _dx)
    # ɛ̇xy =  0.5* (@d_yi(Vx)* _dy + @d_xi(Vy)* _dx)

    # compute stress
    # σijʼ = 2ηvp·Z·ɛ̇ij + [1 - (1- ηvp/GΔτₚᵗ)Z]·σijʼ_old
    # @all(σxxʼ) = 2.0 * @all(ηvp) * @all(Z) * 0.5* ( @d_xa(Vx)* _dx - @d_ya(Vy)* _dy) + (1.0 - (1.0 - @all(ηvp) /GΔτₚᵗ) * @all(Z)) * @all(σxxʼ)
    # @all(σyyʼ) = 2.0 * @all(ηvp) * @all(Z) * 0.5* ( @d_ya(Vy)* _dy - @d_xa(Vx)* _dx) + (1.0 - (1.0 - @all(ηvp) /GΔτₚᵗ) * @all(Z)) * @all(σyyʼ)
    # @all(σxyʼ) = 2.0 * @all(ηvp) * @all(Z) * 0.5* ( @d_yi(Vx)* _dy + @d_xi(Vy)* _dx) + (1.0 - (1.0 - @all(ηvp) /GΔτₚᵗ) * @all(Z)) * @all(σxyʼ)

    # using the vep formulation as in gerya's script
    @all(σxxʼ) = 2.0 * @all(ηvp)* @all(Z) * 0.5* ( @d_xa(Vx)* _dx - @d_ya(Vy)* _dy) + (1.0 - @all(Z)) * @all(σxxʼ)
    @all(σyyʼ) = 2.0 * @all(ηvp)* @all(Z) * 0.5* ( @d_ya(Vy)* _dy - @d_xa(Vx)* _dx) + (1.0 - @all(Z)) * @all(σyyʼ)
    @all(σxyʼ) = 2.0 * @all(ηvp)* @all(Z) * 0.5* ( @d_yi(Vx)* _dy + @d_xi(Vy)* _dx) + (1.0 - @all(Z)) * @all(σxyʼ)


    # second stress invariant i) + ii) σII = √(1/2 σᵢⱼ'²) on staggered grid
    # FIXME: check if correct
    @all(σII)     = sqrt(0.5 * (@av_xa(σxxʼ)^2 + @av_ya(σyyʼ)^2) + @all(σxyʼ)^2)
    
    return nothing
end







# FIXME: compute only on the fault
@inbounds @parallel function rate_and_state_friction!(Vp, σII, Pt, Pf, a, b, Ω, F, Bool, L, σyield, ɛ̇II_plastic, ηvp::Data.Array, V0, γ0, Δt, σyieldmin, Wh, μˢ)
        
    # NOTE: Peff  = Pt - Pf    
    @all(Vp)          = 2.0 * V0 * sinh(max(@all(σII), 0.0)/@all(a)/(@inn(Pt) - @inn(Pf))) * exp(-(@all(b)*@all(Ω) + γ0)/@all(a))

    @all(F)           = @all(Vp) * Δt / @all(L)    # compute new value for slip parameter
    @all(Bool)        = @all(F) > 1.0e-6           # matrix contains boolean values {0, 1}  

    # first term is assigned when the Bool evalutes to 1
    @all(Ω)           = @all(Bool) * log(V0/@all(Vp) + (exp(@all(Ω)) - V0/@all(Vp))*exp(-@all(Vp)*Δt/@all(L))) + (1 - @all(Bool)) * log(exp(@all(Ω)) * (1.0 - @all(Vp)*Δt/@all(L)) + V0*Δt/@all(L))   

    @all(σyield)      = max(σyieldmin, (@all(Pt) - @all(Pf)) * @all(a) * asinh(@all(Vp)/2.0/V0*exp((@all(b) * @all(Ω) + γ0)/@all(a))) )
    @all(ɛ̇II_plastic) = @all(Vp)/2.0/Wh
    @inn(ηvp)         = μˢ* @all(σyield)/(2.0*μˢ*@all(ɛ̇II_plastic) + @all(σyield))

    return nothing
end


@inbounds @parallel function adaptive_timestepping!(ξ, Bool_Δt, Δθmax, Δtdyn, Vp, a, b, Pt, Pf, L, K)

    # ADAPTIVE TIME STEPPING!
    # Timestep criterion, Lapusta et al., 2000 Lapusta and Liu, 2009
    #  with Δtmin = γ Δx/cs with γ = 1/4, minimum grid size Δx 
    # ξ = 1/4 [ (K·L)/(a·Peff) - (b-a)/a]² - (K·L)/(a·Peff)
    @all(ξ) = 0.25*(K*@all(L)/@all(a)/(@all(Pt) - @all(Pf))-(@all(b)-@all(a))/@all(a))^2-K*@all(L)/@all(a)/(@all(Pt) - @all(Pf))

    @all(Bool_Δt) = @all(ξ) < 0


    # ξ < 0: Δθmax = min[1- ((b-a)p)/(K L), 0.2]
    # ξ ≥ 0: Δθmax = min[a p/(K L - (b-a) p), 0.2]
    @all(Δθmax)   = @all(Bool_Δt) * min(1.0-(@all(b)-@all(a))*(@all(Pt) - @all(Pf))/(K*@all(L)),0.2) + (1 - @all(Bool_Δt)) * min(@all(a)*(@all(Pt) - @all(Pf))/(K*@all(L)-(@all(b)-@all(a))*(@all(Pt) - @all(Pf))),0.2)


    @all(Δtdyn) = @all(Δθmax)*@all(L)/@all(Vp)

    return
end



# compute residual for stokes equation

## version 1: no inertia
@inbounds @parallel function injection_compute_residual_momentum_law_newdamping!(fᵛˣ::Data.Array, fᵛʸ::Data.Array, σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Pt::Data.Array, 𝞀g::Data.Array, _dx::Data.Number, _dy::Data.Number)

    # compute residual f_vᵢⁿ for total momentum
    # geological coordinates y-axis positive pointing downwards
    @all(fᵛˣ)    = (@d_xi(σxxʼ)- @d_xi(Pt))* _dx + @d_ya(σxyʼ)* _dy 
    @all(fᵛʸ)    = (@d_yi(σyyʼ)- @d_yi(Pt))* _dy + @d_xa(σxyʼ)* _dx + @av_yi(𝞀g)

    return nothing
end


## Version 2: with inertia
@inbounds @parallel function stokesvep_compute_residual_momentum_law_newdamping_inertia!(fᵛˣ::Data.Array, fᵛʸ::Data.Array, σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Pt::Data.Array, Vx::Data.Array, Vx_o::Data.Array, Vy::Data.Array, Vy_o::Data.Array, ρt::Data.Number, g::Data.Number, _dx::Data.Number, _dy::Data.Number, _Δt::Data.Number)

    # compute residual f_vᵢⁿ for total momentum
    # geological coordinates y-axis positive pointing downwards
    @all(fᵛˣ)    = (@d_xa(σxxʼ) - @d_xa(Pt)) * _dx + @d_ya(σxyʼ) * _dy - ρt * (@inn_x(Vx) - @inn_x(Vx_o)) * _Δt
    @all(fᵛʸ)    = (@d_ya(σyyʼ) - @d_ya(Pt)) * _dy + @d_xa(σxyʼ) * _dx + ρt*g - ρt * (@inn_y(Vy) - @inn_y(Vy_o)) * _Δt


    # FIXME: original formulation from stokes vep
    # @all(fᵛˣ)    = (@d_xa(σxxʼ) - @d_xa(Pt)) * _dx + @d_yi(σxyvʼ) * _dy - ρt * (@inn_x(Vx) - @inn_x(Vx_o)) * _Δt
    # @all(fᵛʸ)    = (@d_ya(σyyʼ) - @d_ya(Pt)) * _dy + @d_xi(σxyvʼ) * _dx + @av_ya(ρg) - ρt * (@inn_y(Vy) - @inn_y(Vy_o)) * _Δt


    return nothing
end




# i).without inertia
@inbounds @parallel function injection_compute_velocity_newdamping!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, fᵛˣ::Data.Array, fᵛʸ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, Pf::Data.Array, Δτᵥ_ρ::Data.Number, ρfg::Data.Number,  _dx::Data.Number, _dy::Data.Number)

    # i). total momentum, velocity update
    # vᵢⁿ = vᵢⁿ⁻¹ + Δτ_vᵢ/ρ f_vᵢⁿ for i in x,y
    @inn(Vx) = @inn(Vx) + Δτᵥ_ρ * @all(fᵛˣ)
    @inn(Vy) = @inn(Vy) + Δτᵥ_ρ * @all(fᵛʸ)

    
    # ii). fluid momentum, velocity update
    # qDᵢⁿ = - k^ɸ/ µ^f (∇Pf - ρf·g)
    # geological coords
    @inn(qDx) = -@av_xi(𝐤ɸ_µᶠ)*(@d_xi(Pf)* _dx)
    @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy - ρfg )
    
    return nothing
end




# ii). with inertia
@inbounds @parallel function stokesvep_compute_velocity_newdamping_inertia!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, Vfx::Data.Array, Vfy::Data.Array, Vfx_o::Data.Array, Vfy_o::Data.Array,  fᵛˣ::Data.Array, fᵛʸ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, Pf::Data.Array, Δτᵥ_ρ::Data.Number, ɸ0::Data.Number, ρf::Data.Number, g::Data.Number,  _dx::Data.Number, _dy::Data.Number, _Δt::Data.Number)
    
    # geological coords    
    # i). total momentum, velocity update
    # vᵢⁿ = vᵢⁿ⁻¹ + Δτ_vᵢ/ρ f_vᵢⁿ for i in x,y
    @inn(Vx) = @inn(Vx) + Δτᵥ_ρ * @all(fᵛˣ)
    @inn(Vy) = @inn(Vy) + Δτᵥ_ρ * @all(fᵛʸ)
    
    # ii). fluid momentum, velocity update
    # qDᵢⁿ = - k^ɸ/ µ^f (∇Pf - ρf·(g- Dvf/Dt))    
    # vf = 1/ɸ·qD + vs
    @inn(Vfx) = @inn(qDx) / ɸ0 +  @inn(Vx)
    @inn(Vfy) = @inn(qDy) / ɸ0 +  @inn(Vy)

    @inn(qDx) = -@av_xi(𝐤ɸ_µᶠ)*(@d_xi(Pf)* _dx + ρf * (@inn(Vfx) - @inn(Vfx_o)) * _Δt)             # no grav. acceleration along x
    @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy - ρf * (g - (@inn(Vfy) - @inn(Vfy_o)) * _Δt) )

    # @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy - ρfg + ρf* (@inn(Vfy) - @inn(Vfy_o)) * _Δt)


    return
end



##################### NUMERICAL ########################

"""The fluid injection model problem consists of a 2D square model domain Ω = [Lx, Ly]
where a 1D fault is embedded along y = Ly/2. We compare the result with the analytical solution
which consists of an error function.

    - no inertia
    - with compressibility

i). viscous rheology

ii).Porosity dependent viscosity
η_ϕ = η_c ⋅ ɸ0/ɸ (1+ 1/2(1/R − 1)(1+tanh(−Pₑ/λₚ)))
ηc = μs/C/φ0

iii). power law permeability
k_ɸ = k0 (ɸ/ɸ0)^nₖ = k0 (ɸ/ɸ0)^3

"""
@views function earthquake_cycles(;t_tot_)


    # MESH
    lx       = 100000.0  # [m]
    ly       = 20000.0  # [m]
    nx       = 1001
    ny       = 201


    dx, dy  = lx/(nx-1), ly/(ny-1)   # grid step in x, y
    mesh    = PTGrid((nx,ny), (lx,ly), (dx,dy))
    _dx, _dy      = inv.(mesh.di)
    max_nxy       = max(nx,ny)
    min_dxy2      = min(dx,dy)^2
        
    # index for accessing the corresponding row of the interface
    h_index = Int((ny - 1) / 2) + 1 # row index where the properties are stored for the fault


    # RHEOLOGY
    # porosity-dependent viscosity - for computing 𝞰ɸ    
    # in order to recover formulation in Dal Zilio (2022)
    C        = 1.0             # bulk/shear viscosity ratio
    R        = 1.0             # Compaction/decompaction strength ratio for bulk rheology

    # from table 1
    ɸ0       = 0.01            # reference porosity   -> 1#
    k0       = 1e-16           # reference permeability [m²]
    μˢ       = 1e23            # solid shear viscosity [Pa·s]
    µᶠ       = 1e-3            # fluid viscosity
    # default values
    # nₖ       = 3.0           # Carman-Kozeny exponent
    # λp       = 0.01          # effective pressure transition zone not used if R set to 1
    # θ_e      = 9e-1, 
    # θ_k      = 1e-1

    rheology = ViscousRheology(μˢ,µᶠ,C,R,k0,ɸ0)

    #====================#

    # TWO PHASE FLOW
    ρf       = 1.0e3                    # fluid density 1000 kg/m^3
    ρs       = 2.7e3                    # solid density 2700 kg/m^3
    ρt       = ρf*ɸ0 + ρs*(1.0-ɸ0)      # density total (background)    
    # forces
    g        = 9.81998                  # gravitational acceleration [m/s^2]
    ρfg      = ρf * g                   # force fluid
    ρsg      = ρs * g                   # force solid
    ρtg      = ρt * g                   # force total - note for total density ρt = ρf·ɸ + ρs·(1-ɸ)
    
    flow                  = TwoPhaseFlow2D(mesh, (ρfg, ρsg, ρtg))
    
    # Initial conditions
    ηɸ                    = μˢ/ɸ0
    𝞰ɸ                    = fill(μˢ/ɸ0, nx, ny)
    𝞀g                    = fill(ρtg, nx, ny)
    
    kɸ_fault              = 1e-15                         # domain with high permeability
    kɸ_domain             = 1e-22                         # domain with low permeability
    𝐤ɸ_µᶠ                 = fill(kɸ_domain/µᶠ, nx, ny)    # porosity-dependent permeability
    𝐤ɸ_µᶠ[:, h_index]    .= kɸ_fault/µᶠ                   # along fault


    pf                   = 10.0e6                         # [Pa] = 10MPa Pf at t = 0
    Pf                   = fill(pf, nx, ny)

    pt                   = 40.0e6                        #  [Pa] = 40MPa
    Pt                   = fill(pt, nx, ny)
    
    flow.𝞰ɸ              = PTArray(𝞰ɸ)
    flow.𝐤ɸ_µᶠ           = PTArray(𝐤ɸ_µᶠ)
    flow.Pf              = PTArray(Pf)
    flow.Pt              = PTArray(Pt)
    flow.𝞀g              = PTArray(𝞀g)     # initialize here because we don't have porosity update in current code
    #====================#
    
    # Fluid injection specific
    Peff                 = pt - pf          # constant effective pressure [Pa] -> 15MPa 

    # PHYSICS FOR COMPRESSIBILITY
    µ   = 25.0e+9       # shear modulus 25 GPa
    ν   = 0.25          # Poisson ratio
    Ks  = 50.0e+9       # bulk modulus  50 GPa
    _Ks = inv(Ks)
    βs  = 2.5e-11       # solid compressibility  # [1/Pa]
    βf  = 4.0e-10       # fluid compressibility  # [1/Pa]

    comp   = Compressibility(mesh, µ, ν, Ks, βs, βf)


    # precomputation of values for compressibility - compute only once since porosity is fixed as constant!
    # could have just used constant numbers here but using arrays for now in case varying porosity added
    @parallel injection_compute_poro_elasticity!(comp.𝗞d, comp.𝗞ɸ, ɸ0, comp.𝝰, comp.𝝱d, comp.𝗕, _Ks, comp.µ, comp.ν, comp.βs, comp.βf)
    

    # BOUNDARY CONDITIONS
    # define scalar values Vpl, p⁻, p⁺
    Vpl     = 1.9977e-9       # loading rate [m/s] = 6.3 cm/yr
    p⁻      = -1.0e-12        # BC top - outward flux [m/s]
    p⁺      =  1.0e-12        # BC bottom - inward flux [m/s]
    @parallel (1:nx+1) injection_dirichlet_y!(flow.V.x, -0.5*Vpl, 0.5*Vpl)



    # PHYSICS FOR INERTIA
    if INERTIA
        Vx_o      = @zeros(nx+1, ny  )
        Vy_o      = @zeros(nx  , ny+1)

        # fluid momentum
        Vfx       = @zeros(nx+1, ny  )
        Vfy       = @zeros(nx  , ny+1)
        Vfx_o     = @zeros(nx+1, ny  )
        Vfy_o     = @zeros(nx  , ny+1)
    end


    if VISCOUS_ELASTO_PLASTICITY

        # allocate Z, ηvp (same size as ∇V)
        Z        = @zeros(nx, ny)
        ηvp_cpu  = fill(μˢ, nx, ny)
        ηvp      = PTArray(ηvp_cpu)

        # same size as σxy
        σII      = @zeros(nx-1, ny-1)
    end

    if RATE_AND_STATE_FRICTION
        #            domain   fault
        a0        = [0.100    0.006]     # a-parameter of RSF
        b0        = [0.003    0.005]     # b-parameter of RSF
        Ω0        = [1e+10    1e-5]      # State variable from the preνous time step
        L0        = [0.010    0.001]     # L-parameter of RSF (characteristic slip distance)
        V0        = 1e-9                 # Reference slip velocity of RSF, m/s
        γ0        = 0.6                  # Ref. Static Friction
        σyieldmin = 1e3
        Wh        = dy                   # fault width

        # define Vp, a, b, F, Bool, Ω, σyield, ɛ̇II_plastic,  same size as SII
        Vp          = @zeros(nx-1, ny-1)
        F           = @zeros(nx-1, ny-1)
        σyield      = @zeros(nx-1, ny-1)
        ɛ̇II_plastic = @zeros(nx-1, ny-1)

        a_cpu       = fill(a0[1], nx-1, ny-1)
        b_cpu       = fill(b0[1], nx-1, ny-1)
        Ω_cpu       = fill(Ω0[1], nx-1, ny-1)
        L_cpu       = fill(L0[1], nx-1, ny-1)
        Bool_cpu    = fill(false, nx-1, ny-1)


        # assign along fault [:, h_index] for rate-strengthing/weanking regions
        @. a_cpu[401:601, h_index] = a0[2]
        @. b_cpu[401:601, h_index] = b0[2]
        @. Ω_cpu[401:601, h_index] = Ω0[2]
        @. L_cpu[401:601, h_index] = L0[2]

        a           = PTArray(a_cpu)
        b           = PTArray(b_cpu)
        Bool        = PTArray(Bool_cpu)
        Ω           = PTArray(Ω_cpu)
        L           = PTArray(L_cpu)



        # adaptive time stepping
        # bulk modulus
        B = 1/βs

        # FIXME: why having two defs of Poisson's ratio?
        ν_timestepping = (3*B-2*µ)/(6*B+2*µ)   # ν = (3B - 2µ)/(6B + 2µ)


        # stiffness K = 2/π µ*/Δx  | with shear modulus µ* = µ/(1-ν)
        #             = 2/π (µ/(1-ν))/Δx
        K_timestepping = 2/π*µ/(1-ν_timestepping)/dx

        # ξ, Bool_Δt, Δθmax same size as a
        ξ           = @zeros(nx-1, ny-1)        
        Δθmax       = @zeros(nx-1, ny-1)
        Δtdyn       = @zeros(nx-1, ny-1)
        Bool_Δt_cpu = fill(false, nx-1, ny-1)
        Bool_Δt     = PTArray(Bool_Δt_cpu)

    end


    # PT COEFFICIENT
    # scalar shear viscosity μˢ = 1.0 was used in porosity wave benchmark to construct dτPt    
    max_lxy   = max(lx, ly)
    min_lxy   = min(lx, ly)
    max_dxy   = max(dx, dy)
    max_dxy2  = max(dx,dy)^2
    CFL       = 0.9/sqrt(2)
    VpΔτ      = CFL*min(dx,dy)
    Re        = 5π
    r         = 1.0

    # stokes damping
    @show Δτᵥ_ρ = VpΔτ*max_lxy/Re/μˢ         # original formulation with ρ = Re·µ
    @show GΔτₚᵗ = VpΔτ^2/(r+2.0)/Δτᵥ_ρ/μˢ   # special case for fluid injection 


    # darcy damping
    dampPf        = 0.6
    ηb            = 1.0
    Δτₚᶠ_cpu      = zeros(nx, ny)
    Δτₚᶠ          = PTArray(Δτₚᶠ_cpu)

    # define different reduce factors for the PT time step
    Pfᵣ_domain    = 1.0e7
    # Pfᵣ_domain    = 1.0e4
    Pfᵣ_fault     = 40.0
    Pfᵣ_cpu       = fill(Pfᵣ_domain, nx, ny)

    # setting reduction of PT time step for fluid pressure along fault
    @. Pfᵣ_cpu[:, h_index]   = Pfᵣ_fault
    
    # @maxloc causes wrong time step size near fault, we need to set it to the same size as other places in the domain
    @. Pfᵣ_cpu[:, h_index-1] = 10.0       
    @. Pfᵣ_cpu[:, h_index+1] = 10.0
    Pfᵣ                      = PTArray(Pfᵣ_cpu)

    @parallel injection_compute_pt_steps!(Δτₚᶠ, flow.𝐤ɸ_µᶠ, Pfᵣ, min_dxy2)
    @parallel (1:ny) injection_free_slip_x!(Δτₚᶠ)  # make sure the Δτₚᶠ time steps are well-defined on boundaries
    @parallel (1:nx) injection_free_slip_y!(Δτₚᶠ)
    @show extrema(Δτₚᶠ)  # extrema(Δτₚᶠ) = (60975.60975609758, 6.097560975609758e11)

    # VISUALIZATION
    if DO_VIZ
        default(size=(1200,1000),fontfamily="Computer Modern", linewidth=3, framestyle=:box, margin=6mm)
        scalefontsizes(); scalefontsizes(1.35)

        ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")


        # calculate for 25 grid points spanning from x ∈ [0, 50]
        X_plot =  LinRange(0,50,nx)

        X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
        Xv          = (-dx/2):dx:(lx+dx/2)
    end
  

    # Time loop
    t_tot    = t_tot_           # total time
    # Δt       = 5.0            # physical time-step fluid injection
    # Δt       = 3.0e7          # physical time-step Luca
    Δt        = 1.0e7           # physical time-step 

    t         = 0.0
    it        = 1
    ε         = 1e-6            # tolerance
    # iterMax   = 1e6           # 5e3 for porosity wave, 5e5 previously
    iterMax   = 1e5             # 5e3 for porosity wave, 5e5 previously
    nout      = 200
    time_year = 365.25*24*3600

    # precomputation
    _Δt        = inv(Δt)
    length_Rx  = length(flow.R.Vx)
    length_Ry  = length(flow.R.Vy)
    length_RPf = length(flow.R.Pf)
    length_RPt = length(flow.R.Pt)
    _C         = inv(rheology.C)
    _ɸ0        = inv(rheology.ɸ0)
    _Ks        = inv(comp.Ks)


    # record evolution of time step size and slip rate
    evo_t = Float64[]; evo_Δt = Float64[]; evo_Vp = Float64[]

    while t<t_tot

        if INERTIA
            @parallel injection_assign_inertia!(flow.∇V_o, flow.∇V,  comp.Pt_o, flow.Pt, comp.Pf_o, flow.Pf, Vx_o, flow.V.x, Vy_o, flow.V.y, Vfx_o, Vfx, Vfy_o, Vfy)
        else
            @parallel injection_assign!(flow.∇V_o, comp.Pt_o, comp.Pf_o, flow.∇V, flow.Pt, flow.Pf)
        end  
     
        err=2*ε; iter=1; niter=0
        
        while err > ε && iter <= iterMax
            if (iter==11)  global wtime0 = Base.time()  end
    
            @parallel injection_compute_∇!(flow.∇V, flow.∇qD, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, _dx, _dy)            
            @parallel injection_compute_residual_mass_law!(flow.R.Pt, flow.R.Pf, flow.𝐤ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, ɸ0, comp.𝗞d, comp.𝝰, comp.Pt_o, comp.Pf_o, comp.𝗕, dampPf, min_dxy2, _Δt)
            
            @parallel injection_compute_pressure_newdamping!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, Δτₚᶠ, GΔτₚᵗ, r)
            

            if VISCOUS_ELASTO_PLASTICITY
                @parallel stokesvep_compute_tensor_newdamping!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy,  σII, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, Z, ηvp, µ, GΔτₚᵗ, _dx, _dy, Δt)    
            else
                # compute stress tensor using viscous law
                @parallel injection_compute_tensor_newdamping!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, GΔτₚᵗ, μˢ, _dx, _dy)
            end
    
            if INERTIA
                @parallel stokesvep_compute_residual_momentum_law_newdamping_inertia!(flow.R.Vx, flow.R.Vy, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.V.x, Vx_o, flow.V.y, Vy_o, ρt, g, _dx, _dy, _Δt)
                @parallel stokesvep_compute_velocity_newdamping_inertia!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, Vfx, Vfy, Vfx_o, Vfy_o,  flow.R.Vx, flow.R.Vy, flow.𝐤ɸ_µᶠ, flow.Pf, Δτᵥ_ρ, ɸ0, ρf, g,  _dx, _dy, _Δt)
            else
                @parallel injection_compute_residual_momentum_law_newdamping!(flow.R.Vx, flow.R.Vy, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, _dx, _dy)
                @parallel injection_compute_velocity_newdamping!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, flow.R.Vx, flow.R.Vy, flow.𝐤ɸ_µᶠ, flow.Pf, Δτᵥ_ρ, ρfg,  _dx, _dy)                
            end
            
            
            # BOUNDARY CONDITIONS
            # FIXME: the x-, y- coords here correspond to the coord before flipping
            @parallel (1:nx+1) injection_dirichlet_y!(flow.V.x, -0.5*Vpl, 0.5*Vpl)
            @parallel (1:nx)   injection_dirichlet_y!(flow.V.y, 0.0, 0.0)
            @parallel (1:ny)   injection_free_slip_x!(flow.V.x)
            @parallel (1:ny+1) injection_free_slip_x!(flow.V.y)

            # FIXME: why no bc for qDx?
            @parallel (1:ny+1) injection_free_slip_x!(flow.qD.y)
            @parallel (1:nx)   injection_constant_flux_y!(flow.qD.y, p⁻, p⁺)
            @parallel (1:nx)   injection_constant_effective_pressure_x!(flow.Pf, flow.Pt, Peff) # confining pressure to boundaries parallel to x-axis

            #                  ny
            #          ---------x---------
            #          |        |        |
            #          |        |        |
            #          |        |        |
            #          |        |        |
            #    nx    |        |        |
            #          |        |        |
            #          |        |        |
            #          |        |        |
            #          |        |        |
            #          ------------------
    
            # NOTE: when we plot we plot the transpose of the matrix and obtain the following
            #           
            #         ----------------------------------
            #         |                                | ny
            #         |                                |
            #         x--------------------------------|
            #         |                                |
            #         |                                |
            #         |---------------------------------
            #                    nx
        
            if mod(iter, nout)==0
                global norm_Rx, norm_Ry, norm_RPf, norm_RPt
                norm_Rx  = norm(flow.R.Vx)/length_Rx
                norm_Ry  = norm(flow.R.Vy)/length_Ry
                norm_RPf = norm(flow.R.Pf)/length_RPf
                norm_RPt = norm(flow.R.Pt)/length_RPt
                err = max(norm_Rx, norm_Ry, norm_RPf, norm_RPt)
                
                if mod(iter,nout*100) == 0
                    @printf("iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_RPf=%1.3e, norm_RPt=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_RPf, norm_RPt)
                end
    
            end
    
    
            iter+=1; niter+=1
        end


        if RATE_AND_STATE_FRICTION
            # @parallel rate_and_state_friction!(Vp, σII, flow.Pt, flow.Pf, a, b, Ω, F, Bool, L, σyield, ɛ̇II_plastic, ηvp, V0, γ0, Δt, σyieldmin, Wh, μˢ)
            @parallel rate_and_state_friction!(Vp[:, h_index], σII[:, h_index], flow.Pt[:, h_index], flow.Pf[:, h_index], a[:, h_index], b[:, h_index], Ω[:, h_index], F[:, h_index], Bool[:, h_index], L[:, h_index], σyield[:, h_index], ɛ̇II_plastic[:, h_index], ηvp[:, h_index], V0, γ0, Δt, σyieldmin, Wh, μˢ)
            @parallel adaptive_timestepping!(ξ, Bool_Δt, Δθmax, Δtdyn, Vp, a, b, flow.Pt, flow.Pf, L, K_timestepping)            
            # Δt = max[Δtmin, Δtdyn]
            #                        Δtdyn = Δθmax L/Vmax
            @show Δt = max(Δt, minimum(Δtdyn))

            # @show Δt = min(Δt, minimum(Δtdyn))

        end


    
        # PERFORMANCE
        wtime    = Base.time() - wtime0
        A_eff    = (8*2)/1e9*nx*ny*sizeof(eltype(flow.Pt))   # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(niter-10)                         # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                           # Effective memory throughput [GB/s]
        @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))
      


        # DEBUG
        @show extrema(Vp)


        # store evolution of physical properties wrt time
        @show max_Vp = maximum(Vp[:, h_index])
        push!(evo_t, t/time_year); push!(evo_Vp, max_Vp); push!(evo_Δt, Δt)

        # store fluid pressure for wanted time points
        if STORE_PRESSURE && mod(it, 1) == 0
            save("earthquake_cycles/Pf_fault" * string(it) * ".jld", "data", Array(flow.Pf[:, h_index])')   # store the fluid pressure along the fault for fluid injection benchmark
        end
    
        
        # Debug
        @show t   = t + Δt
        it += 1

        
        # Visualisation
        if DO_VIZ
            
            if mod(it,1) == 0
                p1 = plot(evo_t, evo_Vp; xlims=(0.0, 300), ylims=(1e-20, 2.0), yaxis=:log, label="", color= :dodgerblue, framestyle= :box, linestyle= :solid, 
                         seriesstyle= :path, title="", 
                            xlabel = "Time [year]", ylabel="Slip Rate [m/s]")

                p2 = plot(evo_t, evo_Δt; xlims=(0.0, 300), ylims=(1e-2, 1e6), yaxis=:log, label="", framestyle= :box, linestyle= :solid, 
                    seriesstyle= :path, title="", 
                    xlabel = "Time [year]", ylabel="Time step size [s]")
   
                display(plot(p1, p2; layout=(2,1))); frame(anim)
            end            

        end
    end
    
    # Visualization
    if DO_VIZ
        gif(anim, "earthquake_cycles.gif", fps = 15)
    end

    # return effective pressure at final time
    return Array(flow.Pt - flow.Pf)[:, h_index-1:h_index+1]'

end


if isinteractive()
    earthquake_cycles(;t_tot_= 1.0e7) # for reproducing fluid injection benchmark
end
