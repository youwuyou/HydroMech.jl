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



const STORE_DATA                = true   # set to true for stroing fluid pressure along fault
const INERTIA                   = true
const VISCOUS_ELASTO_PLASTICITY = true
const RATE_AND_STATE_FRICTION   = true


# visu switch
const PLOTTING_DOMAIN           = false
const PLOTTING_CYCLES           = true


@static if STORE_DATA
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



# without constant porosity
@inbounds @parallel function injection_assign!(∇V_o::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, ∇V::Data.Array,  Pt::Data.Array, Pf::Data.Array)
    @all(∇V_o)  = @all(∇V)

    # use the value from last physical iteration throughout PT iterations
    @all(Pt_o)  = @all(Pt)
    @all(Pf_o)  = @all(Pf)
    return nothing
end


# with inertia
@inbounds @parallel function injection_assign_inertia!(∇V_o::Data.Array, ∇V::Data.Array,  Pt_o::Data.Array, Pt::Data.Array, Pf_o::Data.Array, Pf::Data.Array, Vx_o::Data.Array, Vx::Data.Array, Vy_o::Data.Array, Vy::Data.Array, Vfx_o::Data.Array, Vfx::Data.Array, Vfy_o::Data.Array, Vfy::Data.Array, σxxʼ_o::Data.Array, σxxʼ::Data.Array, σxyʼ_o::Data.Array, σxyʼ::Data.Array, σyyʼ_o::Data.Array, σyyʼ::Data.Array)
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

    # rheology
    @all(σxxʼ_o)   = @all(σxxʼ)
    @all(σxyʼ_o)   = @all(σxyʼ)
    @all(σyyʼ_o)   = @all(σyyʼ)

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


# visco-elasto-plastic
@inbounds @parallel function compute_ve_stress!(ɛ̇xy, Vx, Vy, σxxʼ, σxxʼ_o, σyyʼ, σyyʼ_o, σxyʼ, σxyʼ_o, η_ve_τ, η_e, GΔτₚᵗ, _dx, _dy)

    # visco-elastic pseudo-transient strain rate
    @inn(ɛ̇xy) = 0.5*(@d_yi(Vx)* _dy + @d_xi(Vy)* _dx)

    # visco-elastic stress update containing PT terms
    @all(σxxʼ) = 2.0*η_ve_τ*(@d_xa(Vx)* _dx + 0.5*@all(σxxʼ_o)/η_e + 0.5*@all(σxxʼ)/GΔτₚᵗ)
    @all(σyyʼ) = 2.0*η_ve_τ*(@d_ya(Vy)* _dy + 0.5*@all(σyyʼ_o)/η_e + 0.5*@all(σyyʼ)/GΔτₚᵗ)
    @all(σxyʼ) = 2.0*η_ve_τ*(@av(ɛ̇xy) + 0.5*@all(σxyʼ_o)/η_e + 0.5*@all(σxyʼ)/GΔτₚᵗ)

    return nothing
end


@inbounds @parallel function compute_second_invariant!(σII::Data.Array, σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array)
    # second stress invariant i) + ii) σII = √(1/2 σᵢⱼ'²) on staggered grid
    @all(σII)     = sqrt(0.5 * (@av_xa(σxxʼ)^2 + @av_ya(σyyʼ)^2) + @all(σxyʼ)^2)
    return
end

# compute only on the fault
@inbounds @parallel function compute_plastic_correction!(λ, σII, σyield, σxxʼ, σyyʼ, σxyʼ, η_ve_τ::Data.Number, η_reg::Data.Number)


    # FIXME: checking if lambda is correct here
    @all(λ)    = max(@all(σII)-@all(σyield), 0.0)/(η_ve_τ + η_reg)

    @inn(σxxʼ) = @inn(σxxʼ) - 2.0*η_ve_τ*(@all(λ)*0.5*@inn(σxxʼ)/@inn(σII))
    @inn(σyyʼ) = @inn(σyyʼ) - 2.0*η_ve_τ*(@all(λ)*0.5*@inn(σyyʼ)/@inn(σII))
    @all(σxyʼ) = @all(σxyʼ) - 2.0*η_ve_τ*(@all(λ)*0.5*@all(σxyʼ)/@all(σII))

    return nothing

end


# compute only on the fault
@inbounds @parallel function rate_and_state_friction!(Vp, σII, Pt, Pf, a, b, Ω, F, Bool, L, σyield, ɛ̇II_plastic, ηvp::Data.Array, V0, γ0, Δt, σyieldmin, Wh, μˢ, GΔτₚᵗ)
        
    # NOTE: Peff  = Pt - Pf    
    @all(Vp)          = 2.0 * V0 * sinh(max(@all(σII), 0.0)/@all(a)/(@inn(Pt) - @inn(Pf))) * exp(-(@all(b)*@all(Ω) + γ0)/@all(a))

    @all(F)           = @all(Vp) * Δt / @all(L)    # compute new value for slip parameter
    @all(Bool)        = @all(F) > 1.0e-6           # matrix contains boolean values {0, 1}  

    # first term is assigned when the Bool evalutes to 1
    @all(Ω)           = @all(Bool) * log(V0/@all(Vp) + (exp(@all(Ω)) - V0/@all(Vp))*exp(-@all(Vp)*Δt/@all(L))) + (1 - @all(Bool)) * log(exp(@all(Ω)) * (1.0 - @all(Vp)*Δt/@all(L)) + V0*Δt/@all(L))   

    @all(σyield)      = max(σyieldmin, (@all(Pt) - @all(Pf)) * @all(a) * asinh(@all(Vp)/2.0/V0*exp((@all(b) * @all(Ω) + γ0)/@all(a))) )
    @all(ɛ̇II_plastic) = @all(Vp)/2.0/Wh
    
    # old FIXME: not sure if this is needed for Thibault's formulation
    # @inn(ηvp)         = μˢ* @all(σyield)/(2.0*μˢ*@all(ɛ̇II_plastic) + @all(σyield))

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
    @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy - ρfg)
    
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



@inbounds @parallel function injection_compute_pt_steps!(Δτₚᶠ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, Pfᵣ::Data.Array, min_dxy2::Data.Number)
    @inn(Δτₚᶠ) = min_dxy2/4.1/@maxloc(𝐤ɸ_µᶠ)/@inn(Pfᵣ)
    return nothing
end



##################### NUMERICAL ########################

"""The fluid injection model problem consists of a 2D square model domain Ω = [Lx, Ly]
where a 1D fault is embedded along y = Ly/2. We compare the result with the analytical solution
which consists of an error function.

    - with inertia
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
    lx       = 40000.0  # [m] 40 km
    ly       = 20000.0  # [m] 20 km
    nx       = 321
    ny       = 161

    dx, dy   = lx/(nx-1), ly/(ny-1)   # grid step in x, y
    @show mesh     = PTGrid((nx,ny), (lx,ly), (dx,dy))
    _dx, _dy = inv.(mesh.di)
    max_nxy  = max(nx,ny)
    min_dxy2 = min(dx,dy)^2
        
    # index for accessing the corresponding row of the interface
    @show h_index  = Int((ny - 1) / 2) + 1 # row index where the properties are stored for the fault


    # RHEOLOGY
    # porosity-dependent viscosity - for computing 𝞰ɸ    
    # in order to recover formulation in Dal Zilio (2022)
    C        = 1.0             # bulk/shear viscosity ratio
    R        = 1.0             # Compaction/decompaction strength ratio for bulk rheology

    # from table 1
    ɸ0       = 0.01            # reference porosity
    k0       = 1e-16           # reference permeability [m²]
    # μˢ       = 1e21            # solid shear viscosity [Pa·s]
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
    g        = 9.81998                  # gravitational acceleration [m/s^2]
    ρfg      = ρf * g                   # force fluid
    ρsg      = ρs * g                   # force solid
    ρtg      = ρt * g                   # force total - note for total density ρt = ρf·ɸ + ρs·(1-ɸ)
    
    flow     = TwoPhaseFlow2D(mesh, (ρfg, ρsg, ρtg))
    
    # Initial conditions
    ηɸ                   = μˢ/ɸ0
    𝞰ɸ                   = fill(μˢ/ɸ0, nx, ny)
    𝞀g                   = fill(ρtg, nx, ny)
    kɸ_fault             = 1e-15                         # domain with high permeability
    kɸ_domain            = 1e-22                         # domain with low permeability
    𝐤ɸ_µᶠ                = fill(kɸ_domain/µᶠ, nx, ny)    # porosity-dependent permeability
    𝐤ɸ_µᶠ[:, h_index]   .= kɸ_fault/µᶠ                   # along fault
    pf                   = 10.0e6                        # [Pa] = 10MPa Pf at t = 0
    Pf                   = fill(pf, nx, ny)
    pt                   = 40.0e6                        # [Pa] = 40MPa
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
    Vpl     =  1.0e-9          # loading rate [m/s]
    p⁻      = -1.0e-12         # BC top - outward flux [m/s]
    p⁺      =  1.0e-12         # BC bottom - inward flux [m/s]
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



    if RATE_AND_STATE_FRICTION
        # visco-elasto-plasticity
        # allocate Z, ηvp (same size as ∇V)
        # FIXME: not used
        Z        = @zeros(nx, ny)
        ηvp_cpu  = fill(μˢ, nx, ny)
        ηvp      = PTArray(ηvp_cpu)
        σII      = @zeros(nx-1, ny-1)   # same size as σxy

        Vp          = @zeros(nx-1, ny-1)
        F           = @zeros(nx-1, ny-1)
        σyield      = @zeros(nx-1, ny-1)
        ɛ̇II_plastic = @zeros(nx-1, ny-1)
        Bool_cpu    = fill(false, nx-1, ny-1)


        # Parameters for rsf
        #            domain   fault
        # a0        = [0.018    0.008]     # a-parameter of RSF
        a0        = [0.008    0.008]     # a-parameter of RSF
        b0        = [0.001    0.016]     # b-parameter of RSF
        Ω0        = [15           1]     # State variable from the preνious time step
        L0        = [0.012    0.012]     # L-parameter of RSF (characteristic slip distance)
        V0        = 1e-9                 # Reference slip velocity of RSF, m/s
        γ0        = 0.6                  # Ref. Static Friction
        Wh        = dy                   # fault width
        σyieldmin = 1e-3

        # assign along fault [:, h_index] for rate-strengthing/weakening regions        
        #    0km   4km    6km                 34km    36km  40km
        #    x0    x1     x2                   x3     x4    x5
        #    |*****|xxxxxx|                    |xxxxxx|*****|  
        #    -----------------------------------------------
        x0   = 0.0
        x1   = 4.0e3
        x2   = 6.0e3
        x3   = 34.0e3
        x4   = 36.0e3
        x5   = lx

        # for defining properties for rsf
        X = LinRange(0.0, lx, nx-1)
        Y = LinRange(0.0, ly, ny-1)


        # Use the velocity strengthening parameters as a-b>0 on the left and right extreme of the fault, for a length of 4 km each:
        # from 0 to 4 km in x-direction
        # from 36 to 40 km in x-direction
        a_cpu       = fill(a0[1], nx-1, ny-1)
        b_cpu       = fill(b0[1], nx-1, ny-1)
        Ω_cpu       = fill(Ω0[1], nx-1, ny-1)
        L_cpu       = fill(L0[1], nx-1, ny-1) # L0 is identical on both regions

        # setting up geometry
        # for i in 1:1:nx-1
        #     for j in 1:1:ny-1
    
        #         # if along the fault
        #         if j == h_index
    
        #             # assign domain value
        #             if x0 <= X[i] <= x1 || x4 <= X[i] <= x5
        #                 a_cpu[i,j] = a0[1]
        #                 b_cpu[i,j] = b0[1]
        #                 Ω_cpu[i,j] = Ω0[1]
        #             end
    
        #             # assign fault value
        #             if x2 <= X[i] <= x3
        #                 a_cpu[i,j] = a0[2]
        #                 b_cpu[i,j] = b0[2]
        #                 Ω_cpu[i,j] = Ω0[2]
        #             end
    
        #             # assign transition zone value (left)
        #             if x1 < X[i] < x2
        #                 a_cpu[i, j] = a0[1] - (a0[1] - a0[2]) * ((X[i] - x1) / (x2 - x1))
        #                 b_cpu[i, j] = b0[1] - (b0[1] - b0[2]) * ((X[i] - x1) / (x2 - x1))
        #                 Ω_cpu[i, j] = Ω0[1]
        #             end
    
        #             if x3 < X[i] < x4
        #                 a_cpu[i, j] = a0[2] - (a0[2] - a0[1]) * ((X[i] - x3) / (x4 - x3))
        #                 b_cpu[i, j] = b0[2] - (b0[2] - b0[1]) * ((X[i] - x3) / (x4 - x3))
        #                 Ω_cpu[i, j] = Ω0[1]
        #             end
    
        #         end
    
        #     end
        # end
        
        a           = PTArray(a_cpu)
        b           = PTArray(b_cpu)
        Bool        = PTArray(Bool_cpu)
        Ω           = PTArray(Ω_cpu)
        L           = PTArray(L_cpu)


        # adaptive time stepping
        # bulk modulus
        B = 1/βs
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
    # FIXME: yet will be overwritten later
    @show Δτᵥ_ρ = VpΔτ*max_lxy/Re/μˢ*1e17                    # original formulation with ρ = Re·µ
    @show GΔτₚᵗ = VpΔτ^2/(r+2.0)/Δτᵥ_ρ/1e1                 # special case for fluid injection 

    # darcy damping
    dampPf        = 0.6
    ηb            = 1.0
    Δτₚᶠ_cpu      = zeros(nx, ny)
    Δτₚᶠ          = PTArray(Δτₚᶠ_cpu)

    # define different reduce factors for the PT time step
    Pfᵣ_domain    = 1.0e5
    Pfᵣ_fault     = 40.0
    Pfᵣ_cpu       = fill(Pfᵣ_domain, nx, ny)

    # setting reduction of PT time step for fluid pressure along fault
    @. Pfᵣ_cpu[:, h_index]   = Pfᵣ_fault
    
    # @maxloc causes wrong time step size near fault, we need to set it to the same size as other places in the domain
    @. Pfᵣ_cpu[:, h_index-1] = 10.0       
    @. Pfᵣ_cpu[:, h_index+1] = 10.0
    Pfᵣ                      = PTArray(Pfᵣ_cpu)

    @parallel injection_compute_pt_steps!(Δτₚᶠ, flow.𝐤ɸ_µᶠ, Pfᵣ, min_dxy2)

    # make sure the Δτₚᶠ time steps are well-defined on boundaries
    @parallel (1:ny) injection_free_slip_x!(Δτₚᶠ)
    @parallel (1:nx) injection_free_slip_y!(Δτₚᶠ)
    @show extrema(Δτₚᶠ)


    # VISUALIZATION
    if DO_VIZ
        default(size=(2200,1500),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=7mm)
        scalefontsizes(); scalefontsizes(1.35)

        ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
    end

  
    # Time loop
    t_tot     = t_tot_           # total time
    Δt        = 3.0e6           # physical time-step
    t         = 0.0
    it        = 1
    ε         = 1e-7            # tolerance
    iterMax   = 5.5e4           # 5e3 for porosity wave, 5e5 previously
    nout      = 2
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

    if VISCOUS_ELASTO_PLASTICITY
        ɛ̇xy      = @zeros(nx+1, ny+1)
        σxxʼ_o   = @zeros(nx, ny)
        σxyʼ_o   = @zeros(nx-1, ny-1)
        σyyʼ_o   = @zeros(nx, ny)
        λ        = @zeros(nx-1, ny-1)


        # with solid viscosity shear modulus µ
        # FIXME: may need to make η_e variable with time!
        @show η_e   = µ*Δt
        @show η_ve  = 1.0/(1.0/μˢ  + 1.0/η_e)
        # η_vev = 1.0/(1.0/ηv + 1.0/η_ev)
    
        # @show Δτᵥ_ρ = VpΔτ*max_lxy/Re/μˢ*1e10                    # original formulation with ρ = Re·µ
        # @show GΔτₚᵗ = VpΔτ^2/(r+2.0)/Δτᵥ_ρ/1e8                 # special case for fluid injection 
    
        @show η_ve_τ  = 1.0/(1.0/μˢ + 1.0/η_e + 1.0/GΔτₚᵗ)
        # @. η_ve_τv = 1.0/(1.0/ηv + 1.0/η_ev + 1.0/GΔτₚᵗ)
    
        # FIXME: ?
        η_reg   = 0.5η_ve_τ


    end




    # record evolution of time step size and slip rate
    evo_t = Float64[]; evo_Δt = Float64[]; evo_Vp = Float64[]; evo_Peff = Float64[]

    while t<t_tot

        if INERTIA
            @parallel injection_assign_inertia!(flow.∇V_o, flow.∇V,  comp.Pt_o, flow.Pt, comp.Pf_o, flow.Pf, Vx_o, flow.V.x, Vy_o, flow.V.y, Vfx_o, Vfx, Vfy_o, Vfy, σxxʼ_o, flow.𝞂ʼ.xx, σxyʼ_o, flow.𝞂ʼ.xy, σyyʼ_o, flow.𝞂ʼ.yy)
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
                @parallel compute_ve_stress!(ɛ̇xy, flow.V.x, flow.V.y, flow.𝞂ʼ.xx, σxxʼ_o, flow.𝞂ʼ.yy, σyyʼ_o, flow.𝞂ʼ.xy, σxyʼ_o, η_ve_τ, η_e, GΔτₚᵗ, _dx, _dy)
                @parallel compute_second_invariant!(σII, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy)
                # @parallel rate_and_state_friction!(Vp[:, h_index], σII[:, h_index], flow.Pt[:, h_index], flow.Pf[:, h_index], a[:, h_index], b[:, h_index], Ω[:, h_index], F[:, h_index], Bool[:, h_index], L[:, h_index], σyield[:, h_index], ɛ̇II_plastic[:, h_index], ηvp[:, h_index], V0, γ0, Δt, σyieldmin, Wh, μˢ, GΔτₚᵗ)
                # @parallel compute_plastic_correction!(λ[:, h_index], σII[:, h_index], σyield[:, h_index], flow.𝞂ʼ.xx[:, h_index], flow.𝞂ʼ.yy[:, h_index], flow.𝞂ʼ.xy[:, h_index], η_ve_τ, η_reg)
                # @parallel compute_second_invariant!(σII, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy)
            else
                # compute stress tensor using viscous law
                @parallel injection_compute_tensor_newdamping!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, GΔτₚᵗ, μˢ, _dx, _dy)
                @parallel compute_second_invariant!(σII, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy)
            end
    
            if INERTIA
                @parallel stokesvep_compute_residual_momentum_law_newdamping_inertia!(flow.R.Vx, flow.R.Vy, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.V.x, Vx_o, flow.V.y, Vy_o, ρt, g, _dx, _dy, _Δt)
                @parallel stokesvep_compute_velocity_newdamping_inertia!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, Vfx, Vfy, Vfx_o, Vfy_o,  flow.R.Vx, flow.R.Vy, flow.𝐤ɸ_µᶠ, flow.Pf, Δτᵥ_ρ, ɸ0, ρf, g,  _dx, _dy, _Δt)
            else
                @parallel injection_compute_residual_momentum_law_newdamping!(flow.R.Vx, flow.R.Vy, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, _dx, _dy)
                @parallel injection_compute_velocity_newdamping!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, flow.R.Vx, flow.R.Vy, flow.𝐤ɸ_µᶠ, flow.Pf, Δτᵥ_ρ, ρfg,  _dx, _dy)                
            end
        
            
            # BOUNDARY CONDITIONS
            # the x-, y- coords here correspond to the coord before flipping
            @parallel (1:nx+1) injection_dirichlet_y!(flow.V.x, -0.5*Vpl, 0.5*Vpl)
            @parallel (1:nx)   injection_dirichlet_y!(flow.V.y, 0.0, 0.0)
            @parallel (1:ny)   injection_free_slip_x!(flow.V.x)
            @parallel (1:ny+1) injection_free_slip_x!(flow.V.y)
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
                
                if mod(iter,nout*10) == 0
                    @printf("iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_RPf=%1.3e, norm_RPt=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_RPf, norm_RPt)
                end
    
            end
    
            iter+=1; niter+=1
        end


        # FIXME: for adaptive time stepping
        # if RATE_AND_STATE_FRICTION
        #     @parallel adaptive_timestepping!(ξ, Bool_Δt, Δθmax, Δtdyn, Vp, a, b, flow.Pt, flow.Pf, L, K_timestepping)            
        #     # Δt = max[Δtmin, Δtdyn]
        #     #                        Δtdyn = Δθmax L/Vmax
        #     # @show Δt = max(Δt, minimum(Δtdyn))
        #     @show Δt = min(Δt, minimum(Δtdyn))
        # end


    
        # PERFORMANCE
        wtime    = Base.time() - wtime0
        A_eff    = (8*2)/1e9*nx*ny*sizeof(eltype(flow.Pt))   # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(niter-10)                         # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                           # Effective memory throughput [GB/s]
        @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))
      


        # DEBUG 
        @show extrema(flow.𝞂ʼ.xx[:, h_index])
        @show extrema(flow.𝞂ʼ.xx)

        @show extrema(flow.𝞂ʼ.yy[:, h_index])
        @show extrema(flow.𝞂ʼ.yy)

        @show extrema(flow.𝞂ʼ.xy[:, h_index])
        @show extrema(flow.𝞂ʼ.xy)

        @show extrema(σII[:, h_index])
        @show extrema(σII)

        # showing the update was fine with @all
        @show sum(Vp)
        @show sum(Vp[:, h_index])
        
        @show max_Vp   = maximum(Vp[:, h_index])
        @show max_Peff = maximum((flow.Pt[:, h_index] - flow.Pf[:, h_index])/1e6)
        
        # store evolution of physical properties wrt time
        push!(evo_t, t/time_year); push!(evo_Vp, max_Vp); push!(evo_Δt, Δt); push!(evo_Peff, maximum(max_Peff))

        # store fluid pressure for wanted time points
        if STORE_DATA && mod(it, 5) == 0
            save("earthquake_cycles/max_Vp_fault" * string(it) * ".jld", "data", Array(Vp[:, h_index])')   # store the fluid pressure along the fault for fluid injection benchmark
        end
    
        
        # Debug
        @show t   = t + Δt
        it += 1

        
        # Visualisation
        if DO_VIZ
            
            # plotting the fault
            if PLOTTING_CYCLES && mod(it,1) == 0
                p1 = plot(evo_t, evo_Vp; xlims=(0.0, 1100), ylims=(1.0e-10, 1.0e2), yaxis=:log, yticks =[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], label="", color= :dodgerblue, framestyle= :box, linestyle= :solid, 
                         seriesstyle= :path, title="H-MEC Simulation (t = " * string(@sprintf("%.3f", t/time_year)) * " year )", 
                            xlabel = "Time [year]", ylabel="Slip Rate [m/s]" )

                p2 = plot(evo_t, evo_Δt; xlims=(0.0, 1100), ylims=(1e-2, 8.0e7), yaxis=:log, yticks = [1e-2, 1e0, 1e2, 1e4, 1e6, 1e8], label="", color= :tomato, framestyle= :box, linestyle= :solid, 
                    seriesstyle= :path, title="",
                    xlabel = "Time [year]", ylabel="Time step size [s]" )

                p3 = plot(evo_t, evo_Peff; xlims=(0.0, 1100), ylims=(0.0, 35.0), yticks=[10, 20, 30], label="", color= :forestgreen, framestyle= :box, linestyle= :solid, 
                    seriesstyle= :path, title="", 
                    xlabel = "Time [year]", ylabel="Effective pressure [MPa]" )

   
                display(plot(p1, p2, p3; layout=(3,1))); frame(anim)
            end
            
            # plotting whole domain
            if PLOTTING_DOMAIN && mod(it,1) == 0

                X_plot, Y_plot, Xv_plot, Yv_plot = 0:dx:lx, 0:dy:ly, (-dx/2):dx:(lx+dx/2), (-dy/2):dy:(ly+dy/2)
                p1 = heatmap(Xv_plot, Y_plot, Array(flow.V.x)', aspect_ratio=1, xlims=(Xv_plot[1],Xv_plot[end]), ylims=(Y_plot[1],Y_plot[end]), c=:RdBu, title="horizontal velocity")
                p2 = heatmap(X_plot, Yv_plot, Array(flow.V.y)', aspect_ratio=1, xlims=(X_plot[1],X_plot[end]), ylims=(Yv_plot[1],Yv_plot[end]), c=:RdBu, title="vertical velocity")
                p3 = heatmap(X_plot, Y_plot, Array(σII)', aspect_ratio=1, xlims=(X_plot[1],X_plot[end]), ylims=(Y_plot[1],Y_plot[end]), c=:RdBu, title="2nd stress invariant")
                p4 = heatmap(Xv_plot, Y_plot, Array(flow.qD.x)', aspect_ratio=1, xlims=(Xv_plot[1],Xv_plot[end]), ylims=(Y_plot[1],Y_plot[end]), c=:RdBu, title="horizontal darcy velocity")
                p5 = heatmap(X_plot, Yv_plot, Array(flow.qD.y)', aspect_ratio=1, xlims=(X_plot[1],X_plot[end]), ylims=(Yv_plot[1],Yv_plot[end]), c=:RdBu, title="vertical darcy velocity")
                p6 = heatmap(X_plot, Y_plot, Array(σyield)', aspect_ratio=1, xlims=(X_plot[1],X_plot[end]), ylims=(Y_plot[1],Y_plot[end]), c=:RdBu, title="yielding stress")
                p7 = heatmap(X_plot, Y_plot, Array(flow.Pt-flow.Pf)', aspect_ratio=1, xlims=(X_plot[1],X_plot[end]), ylims=(Y_plot[1],Y_plot[end]), c=:RdBu, title="effective pressure")
                p8 = heatmap(X_plot, Y_plot, Array(flow.Pf)', aspect_ratio=1, xlims=(X_plot[1],X_plot[end]), ylims=(Y_plot[1],Y_plot[end]), c=:RdBu, title="fluid pressure")
                p9 = heatmap(X_plot, Y_plot, Array(ηvp)', aspect_ratio=1, xlims=(X_plot[1],X_plot[end]), ylims=(Y_plot[1],Y_plot[end]), c=:RdBu, title="effective viscosity")

                display(plot(p1, p2, p3, p4, p5, p6, p7, p8, p9; layout=(3,3))); frame(anim)

            end

        end
    end
    
    # Visualization
    if DO_VIZ
        gif(anim, "earthquake_cycles.gif", fps = 15)
    end


    if STORE_DATA
        save("earthquake_cycles/evo_t" * string(it) * ".jld", "data", Array(evo_t)')   # store the fluid pressure along the fault for fluid injection benchmark
        save("earthquake_cycles/evo_Vp" * string(it) * ".jld", "data", Array(evo_Vp)')   # store the fluid pressure along the fault for fluid injection benchmark
        save("earthquake_cycles/evo_Δt" * string(it) * ".jld", "data", Array(evo_Δt)')   # store the fluid pressure along the fault for fluid injection benchmark
        save("earthquake_cycles/evo_Peff" * string(it) * ".jld", "data", Array(evo_Peff)')   # store the fluid pressure along the fault for fluid injection benchmark
    end


    # return effective pressure at final time
    return Array(flow.Pt - flow.Pf)[:, h_index-1:h_index+1]'

end



if isinteractive()
    earthquake_cycles(;t_tot_= 3.0e6) # 1 step
    # earthquake_cycles(;t_tot_= 1.262304e10)  # 400 years
end
