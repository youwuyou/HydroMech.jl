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



const STORE_PRESSURE        = false   # set to true for stroing fluid pressure along fault
const NEW_DAMPING           = true


@static if STORE_PRESSURE
    using JLD
end



##################### ANALYTICAL ########################
""" Uniform pore fluid pressure distribution
Reference: Self-similar fault slip in repsponse to fluid injection (Viesca 2008)

- where ld(t) = √(ɑ * t) is the diffusivity length scale
"""
function P(x::Float64, t::Float64; 
            P₀  = 5e6,         # [Pa] initial pore pressure                   p(x,0)    = P₀ 
            Δpf = 5e6,         # [Pa] injection proceeds at constant pressure p(0, t>0) = Δp
            ηf  = 1e-3,        # [Pa·s] viscosity of the permeating fluid
            # kᵩ  = 1e-16,     # [m²]   Darcy permeability of the layer    (table 1 value) -> calcuated from kᵩ = k* (φ)
            kᵩ  = 1e-15,       # [m²]   Darcy permeability of the layer  -> calcuated from kᵩ = k* (φ)
            
            # calculated from values in table 1  | with βs = 2.5e-11, βf = 4.0e-10
            βd  = 2.5555555555555557e-11
        )

        
    ɑₕ = kᵩ / (ηf * βd)# hydraulic diffusivity
    ɑ = 4 * ɑₕ
    

    return P₀ + Δpf * erfc(norm(x) / sqrt(ɑ * t))
end




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


# compute residual for fluid and solid mass conservation eq
@inbounds @parallel function injection_compute_pressure!(Pt::Data.Array, Pf::Data.Array, fᴾᵗ::Data.Array, fᴾᶠ::Data.Array, Δτₚᶠ::Data.Array, Δτₚᵗ::Data.Number)

    # solid mass, total pressure update
    # ptⁿ = ptⁿ⁻¹ + Δτ_pt f_ptⁿ
    @all(Pt)  = @all(Pt) + Δτₚᵗ * @all(fᴾᵗ)
   
    # fluid mass, fluid pressure update
    # pfⁿ = pfⁿ⁻¹ + Δτ_pf f_pfⁿ
    @all(Pf)  = @all(Pf) + @all(Δτₚᶠ)*@all(fᴾᶠ)
    
    return nothing
end


# old damping approach - direct computation of stress
@inbounds @parallel function injection_compute_tensor!(σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, fᴾᵗ::Data.Array, μˢ::Data.Number, ηb::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # General formula for viscous creep shear rheology
    # μˢ <-> solid shear viscosity
    # σᵢⱼ' = 2μˢ · ɛ̇ᵢⱼ = 2μˢ · (1/2 (∇ᵢvⱼˢ + ∇ⱼvᵢˢ) - 1/3 δᵢⱼ ∇ₖvₖˢ)
    @all(σxxʼ) = 2.0*μˢ*( @d_xa(Vx)* _dx - 1.0/3.0*@all(∇V) - ηb*@all(fᴾᵗ) )
    @all(σyyʼ) = 2.0*μˢ*( @d_ya(Vy)* _dy - 1.0/3.0*@all(∇V) - ηb*@all(fᴾᵗ) )

    # compute the xy component of the deviatoric stress
    # σxy' = 2μˢ · ɛ̇xy = 2μˢ · 1/2 (∂Vx/∂y + ∂Vy/∂x) =  μˢ · (∂Vx/∂y + ∂Vy/∂x)     
    @all(σxyʼ) = 2.0*μˢ*(0.5*( @d_yi(Vx)* _dy + @d_xi(Vy)* _dx ))

    return nothing
end


# compute residual for stokes equation
@inbounds @parallel function injection_compute_residual_momentum_law!(fᵛˣ::Data.Array, fᵛʸ::Data.Array, gᵛˣ::Data.Array, gᵛʸ::Data.Array, σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Pt::Data.Array, 𝞀g::Data.Array, dampVx::Data.Number, dampVy::Data.Number, _dx::Data.Number, _dy::Data.Number)
    
    # common Cartesian coordinates with y-axis positive pointing upwards
    # @all(fᵛˣ)    = (@d_xi(σxxʼ)- @d_xi(Pt))* _dx + @d_ya(σxyʼ)* _dy 
    # @all(fᵛʸ)    = (@d_yi(σyyʼ)- @d_yi(Pt))* _dy + @d_xa(σxyʼ)* _dx - @av_yi(𝞀g)   # 𝞀g = ρtg with total (background) density

    # geological coordinates y-axis positive pointing downwards
    @all(fᵛˣ)    = (@d_xi(σxxʼ)- @d_xi(Pt))* _dx + @d_ya(σxyʼ)* _dy 
    @all(fᵛʸ)    = (@d_yi(σyyʼ)- @d_yi(Pt))* _dy + @d_xa(σxyʼ)* _dx + @av_yi(𝞀g)

    # apply damping terms for the residual
    @all(gᵛˣ) = dampVx * @all(gᵛˣ) + @all(fᵛˣ)
    @all(gᵛʸ) = dampVy * @all(gᵛʸ) + @all(fᵛʸ)
    return nothing
end


# i).without inertia
@inbounds @parallel function injection_compute_velocity!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, gᵛˣ::Data.Array, gᵛʸ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, Pf::Data.Array, Δτᵥ::Data.Number, ρfg::Data.Number,  _dx::Data.Number, _dy::Data.Number)

    # i). total momentum, velocity update
    # vᵢⁿ = vᵢⁿ⁻¹ + Δτ_vᵢ g_vᵢⁿ for i in x,y
    @inn(Vx)  =  @inn(Vx) + Δτᵥ* @all(gᵛˣ)
    @inn(Vy)  =  @inn(Vy) + Δτᵥ* @all(gᵛʸ)

    
    # ii). fluid momentum, velocity update
    # qDᵢⁿ = - k^ɸ/ µ^f (∇Pf - ρf·g)

    # common cartesian coords
    # @inn(qDx) = -@av_xi(𝐤ɸ_µᶠ)*(@d_xi(Pf)* _dx)
    # @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy + ρfg)

    # geological coords
    @inn(qDx) = -@av_xi(𝐤ɸ_µᶠ)*(@d_xi(Pf)* _dx)
    @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy - ρfg )

    
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



@inbounds @parallel function injection_compute_tensor_newdamping!(σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, fᴾᵗ::Data.Array, GΔτₚᵗ::Data.Number, μˢ::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # General formula for viscous creep shear rheology
    # μˢ <-> solid shear viscosity
    @all(σxxʼ) = (@all(σxxʼ) + 2.0*GΔτₚᵗ*@d_xa(Vx)* _dx) / (GΔτₚᵗ/μˢ + 1.0)
    @all(σyyʼ) = (@all(σyyʼ) + 2.0*GΔτₚᵗ*@d_ya(Vy)* _dy) / (GΔτₚᵗ/μˢ + 1.0)
    @all(σxyʼ) = (@all(σxyʼ) + 2.0*GΔτₚᵗ*(0.5*(@d_yi(Vx)* _dy + @d_xi(Vy)* _dx)))/(GΔτₚᵗ/μˢ + 1.0)

    return nothing
end




# compute residual for stokes equation
@inbounds @parallel function injection_compute_residual_momentum_law_newdamping!(fᵛˣ::Data.Array, fᵛʸ::Data.Array, σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Pt::Data.Array, 𝞀g::Data.Array, _dx::Data.Number, _dy::Data.Number)

    # compute residual f_vᵢⁿ for total momentum
    # geological coordinates y-axis positive pointing downwards
    @all(fᵛˣ)    = (@d_xi(σxxʼ)- @d_xi(Pt))* _dx + @d_ya(σxyʼ)* _dy 
    @all(fᵛʸ)    = (@d_yi(σyyʼ)- @d_yi(Pt))* _dy + @d_xa(σxyʼ)* _dx + @av_yi(𝞀g)

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
@views function fluid_injection(;t_tot_)


    # MESH
    lx       = 50.0  # [m]
    ly       = 10.0  # [m]
    nx       = 1001
    ny       = 201

    @show dx, dy  = lx/(nx-1), ly/(ny-1)   # grid step in x, y
    @show mesh    = PTGrid((nx,ny), (lx,ly), (dx,dy))
    _dx, _dy      = inv.(mesh.di)
    max_nxy       = max(nx,ny)
    min_dxy2      = min(dx,dy)^2
        
    # index for accessing the corresponding row of the interface
    @show h_index = Int((ny - 1) / 2) + 1 # row index where the properties are stored for the fault


    # RHEOLOGY
    # porosity-dependent viscosity - for computing 𝞰ɸ    
    # in order to recover formulation in Dal Zilio (2022)
    C        = 1.0             # bulk/shear viscosity ratio
    R        = 1.0             # Compaction/decompaction strength ratio for bulk rheology

    # from table 1
    ɸ0       = 0.01            # reference porosity   -> 1%
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
    # forces
    ρf       = 1.0e3                    # fluid density 1000 kg/m^3
    ρs       = 2.7e3                    # solid density 2700 kg/m^3
    g        = 0.0                      # gravitational acceleration [m/s^2]
    # g        = 9.81998                # g = 0.0 for fluid injection benchmark
    ρfg      = ρf * g                   # force fluid
    ρsg      = ρs * g                   # force solid
    ρBG      = ρf*ɸ0 + ρs*(1.0-ɸ0)      # density total (background)
    ρgBG     = ρBG * g                  # force total - note for total density ρt = ρf·ɸ + ρs·(1-ɸ)
    
    flow                  = TwoPhaseFlow2D(mesh, (ρfg, ρsg, ρgBG))
    
    # Initial conditions
    @show ηɸ              = μˢ/ɸ0
    𝞰ɸ                    = fill(μˢ/ɸ0, nx, ny)
    𝞀g                    = fill(ρgBG, nx, ny)
    
    kɸ_fault              = 1e-15                         # domain with high permeability
    kɸ_domain             = 1e-22                         # domain with low permeability
    𝐤ɸ_µᶠ                 = fill(kɸ_domain/µᶠ, nx, ny)    # porosity-dependent permeability
    𝐤ɸ_µᶠ[:, h_index]    .= kɸ_fault/µᶠ                   # along fault


    pf                   = 5.0e6                         # [Pa] = 5MPa Pf at t = 0
    Pf                   = fill(pf, nx, ny)

    pt                   = 20.0e6                        #  [Pa] = 20MPa
    Pt                   = fill(pt, nx, ny)
    
    flow.𝞰ɸ              = PTArray(𝞰ɸ)
    flow.𝐤ɸ_µᶠ           = PTArray(𝐤ɸ_µᶠ)
    flow.Pf              = PTArray(Pf)
    flow.Pt              = PTArray(Pt)
    flow.𝞀g              = PTArray(𝞀g)     # initialize here because we don't have porosity update in current code
    #====================#
    
    # Fluid injection specific
    @show Peff           = pt - pf          # constant effective pressure [Pa] -> 15MPa 
    p₀f                  = 5.0e6            # initial fluid pressure 5 MPa
    Δpf                  = 5.0e6            # constant amount of fluid to be injected 5 MPa

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
    

    # PT COEFFICIENT
    # scalar shear viscosity μˢ = 1.0 was used in porosity wave benchmark to construct dτPt    
    if NEW_DAMPING
        max_lxy   = max(lx, ly)
        min_lxy   = min(lx, ly)
        max_dxy   = max(dx, dy)
        max_dxy2  = max(dx,dy)^2


        # numerical velocity
        #       Vp = CΔx/Δτ
        # ⇔  VpΔτ = CΔx
        CFL       = 0.9/sqrt(2)
        VpΔτ      = CFL*min(dx,dy)
        Re        = 5π
        r         = 1.0


        # stokes damping
        @show Δτᵥ_ρ = VpΔτ*max_lxy/Re/μˢ    # original formulation with ρ = Re·µ
        # @show Δτᵥ_ρ = VpΔτ*max_lxy/Re    # ρ = Re·µ


        # @show GΔτₚᵗ    = VpΔτ^2/(r+2.0)/Δτᵥ_ρ   # original formulation with G = ρV²/(r+2)
        @show GΔτₚᵗ    = VpΔτ^2/(r+2.0)/Δτᵥ_ρ/μˢ/1e3   # special case for fluid injection 

        # darcy damping
        dampPf        = 0.6
        ηb            = 1.0
        Δτₚᶠ_cpu      = zeros(nx, ny)
        Δτₚᶠ          = PTArray(Δτₚᶠ_cpu)

        # define different reduce factors for the PT time step
        Pfᵣ_domain               = 1.0e7
        Pfᵣ_fault                = 40.0
        Pfᵣ_cpu                  = fill(Pfᵣ_domain, nx, ny)

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

    else
        # old damping approach
        Pfᵣ = 5.0e9
        pt = PTCoeff(OriginalDamping, mesh, 1e23, Pfᵣ = Pfᵣ, Ptᵣ = 1.0e25, Vᵣ = 0.825, dampPf = 1.0, dampV = 4.0)        # choose this norm does not get smaller than norm_Rx=5.261e-01, dt = 15
        @parallel injection_compute_pt_steps!(pt.Δτₚᶠ, flow.𝐤ɸ_µᶠ, min_dxy2, Pfᵣ)
    end    

    

    # Preparation of visualisation
    if DO_VIZ
        default(size=(1000,800), margin=2mm)
        ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")


        # calculate for 25 grid points spanning from x ∈ [0, 50]
        X_plot =  LinRange(0,50,nx)

        X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
        Xv          = (-dx/2):dx:(lx+dx/2)
    end
  

    # Time loop
    t_tot    = t_tot_          # total time
    Δt       = 5.0             # physical time-step
    t        = 0.0
    it       = 1
    ε        = 1e-13           # tolerance
    iterMax  = 1e4             # 5e3 for porosity wave, 5e5 previously
    nout     = 200


    # precomputation
    _Δt        = inv(Δt)
    length_Rx  = length(flow.R.Vx)
    length_Ry  = length(flow.R.Vy)
    length_RPf = length(flow.R.Pf)
    length_RPt = length(flow.R.Pt)
    _C         = inv(rheology.C)
    _ɸ0        = inv(rheology.ɸ0)
    _Ks        = inv(comp.Ks)

    
    iter_evo = Float64[]; err_evo = Float64[]

    while t<t_tot

        @parallel injection_assign!(flow.∇V_o, comp.Pt_o, comp.Pf_o, flow.∇V, flow.Pt, flow.Pf)
        err=2*ε; iter=1; niter=0
        
        while err > ε && iter <= iterMax
            if (iter==11)  global wtime0 = Base.time()  end
    
            @parallel injection_compute_∇!(flow.∇V, flow.∇qD, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, _dx, _dy)            
            @parallel injection_compute_residual_mass_law!(flow.R.Pt, flow.R.Pf, flow.𝐤ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, ɸ0, comp.𝗞d, comp.𝝰, comp.Pt_o, comp.Pf_o, comp.𝗕, dampPf, min_dxy2, _Δt)
            
            
            if NEW_DAMPING
                @parallel injection_compute_pressure_newdamping!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, Δτₚᶠ, GΔτₚᵗ, r)
                @parallel injection_compute_tensor_newdamping!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, GΔτₚᵗ, μˢ, _dx, _dy)
                @parallel injection_compute_residual_momentum_law_newdamping!(flow.R.Vx, flow.R.Vy, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, _dx, _dy)
                @parallel injection_compute_velocity_newdamping!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, flow.R.Vx, flow.R.Vy, flow.𝐤ɸ_µᶠ, flow.Pf, Δτᵥ_ρ, ρfg,  _dx, _dy)                
        
            else
                @parallel (1:ny) injection_free_slip_x!(pt.Δτₚᶠ)
                @parallel (1:nx) injection_free_slip_y!(pt.Δτₚᶠ)
                @parallel injection_compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.Δτₚᶠ, pt.Δτₚᵗ)
                @parallel injection_compute_tensor!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, rheology.μˢ, pt.ηb, _dx, _dy)
                @parallel injection_compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.gᵛˣ, pt.gᵛʸ, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, pt.dampVx, pt.dampVy, _dx, _dy)
                @parallel injection_compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.gᵛˣ, pt.gᵛʸ, flow.𝐤ɸ_µᶠ, flow.Pf, pt.Δτᵥ, flow.ρfg, _dx, _dy)
            end
            
            
            # BOUNDARY CONDITIONS
            @parallel (1:ny)       injection_dirichlet_x!(flow.V.x, 0.0, 0.0)
            @parallel (1:nx+1)     injection_dirichlet_y!(flow.V.x, 0.0, 0.0)
            @parallel (1:ny+1)     injection_dirichlet_x!(flow.V.y, 0.0, 0.0)
            @parallel (1:nx)       injection_dirichlet_y!(flow.V.y, 0.0, 0.0)
            @parallel (1:nx)       injection_constant_effective_pressure_x!(flow.Pf, flow.Pt, Peff) # confining pressure to boundaries parallel to x-axis

            # injection fluid to a single point
            flow.Pf[1, h_index] = p₀f + Δpf

            # used for fluid injection benchmark! Otherwise not!
            #        injection point [1, h_index]
            #                  \      ny
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
            # inject  |                                | ny
            #      \  |                                |
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
    
        # PERFORMANCE
        wtime    = Base.time() - wtime0
        A_eff    = (8*2)/1e9*nx*ny*sizeof(eltype(flow.Pt))   # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(niter-10)                         # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                           # Effective memory throughput [GB/s]
        @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))
      

        push!(iter_evo,iter/nx); push!(err_evo,err)


        # store fluid pressure for wanted time points
        if STORE_PRESSURE && mod(it, 1) == 0
            save("fluid_injection/Pf_fault" * string(it) * ".jld", "data", Array(flow.Pf[:, h_index])')   # store the fluid pressure along the fault for fluid injection benchmark
        end
    
        
        # Debug
        @show it
        @show t   = t + Δt
        it += 1

        
        # Visualisation
        if DO_VIZ
            
            if mod(it,1) == 0
                
                default(size=(1100,780),fontfamily="Computer Modern", linewidth=3, framestyle=:box, margin=6.0mm)
                scalefontsizes(); scalefontsizes(1.35)


                # compute analytical solution at current time
                Pressure = P.(X_plot,t)
                Pressure /= 1e6

                if t == 4000
                    @show Pressure[1:8]
                end
        
                # plotting with unit [MPa x m]
                p1 = plot(X_plot, Pressure; label="Analytical", color= :dodgerblue,
                            xlims=(0, 50), ylims = (4,10.8), framestyle= :box, linestyle= :dash, seriesstyle= :path, 
                            aspect_ratio = 5.0, title="Fluid injection benchmark (BP1) (t = " * string(t) * " sec )", 
                            xlabel = "Distance from the injection point [m]", ylabel="Fluid Pressure [MPa]")

                plot!(p1, X_plot, Array(flow.Pf[:, h_index]/1e6); color= :tomato, label="Numerical", linestyle= :solid)

                display(plot(p1;layout=(1,1))); frame(anim)

                # DEBUG
                @show norm(Array(flow.Pf[:, h_index])/1e6 - Pressure)/norm(Pressure)   # compute relative error


            end            

        end
    end
    
    # Visualization
    if DO_VIZ
        gif(anim, "fault1D_injection_compressible.gif", fps = 15)
    end

    # return effective pressure at final time
    return Array(flow.Pt - flow.Pf)'

end


if isinteractive()
    fluid_injection(;t_tot_= 4000) # for reproducing fluid injection benchmark
end
