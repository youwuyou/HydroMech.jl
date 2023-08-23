# Physics: Incompressible stokes equation with VEP + rate- and state-dependent friction
#          - momentum equation with inertia effects included

using HydroMech

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)


# NOTE: despite of using the package we initialize here again because 
# we need to use the type Data.Array, Data.Number for argument types
const USE_GPU    = true  # Use GPU? If this is set false, then no GPU needs to be available
const STORE_DATA = true

@static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
else
        @init_parallel_stencil(Threads, Float64, 2)
end

@static if STORE_DATA
    using JLD
end

const ASYMMETRIC      = false

using Plots, Plots.Measures, LinearAlgebra,Printf, LaTeXStrings

# helper functions
@views maxloc(A) = max.(A[1:end-2,1:end-2],A[1:end-2,2:end-1],A[1:end-2,3:end],
                        A[2:end-1,1:end-2],A[2:end-1,2:end-1],A[2:end-1,3:end],
                        A[3:end  ,1:end-2],A[3:end  ,2:end-1],A[3:end  ,3:end])
@views   bc2!(A) = begin A[1,:] = A[2,:]; A[end,:] = A[end-1,:]; A[:,1] = A[:,2]; A[:,end] = A[:,end-1] end


# VISUALIZATION
if DO_VIZ
    # default(size=(3500,1700),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=7mm)
    default(size=(2000,1900),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=7mm)
    scalefontsizes(); scalefontsizes(1.35)
    # scalefontsizes(); scalefontsizes(1.20)
    
    ENV["GKSwstype"]="nul"; if isdir("vizGPU_out")==false mkdir("vizGPU_out") end; loadpath = "./vizGPU_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
end

# KERNELS
@inbounds @parallel function assign!(τxx_o, τxx, τyy_o, τyy, τxy_o, τxy, τxyc_o, τxyc, Pr_o, Pr, Vx_o, Vx, Vy_o, Vy, Ω_o, Ω)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    @all(τxyc_o) = @all(τxyc)

    @all(Pr_o) = @all(Pr)

    # inertia
    @all(Vx_o) = @all(Vx)
    @all(Vy_o) = @all(Vy)

    # aging law
    @all(Ω_o) = @all(Ω)

    return nothing
end

# PT parameters update
@inbounds @parallel function compute_params1!(η_e, η_ev, G, Gv, dt)
    @all(η_e)   = @all(G)*dt
    @all(η_ev)  = @all(Gv)*dt        

    return nothing
end

@inbounds @parallel function compute_params2!(η_ve, η_vev, η, ηv, η_e, η_ev)
    @all(η_ve)  = 1.0/(1.0/@all(η)  + 1.0/@all(η_e))
    @all(η_vev) = 1.0/(1.0/@all(ηv) + 1.0/@all(η_ev))

    return nothing
end

@inbounds @parallel function compute_params3!(η_vem, η_vevm, η_ve, η_vev)
    @inn(η_vem)  = @maxloc(η_ve)
    @inn(η_vevm) = @maxloc(η_vev)
    return nothing
end



@inbounds @parallel function compute_params4!(dτ_ρ, dτ_ρv, vpdτ, max_lxy, _Re, η_vem, η_vevm)
    @all(dτ_ρ)    = vpdτ*max_lxy* _Re/@all(η_vem)
    @all(dτ_ρv)   = vpdτ*max_lxy* _Re/@all(η_vevm)
    return nothing
end


@inbounds @parallel function compute_params5!(Gdτ, Gdτv, vpdτ, dτ_ρ, dτ_ρv, r)
    @all(Gdτ)     = vpdτ^2/@all(dτ_ρ)/(r+2.0)
    @all(Gdτv)    = vpdτ^2/@all(dτ_ρv)/(r+2.0)    

    return nothing
end


@inbounds @parallel function compute_params6!(η_ve_τ, η_ve_τv, η, ηv, η_e, η_ev, Gdτ, Gdτv)

    @all(η_ve_τ)  = 1.0/(1.0/@all(η) + 1.0/@all(η_e) + 1.0/@all(Gdτ))
    @all(η_ve_τv) = 1.0/(1.0/@all(ηv) + 1.0/@all(η_ev) + 1.0/@all(Gdτv))        

    return nothing
end


@inbounds @parallel function compute_∇!(∇V, Vx, Vy, _dx, _dy)
    @all(∇V)    = @d_xa(Vx)* _dx  + @d_ya(Vy)* _dy
    return nothing
end

@inbounds @parallel function compute_residual_mass_law!(Rp, ∇V)
    @all(Rp)    = -@all(∇V)
    return nothing
end


@inbounds @parallel function compute_pressure!(Pr, Gdτ, Rp, r)
    @all(Pr) = @all(Pr) + r*@all(Gdτ)*@all(Rp)
    return nothing
end



@inbounds @parallel function compute_strain_rate!(Exy, Vx, Vy, _dx, _dy)

    @inn(Exy)  = 0.5 * (@d_yi(Vx) * _dy + @d_xi(Vy) * _dx)

    return nothing
end




@inbounds @parallel function compute_ve_stress!(Exy, τxx, τyy, τxyc, τxx_o, τyy_o, τxyc_o, Vx, Vy, η_ve_τ, η_e, Gdτ, _dx, _dy)

    @all(τxx)  = 2.0*@all(η_ve_τ)*(@d_xa(Vx)* _dx + 0.5*@all(τxx_o)/@all(η_e) + 0.5*@all(τxx)/@all(Gdτ))
    @all(τyy)  = 2.0*@all(η_ve_τ)*(@d_ya(Vy)* _dy + 0.5*@all(τyy_o)/@all(η_e) + 0.5*@all(τyy)/@all(Gdτ))
    @all(τxyc) = 2.0*@all(η_ve_τ)*(@av(Exy) + 0.5*@all(τxyc_o)/@all(η_e) + 0.5*@all(τxyc)/@all(Gdτ))

    return nothing
end 


@inbounds @parallel function compute_tensor!(τxy, η_ve_τv, Exy, τxy_o, η_ev, Gdτv)

    @all(τxy) = 2.0*@all(η_ve_τv)*(@all(Exy) + 0.5*@all(τxy_o)/@all(η_ev) + 0.5*@all(τxy)/@all(Gdτv))

    return nothing
end


@inbounds @parallel function compute_slip_rate!(Vp, τii, a, b, Ω, γ, Pt, V0, mask)

    # if no smoothing was used computation would have been:
    @all(Vp) = 2.0*V0*sinh(@all(τii)/(@all(a)*Pt))/exp((@all(b)*@all(Ω) + γ)/@all(a)) * @all(mask)

    # we employ a smoothing technique, chosen relaxation parameter θ = 0.5 here
    # @all(Vp) = (1.0 - 0.5) * @all(Vp) + 0.5*(2.0*V0*sinh(@all(τii)/@all(a)/Pt)/exp((@all(b)*@all(Ω) + γ)/@all(a)))* @all(mask)

    return nothing
end




@inbounds @parallel function compute_plastic_multiplier2!(λ, Vp, _dx, dt, η_ve_τ)

    # estimation of plastic multiplier => blowing up before seismic events (Casper)
    # @all(λ)  = 0.5*@all(Vp)*_dx


    # estimation of plastic multiplier => stagnates after Vp reaches 1e-1 (You)
    @all(λ)  = (dt*@all(Vp))/(@all(η_ve_τ))
    
    return nothing
end



@inbounds @parallel function compute_plastic_correction!(λ, τxx, τyy, τxyc, τii, η_ve_τ)

    @all(τxx)  = @all(τxx) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τxx)/@all(τii))
    @all(τyy)  = @all(τyy) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τyy)/@all(τii))
    @all(τxyc) = @all(τxyc) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τxyc)/@all(τii))

    return nothing
end

@parallel_indices (i, j) function compute_plastic_correction2!(λ, τxx, τyy, τxyc, τii, η_ve_τ)

    a          = 2.0 * η_ve_τ[i,j]
    b          = λ[i, j] * 0.5 / τii[i, j]
    c          = a*b
    τxx[i,j]  -= c * τxx[i, j] 
    τyy[i,j]  -= c * τyy[i, j] 
    τxyc[i,j] -= c * τxyc[i, j]

    return nothing
end

@inbounds @parallel function compute_plastic_correction_xy!(λ, τxy, τii, η_ve_τv, mask_tauxy)
    @inn(τxy) = @inn(τxy) - 2.0*@inn(η_ve_τv)*(@inn(λ)*0.5*@inn(τxy)/@av(τii))* @inn(mask_tauxy)
    return
end

@parallel function center2vertex!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end

@inbounds @parallel function compute_second_invariant!(τii, τxx, τyy, τxyc)

    @all(τii) = sqrt(0.5*(@all(τxx)^2 + @all(τyy)^2) + @all(τxyc)^2)

    return nothing
end

@inbounds @parallel function compute_residual_momentum_law!(Rx, Ry, τxx, τyy, τxy, Pr, ρ, ρg, Vx, Vx_o, Vy, Vy_o, _dx, _dy, _dt)
    @all(Rx) = (@d_xa(τxx) - @d_xa(Pr)) * _dx + @d_yi(τxy) * _dy - @av_xa(ρ) * ( @inn_x(Vx) - @inn_x(Vx_o)) * _dt
    @all(Ry) = (@d_ya(τyy) - @d_ya(Pr)) * _dy + @d_xi(τxy) * _dx + @av_ya(ρg) - @av_ya(ρ) * ( @inn_y(Vy) - @inn_y(Vy_o)) * _dt
    
    return nothing
end


@inbounds @parallel function compute_velocity!(Vx, Vy, Rx, Ry, dτ_ρ)

    @inn_x(Vx) = @inn_x(Vx) + @av_xa(dτ_ρ) * @all(Rx)
    @inn_y(Vy) = @inn_y(Vy) + @av_ya(dτ_ρ) * @all(Ry)

    return nothing
end



@inbounds @parallel function compute_state_parameter!(Ω, Ω_old, Vp, dt, V0, L, w)

    @all(Ω) = @all(Ω_old) + dt * (V0 * exp(-@all(Ω_old)) - @all(Vp)) / L

    return nothing
end

@parallel_indices (i, j) function compute_state_parameter2!(Ω, Ω_old, Vp, dt, V0, L, w)
    
    Vpij      = @inbounds Vp[i,j]
    _Vpij     = @inbounds inv(Vpij)
    Ω_oldij   = @inbounds exp(Ω_old[i,j])
    dt_L      = dt * inv(L)
    condition = Vpij*dt_L > 1e-6

    # Dal Zilio formulation
    # with extra smoothing for nonlinear terms
    # Ω[i,j] =  (1.0 - w) * Ω[i,j] + w * (condition * (log(V0 * _Vpij + (Ω_oldij -V0 * _Vpij)*exp(-Vpij*dt_L))) +
    #     !condition * log(Ω_oldij*(1-Vpij*dt_L)+V0*dt_L))

    # without extra smoothing
    Ω[i,j] =  (condition * (log(V0 * _Vpij + (Ω_oldij -V0 * _Vpij)*exp(-Vpij*dt_L))) +
        !condition * log(Ω_oldij*(1-Vpij*dt_L)+V0*dt_L))


    # gerya formulation
    # Ω[i,j] = Ω_old[i,j] + dt * (V0 * exp(-Ω_old[i,j]) - Vpij) / L

    return nothing
end


# Boundary conditions
@inline @inbounds @parallel_indices (iy) function dirichlet_x!(A::Data.Array, val_top::Data.Number, val_bottom::Data.Number) 
    A[1, iy]   = val_top
    A[end, iy] = val_bottom 
    return nothing
end

@inline @inbounds @parallel_indices (ix) function dirichlet_y!(A::Data.Array, val_left::Data.Number, val_right::Data.Number)
    A[ix, 1]   = val_left
    A[ix, end] = val_right
    return nothing
end

@inline @inbounds @parallel_indices (iy) function free_slip_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return nothing
end





# main function
@views function Stokes2D_vep()
    # numerics
    lx            = 10e3  # 10000.0  # [m] = 10 km
    ly            = 10e3  # 6000.0   # [m] = 10 km
    nx            = 160
    ny            = 160 

    nc            = nx, ny
    nv            = nx+1, ny+1
    X             = LinRange(0.0, lx, nx)       
    Y             = LinRange(0.0, ly, ny-1)
    @show h_index = ceil(Int, (ny - 1) / 2) + 1 # row index where the properties are stored for the fault
    h_indices_v   = [h_index, h_index+1]
    εnl           = 1.0e-15
    maxiter       = 4000
    nchk          = 10max(nx,ny)     # error checking frequency
    nviz          = 2                # visualization frequency
    Re            = 5*π
    _Re           = inv(Re)
    r             = 1.0
    CFL           = 0.95/sqrt(2)
    time_year     = 365.25*24*3600
    time_day      = 24*3600


    # preprocessing
    @show dx,dy   = lx/nx,ly/ny
    _dx, _dy      = inv(dx), inv(dy)
    max_lxy       = max(lx,ly)
    vpdτ          = CFL*min(dx,dy)
    xc,yc         = LinRange(-(lx-dx)/2,(lx-dx)/2,nx),LinRange(-(ly-dy)/2,(ly-dy)/2,ny)
    xv,yv         = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1)

    # phyics
    η0            = 1.0e23          # viscosity
    G0            = 30.0e9          # shear modulus

    # allocate arrays
    Pr_o    = @zeros(nx  ,ny  )
    λ       = @zeros(nx  ,ny  )
    τxx     = @zeros(nx  ,ny  )
    τyy     = @zeros(nx  ,ny  )
    τxx_o   = @zeros(nx  ,ny  )
    τyy_o   = @zeros(nx  ,ny  )
    τxyc_o  = @zeros(nx  ,ny  )
    τxy_o   = @zeros(nx+1,ny+1)
    ∇V      = @zeros(nx  ,ny  )
    Vy      = @zeros(nx  ,ny+1)
    Rp      = @zeros(nx  ,ny  )
    Rx      = @zeros(nx-1,ny  )
    Ry      = @zeros(nx  ,ny-1)
    τxy_cpu = zeros(nx+1,ny+1)
    τxyc_cpu= zeros(nx  ,ny  )
    τii_cpu = zeros(nx  ,ny  )
    τii_v   = @zeros(nx+1, ny+1)
    Vx_cpu  = zeros(nx+1,ny  )


    mask_tauxx_cpu = falses(nx, ny)
    mask_tauxy_cpu = falses(nx+1, ny+1)

    mask_tauxx_cpu[:, h_index]     .= true
    mask_tauxy_cpu[:, h_indices_v] .= true

    mask_tauxx = PTArray(mask_tauxx_cpu)
    mask_tauxy = PTArray(mask_tauxy_cpu)

    # DEBUG
    # plota = heatmap(xv,yc,Array(Vx)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
    # plotb = heatmap(xc,yv,Array(Vy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
    # display(plot(plota, plotb))
    # savefig("vizGPU_out/VxVy.png")

    Pt      = 40.0e6
    Pr_cpu  = fill(Pt, nx, ny)
    Pr      = PTArray(Pr_cpu)
    Prv_cpu  = fill(Pt, nx+1, ny+1)
    Prv     = PTArray(Prv_cpu)

    
    # added inertia
    ρ0      = 2700.0
    g       = 0.0
    ρ_cpu   = fill(ρ0, nx ,ny)
    ρ       = PTArray(ρ_cpu)
    ρg_cpu  = fill(ρ0*g, nx, ny)
    ρg      = PTArray(ρg_cpu)
    Vx_o    = @zeros(nx+1,ny  )
    Vy_o    = @zeros(nx  ,ny+1)
    
    # Parameters for rate-and-state dependent friction
    #           rate-s    rate-w
    a0        = [0.008    0.008]     # a-parameter of RSF
    b0        = [0.001    0.017]     # b-parameter of RSF    
    V0        = 1.0e-9               # characteristic slip rate for aseismic slip
    γ0        = 0.6                  # Reference Friction
    L         = 0.008 
    Vp        = @zeros(nx, ny)
    Vpv       = @zeros(nv...)
    Vp_o      = @zeros(nx, ny)
    γ_cpu     = fill(γ0, nx,ny)
    γ         = PTArray(γ_cpu)
    a_cpu     = fill(a0[1], nx, ny)
    b_cpu     = fill(b0[1], nx, ny)
    av_cpu    = fill(a0[1], nv...)
    bv_cpu    = fill(b0[1], nv...)

    # State variable from the preνious time step
    #           bulk     fault
    Ω0        = [40.0    -1.0]       
    Ω_o       = @zeros(nx,ny)        # for state variable ODE update
    Ω_cpu     = fill(Ω0[1],nx,ny)    # in bulk domain
    Ω_cpu[:, h_index] .= Ω0[2]       #  along the fault
    Ωv_o       = @zeros(nv...)       # for state variable ODE update
    Ωv_cpu     = fill(Ω0[1],nv...)   # in bulk domain
    Ωv_cpu[:, h_indices_v] .= Ω0[2]    # along the fault


    # setting up geometry
    if ASYMMETRIC
        # assign along fault [:, h_index] for rate-strengthing/weakening regions        
        #    0km   0.75km  1.5km                9km   9.5km  10m
        #    x0    x1     x2                   x3     x4    x5
        #    |*****|xxxxxx|                    |xxxxxx|*****|  
        #    -----------------------------------------------
        x0   = 0.0
        x1   = 0.075lx
        x2   = 0.15lx
        x3   = 0.9lx
        x4   = 0.95lx
        x5   = lx
    else
        #    0km   0.5km    1km                9km   9.5km  10m
        #    x0    x1     x2                   x3     x4    x5
        #    |*****|xxxxxx|                    |xxxxxx|*****|  
        #    -----------------------------------------------
        x0   = 0.0
        x1   = 0.05lx
        x2   = 0.1lx
        x3   = 0.9lx
        x4   = 0.95lx
        x5   = lx
    end




    yfaultv = tuple(yv[h_indices_v]...)
    for (i, xi) in enumerate(xv), (j, yi) in enumerate(yv)
        xi += lx/2 # need to augment the x-coordinates to match the geometry
        yi ∉ yfaultv && continue # early escape: skip to next iteration if we are not along the fault

        # assign domain value
        if x0 <= xi <= x1 || x4 <= xi <= x5
            av_cpu[i,j]  = a0[1]
            bv_cpu[i,j]  = b0[1]
        end

        # assign fault value
        if x2 <= xi <= x3
            av_cpu[i,j]  = a0[2]
            bv_cpu[i,j]  = b0[2]
        end

         # assign transition zone value (left)
         if x1 < xi < x2
            av_cpu[i, j] = a0[1] - (a0[1] - a0[2]) * ((xi - x1) / (x2 - x1))
            bv_cpu[i, j] = b0[1] - (b0[1] - b0[2]) * ((xi - x1) / (x2 - x1))
        end

        if x3 < xi < x4
            av_cpu[i, j] = a0[2] - (a0[2] - a0[1]) * ((xi - x3) / (x4 - x3))
            bv_cpu[i, j] = b0[2] - (b0[2] - b0[1]) * ((xi - x3) / (x4 - x3))
        end

    end

    for i in 1:1:nx
        for j in 1:1:ny

            # if along the fault
            if j == h_index
                # assign domain value
                if x0 <= X[i] <= x1 || x4 <= X[i] <= x5
                    a_cpu[i,j]  = a0[1]
                    b_cpu[i,j]  = b0[1]
                end

                # assign fault value
                if x2 <= X[i] <= x3
                    a_cpu[i,j]  = a0[2]
                    b_cpu[i,j]  = b0[2]
                end

                # assign transition zone value (left)
                if x1 < X[i] < x2
                    a_cpu[i, j]    = a0[1] - (a0[1] - a0[2]) * ((X[i] - x1) / (x2 - x1))
                    b_cpu[i, j]    = b0[1] - (b0[1] - b0[2]) * ((X[i] - x1) / (x2 - x1))
                end

                if x3 < X[i] < x4
                    a_cpu[i, j]  = a0[2] - (a0[2] - a0[1]) * ((X[i] - x3) / (x4 - x3))
                    b_cpu[i, j]  = b0[2] - (b0[2] - b0[1]) * ((X[i] - x3) / (x4 - x3))
                end

            end

        end
    end

    a = PTArray(a_cpu)
    b = PTArray(b_cpu)
    Ω = PTArray(Ω_cpu)
    a_v = PTArray(av_cpu)
    b_v = PTArray(bv_cpu)
    Ω_v = PTArray(Ωv_cpu)
    λ_v = @zeros(nv...)
    Pr_v = @zeros(nv...)

    
    # initial velocity using gradient change of loading slip rate
    VL      = 2.0e-9      # loading rate
    Vx_cpu  = [(2.0*(y+ly/2.0)/ly-1)*VL for x in xv, y in (yc)]
    Vy_cpu  = [0.0 for x in xc, y in yv]

    # prescribe loading rate B.C. see explaination in pt b.c. update for update details
    Vx_cpu[:, 1]    .= -VL
    Vx_cpu[:, end]  .= +VL

    Vx = PTArray(Vx_cpu)
    Vy = PTArray(Vy_cpu)

    
    # DEBUG
    # plota = heatmap(xv,yc,Array(Vx)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
    # plotb = heatmap(xc,yv,Array(Vy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
    # display(plot(plota, plotb))
    # savefig("vizGPU_out/VxVy.png")

    # FIXME: intialize only shear stress, be careful when setting this to zero, results in division by zero for Vp computation
    @. τxy_cpu   = 0.55*Pt
    @. τxyc_cpu  = 0.55*Pt
    τxy  = PTArray(τxy_cpu)
    τxyc = PTArray(τxyc_cpu)

    # precomputation of τII as slip rate requires this
    @. τii_cpu   = sqrt(#=0.5*(τxx^2 + τyy^2)=# +τxyc_cpu*τxyc_cpu)  # τxx, τyy not initialized
    τii          = PTArray(τii_cpu)
    @parallel center2vertex!(τii_v, τii)
    @show max_tauii = maximum(τii_v)

        
    # PT damping parameters arrays allocation
    Exy     = @zeros(nx+1,ny+1)
    η_ve_τ  = @zeros(nx  ,ny  )
    η_ve_τv = @zeros(nx+1,ny+1)
    η_ve    = @zeros(nx  ,ny  )
    η_vem   = @zeros(nx  ,ny  )
    η_vev   = @zeros(nx+1,ny+1)
    η_vevm  = @zeros(nx+1,ny+1)
    dτ_ρ    = @zeros(nx  ,ny  )
    dτ_ρv   = @zeros(nx+1,ny+1)
    Gdτ     = @zeros(nx  ,ny  )
    Gdτv    = @zeros(nx+1,ny+1)

    η_cpu  = fill(η0,nx,ny); ηv_cpu = fill(η0,nx+1,ny+1)
        η  = PTArray(η_cpu); ηv     = PTArray(ηv_cpu)
    
    G_cpu  = fill(G0,nx,ny); Gv_cpu = fill(G0,nx+1,ny+1)
        G  = PTArray(G_cpu); Gv     = PTArray(Gv_cpu)

    η_e   = @zeros(nx, ny)
    η_ev  = @zeros(nx+1, ny+1)

    # total simulation time
    total_year   = 20
    
    # action
    t = 0.0; it = 0; dt = 1.0

    # record evolution of time step size, pt iteration number and slip rate, stress
    evo_t_year = Float64[]; evo_dt = Float64[]; evo_ptno = Float64[]; evo_Vp = Float64[]; evo_τxx = Float64[]


    # for it = 1:300
    while t/time_year ≤ total_year

        it += 1

        #==========================================#
        # using adaptive time stepping scheme
        # 1). stability-induced timestep constraint
        # 2). using empirically found minimal timestep threshold
        @show dt = min(1.9/(maximum(Vp)*50),0.1*time_year) 

        # precompute reprocical due to performance
        _dt     = inv(dt)

        # assign old values from previous physical timestep
        @parallel assign!(τxx_o, τxx, τyy_o, τyy, τxy_o, τxy, τxyc_o, τxyc, Pr_o, Pr, Vx_o, Vx, Vy_o, Vy, Ω_o, Ω)
        Ωv_o = deepcopy(Ω_v)
        Vp_o = deepcopy(Vp)
        

        # compute pt parameters
        @parallel compute_params1!(η_e, η_ev, G, Gv, dt)
        @parallel compute_params2!(η_ve, η_vev, η, ηv, η_e, η_ev)
        @parallel compute_params3!(η_vem, η_vevm, η_ve, η_vev)
        bc2!(η_vem)
        bc2!(η_vevm)
    
        @parallel compute_params4!(dτ_ρ, dτ_ρv, vpdτ, max_lxy, _Re, η_vem, η_vevm)
        @parallel compute_params5!(Gdτ, Gdτv, vpdτ, dτ_ρ, dτ_ρv, r)
        @parallel compute_params6!(η_ve_τ, η_ve_τv, η, ηv, η_e, η_ev, Gdτ, Gdτv)

        # compute slip rate, state parameter and the plastic multiplier
        @parallel compute_slip_rate!(Vp, τii, a, b, Ω, γ0, Pt, V0, mask_tauxx)
        @parallel compute_slip_rate!(Vpv, τii_v, a_v, b_v, Ω_v, γ0, Pt, V0, mask_tauxy)

        @parallel (1:nx, 1:ny) compute_state_parameter2!(Ω, Ω_o, Vp, dt, V0, L, 0.1)
        @parallel (1:nx, 1:ny) compute_state_parameter2!(Ω_v, Ωv_o, Vpv, dt, V0, L, 0.1)

        @parallel compute_plastic_multiplier2!(λ, Vp, _dx, dt, η_ve_τ)
        @parallel compute_plastic_multiplier2!(λ_v, Vpv, _dx, dt, η_ve_τv)

        #==========================================#

        
        err  = 2εnl; iter = 0; niter = 0
        while err > εnl && iter < maxiter
            # performance
            if (iter==11)  global wtime0 = Base.time()  end

            #==========================================#
            # pressure update
            @parallel compute_∇!(∇V, Vx, Vy, _dx, _dy)
            @parallel compute_residual_mass_law!(Rp, ∇V)
            @parallel compute_pressure!(Pr, Gdτ, Rp, r)
            @parallel (1:nx) free_slip_y!(Pr)
            @parallel (1:ny) free_slip_x!(Pr)

            #==========================================#            
            # visco-elastic stress update containing PT terms
            @parallel compute_strain_rate!(Exy, Vx, Vy, _dx, _dy)     # avoid race condition
            @parallel compute_ve_stress!(Exy, τxx, τyy, τxyc, τxx_o, τyy_o, τxyc_o, Vx, Vy, η_ve_τ, η_e, Gdτ, _dx, _dy)
            @parallel (1:nx+1) free_slip_y!(Exy)
            @parallel (1:ny+1) free_slip_x!(Exy)

            # preventing race condition!
            @parallel compute_tensor!(τxy, η_ve_τv, Exy, τxy_o, η_ev, Gdτv)                        
        
            # slip rate update
            @parallel (1:nx, 1:ny) compute_plastic_correction2!(λ, τxx, τyy, τxyc, τii, η_ve_τ)
            @parallel compute_plastic_correction_xy!(λ_v, τxy, τii, η_ve_τv, mask_tauxy)
            @parallel (1:nx+1) free_slip_y!(τxy)
            @parallel (1:ny+1) free_slip_x!(τxy)
            @parallel (1:nx) free_slip_y!(τxyc)
            @parallel (1:ny) free_slip_x!(τxyc)
            @parallel (1:nx) free_slip_y!(τxx)
            @parallel (1:ny) free_slip_x!(τxx)
            @parallel (1:nx) free_slip_y!(τyy)
            @parallel (1:ny) free_slip_x!(τyy)


            
            # compute second invariant
            @parallel compute_second_invariant!(τii, τxx, τyy, τxyc)
            @parallel center2vertex!(τii_v, τii)
            @parallel (1:nx+1) free_slip_y!(τii_v)
            @parallel (1:ny+1) free_slip_x!(τii_v)

            #==========================================#
            # compute velocity
            @parallel compute_residual_momentum_law!(Rx, Ry, τxx, τyy, τxy, Pr, ρ, ρg, Vx, Vx_o, Vy, Vy_o, _dx, _dy, _dt)
            @parallel (1:nx)   free_slip_y!(Ry)
            @parallel (1:ny-1) free_slip_x!(Ry)
            @parallel (1:ny)   free_slip_x!(Rx)            
            @parallel (1:nx-1) free_slip_y!(Rx)            

            @parallel compute_velocity!(Vx, Vy, Rx, Ry, dτ_ρ)
           

            #==========================================#
            # Boundary conditions
            @parallel (1:nx+1) dirichlet_y!(Vx, -VL, VL)
            @parallel (1:nx)   dirichlet_y!(Vy, 0.0, 0.0)
            @parallel (1:ny)   free_slip_x!(Vx) # Open boundary condition (Luca's paper)
            # @parallel (1:ny+1) free_slip_x!(Vy) # Free-slip boundary condition (Luca's paper)
            @parallel (1:ny+1)   dirichlet_x!(Vy, 0.0, 0.0) # Prof. Gerya
            

            # NOTE: our properties are stored in the following format
            #     for example, dirichlet_y! will be apply to the y-axis
            #     from our perspective of seeing, which corresponds to
            #     boundaries parallel to where "nx" stands in the following figure
            #
            # sanity check: apply kernel dirichlet_y!(Vx, val_left, val_right) 
            #      with Vx of size (nx+1,ny)
            #      updates       - A[Vx, 1]   = val_left     for ix in [1, nx+1]
            #                    - A[Vx, end] = val_right

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

            if iter % nchk == 0
                norm_Rx = norm(Rx)/sqrt(length(Rx)); norm_Ry = norm(Ry)/sqrt(length(Ry)); norm_∇V = norm(∇V)/sqrt(length(∇V))
                err = maximum([norm_Rx, norm_Ry, norm_∇V])
                @printf("it = %d, iter = %d, err = %1.5e norm[Rx=%1.5e, Ry=%1.5e, ∇V=%1.5e] \n", it, iter, err, norm_Rx, norm_Ry, norm_∇V)
            end
            iter += 1; niter += 1
        end


        @show max_Vp = maximum(Vp[:, h_index])
        @show max_λ  = maximum(λ)
        @show min_λ  = minimum(λ)


        # store fluid pressure for wanted time points
        # if STORE_DATA && mod(it, nviz) == 0
        #     save("earthquake_cycles/Vp_fault" * string(it) * ".jld", "data", Array(Vp[:, h_index])')   # store the fluid pressure along the fault for fluid injection benchmark
        #     if it ≥ 250
        #         save("earthquake_cycles/Vx" * string(it) * ".jld", "data", Array(Vx)')
        #         save("earthquake_cycles/Vy" * string(it) * ".jld", "data", Array(Vy)')
        #     end        
        # end
        
        # advance in physical time
        t += dt; push!(evo_τxx, maximum(τxx)); push!(evo_dt, dt); push!(evo_ptno, iter)
        

        
        # store evolution of physical properties wrt time
        push!(evo_t_year, t/time_year); push!(evo_Vp, max_Vp);

        # PERFORMANCE
        wtime    = Base.time()-wtime0
        A_eff    = (27*2 + 7)/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(niter-10)                          # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                            # Effective memory throughput [GB/s]
        @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))


        # VISUALIZATION
        if mod(it, nviz) == 0
            p1 = heatmap(xc,yc,log.(Array(Vp)/V0)',c=:balance, aspect_ratio=1, xticks=nothing, yticks=nothing, xlabel="", ylabel="", xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\log_{10}(V_\mathrm{p}/V_0)$")
            p2 = plot(X./1000, Array(τxyc.*1e-6)[:, h_index] , legend=false, xticks=nothing, xlabel="", xlims=(0.0,lx/1000), title="Shear stress "* L"\sigma_\mathrm{xy}" *" on the fault [MPa]", framestyle=:box)
            p3 = heatmap(xc,yc,Array(Pr.*1e-6)',c=:balance, aspect_ratio=1, xticks=nothing, yticks=nothing, xlabel="", ylabel="", xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Pressure [MPa]", framestyle=:box)

            p4 = heatmap(xv,yc, #=1e9*=#Array(Vx)',c=:balance, aspect_ratio=1, #=colorbar_title="1e-9", colorbar_titlefontrotation=180, =# xticks=nothing, yticks=nothing, xlabel="", ylabel="", xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
            p5 = plot(X./1000, Array(Ω)[:, h_index] , legend=false, xlabel="",  xticks=nothing, xlims=(0.0,lx/1000), title="State variable "* L"\Omega" *" on the fault", framestyle=:box, markersize=3)
            p6 = heatmap(xc,yc,Array(τxx.*1e-6)',c=:balance, aspect_ratio=1, xticks=nothing, yticks=nothing, xlabel="", ylabel="", xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Normal stress tauxx [MPa]", framestyle=:box)

            p7 = heatmap(xc,yv,Array(Vy)',c=:balance,aspect_ratio=1,#=colorbar_title="1e-12", fontrotation=180, =#xticks=nothing, yticks=nothing,  xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
            p8 = plot(X./1000, ((Array((Vp))[:, h_index])) , legend=false, xlabel="Distance [km]", xlims=(0.0,lx/1000), yaxis=:log, title="Slip rate "* L"V_\mathrm{p}" *" on the fault [m/s]", framestyle=:box, markersize=3)
            p9 = heatmap(xv,yv,Array(τxy.*1e-6)',c=:balance, aspect_ratio=1, xticks=nothing, yticks=nothing, xlabel="", ylabel="", xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Shear stress tauxy [MPa]", framestyle=:box)

            p10 = plot(evo_t_year, evo_Vp; xlims=(0.0, total_year), ylims=(1e-12, 1.0e1), yaxis=:log, yticks =[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0], label="", color= :dodgerblue, framestyle= :box, linestyle= :solid, 
            seriesstyle= :path, title="Seismo-Mechanical Earthquake Cycles (t = " * string(@sprintf("%.7f", t/time_year)) * " year )", xlabel = "", ylabel="Maximum Slip Rate [m/s]" )
            p11 = plot(evo_t_year, evo_ptno; xlims=(0.0, total_year), ylims=(0, 1.1*maxiter), label="", color= :orange, framestyle= :box, linestyle= :solid, 
            seriesstyle= :path, title="", 
            xlabel = "", ylabel="No. PT iterations" )
            p12 = plot(evo_t_year, evo_dt; xlims=(0.0, total_year), ylims=(1.0e-6, 1.0e8), yaxis=:log, yticks =[1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8], label="", color= :green, framestyle= :box, linestyle= :solid, 
            seriesstyle= :path, title="", 
            xlabel = "Time [year]", ylabel="Time Step [s]" )

            l = @layout [a b c; d e f; g h i; j; k; l]

            display(plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12; layout=l)); frame(anim)    

        end

    end


    gif(anim, "vizGPU_out/rsf_strikeslip_fault.gif", fps = 10)


    # DEBUG
    @show evo_τxx
    @show evo_Vp

    return
end

# action
Stokes2D_vep() 
