# Physics: Incompressible stokes equation with VEP + rate- and state-dependent friction
#          - momentum equation with inertia effects included

using HydroMech

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)


# NOTE: despite of using the package we initialize here again because 
# we need to use the type Data.Array, Data.Number for argument types
const USE_GPU    = true  # Use GPU? If this is set false, then no GPU needs tio be available
const STORE_DATA = true

@static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
else
        @init_parallel_stencil(Threads, Float64, 2)
end

@static if STORE_DATA
    using JLD
end


using Plots, Plots.Measures, LinearAlgebra,Printf
using LaTeXStrings
# helper functions
@views     av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views  av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views  av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views maxloc(A) = max.(A[1:end-2,1:end-2],A[1:end-2,2:end-1],A[1:end-2,3:end],
                        A[2:end-1,1:end-2],A[2:end-1,2:end-1],A[2:end-1,3:end],
                        A[3:end  ,1:end-2],A[3:end  ,2:end-1],A[3:end  ,3:end])
@views   bc2!(A) = begin A[1,:] = A[2,:]; A[end,:] = A[end-1,:]; A[:,1] = A[:,2]; A[:,end] = A[:,end-1] end


# VISUALIZATION
if DO_VIZ
    # default(size=(3500,1700),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=7mm)
    default(size=(1500,1800),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=7mm)
    scalefontsizes(); scalefontsizes(1.35)

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


@inbounds @parallel function compute_ve_stress!(Exy, τxx, τyy, τxy, τxyc, τxx_o, τyy_o, τxy_o, τxyc_o, Vx, Vy, η_ve_τ, η_ve_τv, η_e, η_ev, Gdτ, Gdτv, _dx, _dy)

    @inn(Exy)  = 0.5 * (@d_yi(Vx) * _dy + @d_xi(Vy) * _dx)
    @all(τxx)  = 2.0*@all(η_ve_τ)*(@d_xa(Vx)* _dx + 0.5*@all(τxx_o)/@all(η_e) + 0.5*@all(τxx)/@all(Gdτ))
    @all(τyy)  = 2.0*@all(η_ve_τ)*(@d_ya(Vy)* _dy + 0.5*@all(τyy_o)/@all(η_e) + 0.5*@all(τyy)/@all(Gdτ))
    @all(τxyc) = 2.0*@all(η_ve_τ)*(@av(Exy) + 0.5*@all(τxyc_o)/@all(η_e) + 0.5*@all(τxyc)/@all(Gdτ))

    return nothing
end 


@inbounds @parallel function compute_tensor!(τxy, η_ve_τv, Exy, τxy_o, η_ev, Gdτv)

    @all(τxy) = 2.0*@all(η_ve_τv)*(@all(Exy) + 0.5*@all(τxy_o)/@all(η_ev) + 0.5*@all(τxy)/@all(Gdτv))

    return nothing
end


@inbounds @parallel function compute_slip_rate!(Vp, τii, a, b, Ω, γ, Pt, V0)

    # if no smoothing was used computation would have been:
    # @all(Vp) = 2.0*V0*sinh(@all(τii)/@all(a)/Pt)/exp((@all(b)*@all(Ω) + @all(γ))/@all(a))

    # we employ a smoothing technique, chosen relaxation parameter θ = 0.5 here
    @all(Vp) = (1.0 - 0.01) * @all(Vp) + 0.01*(2.0*V0*sinh(@all(τii)/@all(a)/Pt)/exp((@all(b)*@all(Ω) + @all(γ))/@all(a)))

    return nothing
end



@inbounds @parallel function compute_plastic_multiplier!(λ, Vp, _dx)

    # estimation of plastic multiplier (Casper)
    @all(λ)  = 0.5*@all(Vp)*_dx

    return nothing
end

@inbounds @parallel function compute_plastic_correction!(λ, Vp, τxx, τyy, τxyc, τii, η_ve_τ, _dx, mask_tauxx)


    @all(τxx) = @all(τxx) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τxx)/@all(τii)) * @all(mask_tauxx)
    @all(τyy) = @all(τyy) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τyy)/@all(τii)) * @all(mask_tauxx)
    @all(τxyc) = @all(τxyc) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τxyc)/@all(τii))  * @all(mask_tauxx)
    # @inn_x(τxy) = @inn_x(τxy) - 2.0*@inn_x(η_ve_τv)*(#=0.5 *@av_xi(λ)*=#@inn_x(τxyc)/@(τii))* @all(mask_tauxy)

    return nothing
    
end


@parallel function center2vertex!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end




@inbounds @parallel function compute_second_invariant!(τii, τxx, τyy, τxyc)

    @all(τii)   = sqrt(0.5*(@all(τxx)^2 + @all(τyy)^2) + @all(τxyc)*@all(τxyc))

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



@inbounds @parallel function compute_state_parameter!(Ω, Vp, dt, V0, L)

    # @all(Ω) = @all(Ω) + dt * (V0 * exp(-@all(Ω)) - @all(Vp)) / L
    @all(Ω) = (1.0 - 0.01) * @all(Ω) + 0.01*(@all(Ω) + dt * (V0 * exp(-@all(Ω)) - @all(Vp)) / L)

    return nothing
end



# main function
@views function Stokes2D_vep()
    # numerics
    lx            = 10e3#10000.0  # [m] = 10 km
    ly            = 10e3#6000.0   # [m] = 6 km
    # nx            = 64*2 #167
    # ny            = 64*2 #100
    nx            = 300 #167
    ny            = 300 #100

    # not squared
    # lx            = 10000.0  # [m] = 10 km
    # ly            = 6000.0   # [m] = 6 km
    # nx            = 167
    # ny            = 100

    X             = LinRange(0.0, lx, nx)       
    Y             = LinRange(0.0, ly, ny-1)
    @show h_index = ceil(Int, (ny - 1) / 2) + 1 # row index where the properties are stored for the fault

    nt            = 300                # small number for the first-time compilation run, set to other values for longer run
    εnl           = 1.0e-8
    maxiter       = 10000
    nchk          = 10max(nx,ny)     # error checking frequency
    nviz          = 1                # visualization frequency
    Re            = 5*π
    r             = 1
    CFL           = 0.95/sqrt(2)
    time_year     = 365.25*24*3600


    # preprocessing
    @show dx,dy   = lx/nx,ly/ny
    _dx, _dy      = inv(dx), inv(dy)
    max_lxy       = max(lx,ly)
    vpdτ          = CFL*min(dx,dy)
    xc,yc         = LinRange(-(lx-dx)/2,(lx-dx)/2,nx),LinRange(-(ly-dy)/2,(ly-dy)/2,ny)
    xv,yv         = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1)


    # phyics
    η0      = 1.0e23          # viscosity
    G0      = 30.0e9          # shear modulus

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
    Vx_cpu  = zeros(nx+1,ny  )

    mask_tauxx_cpu  = zeros(nx, ny)
    mask_tauxy_cpu  = zeros(nx+1, ny+1)

    @. mask_tauxx_cpu[:, h_index]  = 1.0
    # @. mask_tauxy_cpu[:, h_index]  = 1.0

    mask_tauxx  = PTArray(mask_tauxx_cpu)
    mask_tauxy  = PTArray(mask_tauxy_cpu)


    # DEBUG
    # plota = heatmap(xv,yc,Array(Vx)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
    # plotb = heatmap(xc,yv,Array(Vy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
    # display(plot(plota, plotb))
    # savefig("vizGPU_out/VxVy.png")

    Pt      = 40.0e6
    Pr_cpu  = fill(Pt, nx, ny)
    Pr      = PTArray(Pr_cpu)
    
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
    V0        = 4.0e-9               # characteristic slip rate for aseismic slip
    γ0        = 0.6                  # Reference Friction
    L         = 0.008 
    Vp        = @zeros(nx, ny)
    γ_cpu     = fill(γ0, nx,ny)
    γ         = PTArray(γ_cpu)
    a_cpu     = fill(a0[1],nx,ny)
    b_cpu     = fill(b0[1],nx,ny)

    # State variable from the preνious time step
    #            bulk    fault
    Ω0        = [40.0    -1.0]       
    Ω_o       = @zeros(nx,ny)   # for state variable ODE update
    Ω_cpu     = fill(Ω0[1],nx,ny)    # in bulk domain
    @. Ω_cpu[:, h_index] = Ω0[2]     # along the fault
    
    # setting up geometry
    # assign along fault [:, h_index] for rate-strengthing/weakening regions        
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

    a   = PTArray(a_cpu)
    b   = PTArray(b_cpu)
    Ω   = PTArray(Ω_cpu)

 
    # initial velocity using gradient change of loading slip rate
    VL      = 1.0e-9      # loading rate
    Vx_cpu  = [(2.0*(y+ly/2.0)/ly-1)*VL for x in xv, y in (yc)]
    Vy_cpu  = [0.0 for x in xc, y in yv]

    # prescribe loading rate B.C. see explaination in pt b.c. update for update details
    @. Vx_cpu[:, 1]    = -VL
    @. Vx_cpu[:, end]  = +VL

    Vx      = PTArray(Vx_cpu)
    Vy      = PTArray(Vy_cpu)

    
    # DEBUG
    # plota = heatmap(xv,yc,Array(Vx)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
    # plotb = heatmap(xc,yv,Array(Vy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
    # display(plot(plota, plotb))
    # savefig("vizGPU_out/VxVy.png")

    # FIXME: intialize only shear stress, be careful when setting this to zero, results in division by zero for Vp computation
    @. τxy_cpu   = 0.5*Pt
    @. τxyc_cpu  = 0.5*Pt

    τxy  = PTArray(τxy_cpu)
    τxyc = PTArray(τxyc_cpu)

    # precomputation of τII as slip rate requires this
    @. τii_cpu   = sqrt(#=0.5*(τxx^2 + τyy^2)=# +τxyc_cpu*τxyc_cpu)  # τxx, τyy not initialized
    τii          = PTArray(τii_cpu)

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

    
    # action
    t = 0.0; evo_t = Float64[]; evo_τxx = Float64[]

    # record evolution of time step size and slip rate
    evo_t_year = Float64[]; evo_dt = Float64[]; evo_Vp = Float64[]; evo_ptno = Float64[]

    for it = 1:nt

        # assign old values from previous physical timestep
        @parallel assign!(τxx_o, τxx, τyy_o, τyy, τxy_o, τxy, τxyc_o, τxyc, Pr_o, Pr, Vx_o, Vx, Vy_o, Vy, Ω_o, Ω)

        #==========================================#
        # FIXME: will use parallelstencil kernel after for this section        
        # using adaptive time stepping scheme
        # 1). stability-induced timestep constraint
        # 2). using empirically found minimal timestep threshold
        @show dt = min(1.9*L*0.5/(maximum(Vp[:,h_index])*100), 0.01time_year)
        
        η_e   = G.*dt; η_ev = Gv.*dt
        @. η_ve  = 1.0/(1.0/η  + 1.0/η_e)
        @. η_vev = 1.0/(1.0/ηv + 1.0/η_ev)
        
        # compute pt parameters
        η_vem[2:end-1,2:end-1]  .= maxloc(η_ve) ; bc2!(η_vem)
        η_vevm[2:end-1,2:end-1] .= maxloc(η_vev); bc2!(η_vevm)
        
        # for velocity update
        @. dτ_ρ    = vpdτ*max_lxy/Re/η_vem
        @. dτ_ρv   = vpdτ*max_lxy/Re/η_vevm
    
        # for pressure and stress update
        @. Gdτ     = vpdτ^2/dτ_ρ/(r+2.0)
        @. Gdτv    = vpdτ^2/dτ_ρv/(r+2.0)
    
    
        # for stress update
        @. η_ve_τ  = 1.0/(1.0/η + 1.0/η_e + 1.0/Gdτ)
        @. η_ve_τv = 1.0/(1.0/ηv + 1.0/η_ev + 1.0/Gdτv)        

        # precompute reprocical due to performance
        _dt     = inv(dt)
        #==========================================#


        err  = 2εnl; iter = 0
        while err > εnl && iter < maxiter
            #==========================================#
            # pressure update
            @parallel compute_∇!(∇V, Vx, Vy, _dx, _dy)
            @parallel compute_residual_mass_law!(Rp, ∇V)
            @parallel compute_pressure!(Pr, Gdτ, Rp, r)

            #==========================================#            
            # visco-elastic stress update containing PT terms
            @parallel compute_ve_stress!(Exy, τxx, τyy, τxy, τxyc, τxx_o, τyy_o, τxy_o, τxyc_o, Vx, Vy, η_ve_τ, η_ve_τv, η_e, η_ev, Gdτ, Gdτv, _dx, _dy)

            # preventing race condition!
            @parallel compute_tensor!(τxy, η_ve_τv, Exy, τxy_o, η_ev, Gdτv)                        
            
            # slip rate update
            # FIXME: using constant pressure for slip rate computation in compressible case (Casper)
            @parallel compute_slip_rate!(Vp, τii, a, b, Ω, γ, Pt, V0)

            # plastic correction update
            @parallel compute_plastic_multiplier!(λ, Vp, _dx)
            @parallel compute_plastic_correction!(λ, Vp, τxx, τyy, τxyc, τii, η_ve_τ, _dx, mask_tauxx)
            @parallel center2vertex!(τxy, τxyc)
            
            # compute second invariant
            @parallel compute_second_invariant!(τii, τxx, τyy, τxyc)

            #==========================================#
            # compute velocity
            @parallel compute_residual_momentum_law!(Rx, Ry, τxx, τyy, τxy, Pr, ρ, ρg, Vx, Vx_o, Vy, Vy_o, _dx, _dy, _dt)

            @parallel (1:nx)   free_slip_y!(Ry)
            @parallel (1:ny-1) free_slip_x!(Ry)
            @parallel (1:ny)   free_slip_x!(Rx)            
            @parallel (1:nx-1) free_slip_y!(Rx)            

            @parallel compute_velocity!(Vx, Vy, Rx, Ry, dτ_ρ)

            # update state parameter, within the PT loop as in Räss et al. (2019) porosity wave benchmark
            @parallel compute_state_parameter!(Ω, Vp, dt, V0, L)

            #==========================================#
            # Boundary conditions
            @parallel (1:nx+1) dirichlet_y!(Vx, -VL, VL)
            # @parallel (1:nx)   dirichlet_y!(Vy, 0.0, 0.0) # IGNORE THIS ONE FROM TARAS BOOK
            @parallel (1:ny)   free_slip_x!(Vx) # Open boundary condition (Luca's paper)
            @parallel (1:ny+1) free_slip_x!(Vy) # Free-slip boundary condition (Luca's paper)
            

            # alternatively using Dirichlet for Vy as in intro to numerical modelling
            # though B.C. of Vy shouldn't make much difference here
            # @parallel (1:ny+1) dirichlet_x!(Vy, 0.0, 0.0)       
            
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
                @printf("it = %d, iter = %d, err = %1.5e norm[Rx=%1.15e, Ry=%1.15e, ∇V=%1.5e] \n", it, iter, err, norm_Rx, norm_Ry, norm_∇V)
            end
            iter += 1
        end


        @show max_Vp          = maximum(Vp[:, h_index])

        # store fluid pressure for wanted time points
        if STORE_DATA && mod(it, nviz) == 0
            save("earthquake_cycles/max_Vp_fault" * string(it) * ".jld", "data", Array(Vp[:, h_index])')   # store the fluid pressure along the fault for fluid injection benchmark
        end

        
        # advance in physical time
        t += dt; push!(evo_t, t); push!(evo_τxx, maximum(τxx)); push!(evo_dt, dt); push!(evo_ptno, iter)
        
        # store evolution of physical properties wrt time
        push!(evo_t_year, t/time_year); push!(evo_Vp, max_Vp);
        
        if mod(it, nviz) == 0
            # visualization
            p1 = heatmap(xc,yc,log.(Array(Vp)/V0)',c=:balance, aspect_ratio=1, xticks=nothing, yticks=nothing, xlabel="", ylabel="", xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\log_{10}(\frac{V_\mathrm{pl}}{V_0})$")
            p2 = plot(X./1000, Array(τxyc)[:, h_index] , legend=false, xticks=nothing, xlabel="", xlims=(0.0,lx/1000), title="Shear stress "* L"\sigma_\mathrm{xy}" *" on the fault", framestyle=:box, markersize=3)
            p3 = heatmap(xv,yc, 1e9*Array(Vx)',c=:balance, aspect_ratio=1, colorbar_title="1e-9", colorbar_titlefontrotation=180,  xticks=nothing, yticks=nothing, xlabel="", ylabel="", xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
            p4 = plot(X./1000, Array(Ω)[:, h_index] , legend=false, xlabel="",  xticks=nothing, xlims=(0.0,lx/1000), title="State variable "* L"\Omega" *" on the fault", framestyle=:box, markersize=3)
            p5 = heatmap(xc,yv,Array(Vy)',c=:balance,aspect_ratio=1,colorbar_titlefontrotation=180,xticks=nothing, yticks=nothing,  xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
            p6 = plot(X./1000, Array(Vp)[:, h_index] , legend=false, xlabel="Distance [km]", xlims=(0.0,lx/1000), yaxis=:log, title="Slip rate "* L"V_\mathrm{p}" *" on the fault", framestyle=:box, markersize=3)
            p7 = plot(evo_t_year, evo_Vp; xlims=(0.0, 50), ylims=(minimum(evo_Vp), 1.0e2), yaxis=:log, yticks =[1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2], label="", color= :dodgerblue, framestyle= :box, linestyle= :solid, 
            seriesstyle= :path, title="Seismo-Mechanical Earthquake Cycles (t = " * string(@sprintf("%.3f", t/time_year)) * " year )", xlabel = "", ylabel="Maximum Slip Rate [m/s]" )
            p8 = plot(evo_t_year, evo_ptno; xlims=(0.0, 50), ylims=(0, 10000), label="", color= :orange, framestyle= :box, linestyle= :solid, 
            seriesstyle= :path, title="", 
            xlabel = "Time [year]", ylabel="No. PT iterations" )
            p9 = plot(evo_t_year, evo_dt; xlims=(0.0, 50), ylims=(1.0e-3, 1.0e7), yaxis=:log, yticks =[1e-3, 1e-2, 1.0, 1e2, 1e4, 1e6], label="", color= :orange, framestyle= :box, linestyle= :solid, 
            seriesstyle= :path, title="", 
            xlabel = "Time [year]", ylabel="Time Step [s]" )

            l = @layout [a b; c d; e f; g; h; i]

            display(plot(p1,p2,p3,p4,p5,p6,p7,p8,p9; layout=l)); frame(anim)
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