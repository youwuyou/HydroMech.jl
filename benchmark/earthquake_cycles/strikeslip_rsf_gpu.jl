# Script derived from Stokes2D_vep_reg_IU.jl
# Physics: Incompressible stokes equation with VEP
#          - momentum equation with inertia effects included
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
    default(size=(3200,1700),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=7mm)
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

@inbounds @parallel function compute_residual_mass_law_inertia!(Rp, ∇V, Pr, Pr_o, _dt)
    @all(Rp)    = -@all(∇V) - (@all(Pr) - @all(Pr_o)) * _dt
    return nothing
end

# pressure
# Rp    .= -∇V - (Pr - Pr_o)/dt


@inbounds @parallel function compute_pressure!(Pr, Gdτ, Rp, r)
    @all(Pr) = @all(Pr) + r*@all(Gdτ)*@all(Rp)
    return nothing
end


@inbounds @parallel function compute_ve_stress!(Exy, τxx, τyy, τxy, τxyc, τxx_o, τyy_o, τxy_o, τxyc_o, Vx, Vy, η_ve_τ, η_ve_τv, η_e, η_ev, Gdτ, Gdτv, _dx, _dy)

    @inn(Exy) = 0.5 * (@d_yi(Vx) * _dy + @d_xi(Vy) * _dx)
    @all(τxx) = 2.0*@all(η_ve_τ)*(@d_xa(Vx)* _dx + 0.5*@all(τxx_o)/@all(η_e) + 0.5*@all(τxx)/@all(Gdτ))
    @all(τyy) = 2.0*@all(η_ve_τ)*(@d_ya(Vy)* _dy + 0.5*@all(τyy_o)/@all(η_e) + 0.5*@all(τyy)/@all(Gdτ))
    @all(τxyc) = 2.0*@all(η_ve_τ)*(@av(Exy) + 0.5*@all(τxyc_o)/@all(η_e) + 0.5*@all(τxyc)/@all(Gdτ))

    return nothing
end 


@inbounds @parallel function compute_tensor!(τxy, η_ve_τv, Exy, τxy_o, η_ev, Gdτv)

    @all(τxy) = 2.0*@all(η_ve_τv)*(@all(Exy) + 0.5*@all(τxy_o)/@all(η_ev) + 0.5*@all(τxy)/@all(Gdτv))

    return nothing
end


@inbounds @parallel function compute_slip_rate!(Vp, τii, a, b, Ω, γ, Pt, V0)
    @all(Vp) = 2.0*V0*sinh(@all(τii)/@all(a)/Pt)/exp((@all(b)*@all(Ω) + @all(γ))/@all(a))
    # @all(Vp) = 2.0*V0*sinh(@all(τii)/@all(a)/@all(Pr))/exp((@all(b)*@all(Ω) + @all(γ))/@all(a))
    
    return nothing
end


@inbounds @parallel function compute_plastic_correction!(λ, Vp, τxx, τyy, τxy, τxyc, τii, η_ve_τ, η_ve_τv, _dx)
            
    @all(λ)  = 0.5*@all(Vp)*_dx

    @all(τxx) = @all(τxx) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τxx)/@all(τii))
    @all(τyy) = @all(τyy) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τyy)/@all(τii))
    @all(τxyc) = @all(τxyc) - 2.0*@all(η_ve_τ)*(@all(λ)*0.5*@all(τxyc)/@all(τii))
    @inn_x(τxy) = @inn_x(τxy) - 2.0*@inn_x(η_ve_τv)*(0.5 * @all(λ)*@all(τxyc)/@all(τii))

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

    @all(Ω) = @all(Ω) + dt * (V0 * exp(-@all(Ω)) - @all(Vp)) / L

    return nothing
end



# main function
@views function Stokes2D_vep()
    # numerics
    lx       = 40000.0  # [m] 40 km
    ly       = 20000.0  # [m] 20 km

    nx       = 40
    ny       = 20


    # nx       = 80
    # ny       = 40

    # nx       = 640
    # ny       = 320

    nt       = 10
    # nt       = 1000
    εnl      = 1e-6
    time_year = 365.25*24*3600
    maxiter = 10000


    nchk    = 100max(nx,ny)
    Re      = 5π
    r       = 1.0
    CFL     = 0.99/sqrt(2)


    # preprocessing
    @show dx,dy   = lx/nx,ly/ny
    _dx, _dy      = inv(dx), inv(dy)

    max_lxy = max(lx,ly)
    vpdτ    = CFL*min(dx,dy)
    xc,yc   = LinRange(-(lx-dx)/2,(lx-dx)/2,nx),LinRange(-(ly-dy)/2,(ly-dy)/2,ny)
    xv,yv   = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1)
    @show h_index  = ceil(Int, (ny - 1) / 2) + 1 # row index where the properties are stored for the fault


    # phyics
    # η0      = 1.0e15          # viscosity
    # η0      = 5.0e15          # viscosity
    # η0      = 1.0e16          # viscosity

    η0      = 1.0e23          # viscosity
    G0      = 3.0e10          # shear modulus


    # allocate arrays
    Pr_o    = @zeros(nx  ,ny  )
    Rp      = @zeros(nx  ,ny  )
    τxx     = @zeros(nx  ,ny  )
    τyy     = @zeros(nx  ,ny  )
    λ       = @zeros(nx  ,ny  )
    F       = @zeros(nx  ,ny  )
    τxx_o   = @zeros(nx  ,ny  )
    τyy_o   = @zeros(nx  ,ny  )
    τxyc_o  = @zeros(nx  ,ny  )
    τxy_o   = @zeros(nx+1,ny+1)
    Vy      = @zeros(nx  ,ny+1)
    Rx      = @zeros(nx-1,ny  )
    Ry      = @zeros(nx  ,ny-1)
    ∇V      = @zeros(nx  ,ny  )
    
    Pt      = 5.0e6
    Pr_cpu  = fill(Pt, nx, ny)
    Pr      = PTArray(Pr_cpu)
    τxy_cpu = zeros(nx+1,ny+1)
    τxyc_cpu= zeros(nx  ,ny  )
    τii_cpu = zeros(nx  ,ny  )
    Vx_cpu  = zeros(nx+1,ny  )
    
    # added inertia    
    ρ0      = 2700.0
    g       = 0.0
    ρ_cpu   = fill(ρ0, nx ,ny)
    ρ       = PTArray(ρ_cpu)
    ρg_cpu  = fill(ρ0*g, nx, ny)
    ρg      = PTArray(ρg_cpu)
    Vx_o    = @zeros(nx+1,ny  )
    Vy_o    = @zeros(nx  ,ny+1)
    
    # rate and state friction
    Vp_cpu      = zeros(nx, ny)
    
    # Parameters for rate-and-state dependent friction
    #            domain   fault
    a0        = [0.011    0.011]     # a-parameter of RSF
    b0        = [0.017    0.001]     # b-parameter of RSF    
    # Ω0        = [40.0    -1.0]     # State variable from the preνious time step
    Ω0        = [20.0    -1.0]     # State variable from the preνious time step


    V0        = 4.0e-9               # characteristic slip rate
    γ0        = 0.2                  # Reference Friction

    γ_cpu     = fill(γ0, nx,ny)
    γ         = PTArray(γ_cpu)

    a_cpu   = fill(a0[1],nx,ny)
    b_cpu   = fill(b0[1],nx,ny)
    # L_cpu   = fill(L0[1],nx,ny)
    Ω_cpu   = fill(Ω0[1],nx,ny)
    Ω_o     = @zeros(nx,ny)   # for state variable ODE update

    # setting up geometry
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
    # X = LinRange(0.0, lx, nx-1)
    X = LinRange(0.0, lx, nx)

    for i in 1:1:nx
        for j in 1:1:ny

            # if along the fault
            if j == h_index

                # assign domain value
                if x0 <= X[i] <= x1 || x4 <= X[i] <= x5
                    a_cpu[i,j]  = a0[1]
                    b_cpu[i,j]  = b0[1]
                    Ω_cpu[i,j]  = Ω0[1]
                    # L_cpu[i,j]  = L0[1]
                end

                # assign fault value
                if x2 <= X[i] <= x3
                    a_cpu[i,j]  = a0[2]
                    b_cpu[i,j]  = b0[2]
                    Ω_cpu[i,j]  = Ω0[2]
                    # L_cpu[i,j]  = L0[2]
                end

                # assign transition zone value (left)
                if x1 < X[i] < x2
                    a_cpu[i, j]    = a0[1] - (a0[1] - a0[2]) * ((X[i] - x1) / (x2 - x1))
                    b_cpu[i, j]    = b0[1] - (b0[1] - b0[2]) * ((X[i] - x1) / (x2 - x1))
                    Ω_cpu[i, j]    = Ω0[1] - (Ω0[1] - Ω0[2]) * ((X[i] - x1) / (x2 - x1))
                    # L_cpu[i, j]    = L0[1] - (L0[1] - L0[2]) * ((X[i] - x1) / (x2 - x1))

                end

                if x3 < X[i] < x4
                    a_cpu[i, j]  = a0[2] - (a0[2] - a0[1]) * ((X[i] - x3) / (x4 - x3))
                    b_cpu[i, j]  = b0[2] - (b0[2] - b0[1]) * ((X[i] - x3) / (x4 - x3))
                    Ω_cpu[i, j]  = Ω0[2] - (Ω0[2] - Ω0[1]) * ((X[i] - x3) / (x4 - x3))
                    # L_cpu[i, j]  = L0[2] - (L0[2] - L0[1]) * ((X[i] - x3) / (x4 - x3))
                end

            end

        end
    end

    a   = PTArray(a_cpu)
    b   = PTArray(b_cpu)
    Ω   = PTArray(Ω_cpu)
    # L   = PTArray(L_cpu)

    L   = 0.01
 
    # initial velocity
    VL        = 2.0e-9    
    Y = LinRange(0.0, ly, ny-1)
    y1 = 0.0
    y2 = 10.0e3
    y3 = 20.0e3
    
    Vx0 = [VL 0 -VL]

    for i in 1:1:nx
        for j in 1:1:ny-1
            
            # assign transition zone value (upper)
            if y1 <= Y[j] <= y2
                Vx_cpu[i, j]    = Vx0[3] - (Vx0[3] - Vx0[2]) * ((Y[j] - y1) / (y2 - y1))
            end
            
            if y2 <= Y[j] <= y3
                Vx_cpu[i, j]    = Vx0[2] - (Vx0[2] - Vx0[1]) * ((Y[j] - y2) / (y3 - y2))
            end
            

        end
    end
    
    Vx_cpu[:,1]    .= -VL
    Vx_cpu[:,end]  .= VL
    Vx  = PTArray(Vx_cpu)
    
    plota = heatmap(xv,yc,Array(Vx)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
    plotb = heatmap(xc,yv,Array(Vy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")

    # DEBUG
    # display(plot(plota, plotb))
    # savefig("vizGPU_out/VxVy.png")

    # FIXME: intialize stress such that frictional coefficient is of order 1e-2
    @. τxy_cpu   = 0.1*Pt
    @. τxyc_cpu  = 0.1*Pt

    τxy  = PTArray(τxy_cpu)
    τxyc = PTArray(τxyc_cpu)


    # precomputation of τII as slip rate requires this
    @. τii_cpu   = sqrt(#=0.5*(τxx^2 + τyy^2) + =#τxyc_cpu*τxyc_cpu)   # TODO: new compute_second_invariant!()
    # @. Vp_cpu = 2*V0*sinh((τii_cpu)/a_cpu/Pt)/exp((b_cpu*Ω_cpu + γ_cpu)/a_cpu)
    τii  = PTArray(τii_cpu)
    Vp   = PTArray(Vp_cpu)



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

    F     = @zeros(nx, ny)

    
    # action
    t = 0.0; evo_t = Float64[]; evo_τxx = Float64[]

    # record evolution of time step size and slip rate
    evo_t_year = Float64[]; evo_Δt = Float64[]; evo_Vp = Float64[]; evo_Peff = Float64[]

    for it = 1:nt

        # [x] correct!
        @parallel assign!(τxx_o, τxx, τyy_o, τyy, τxy_o, τxy, τxyc_o, τxyc, Pr_o, Pr, Vx_o, Vx, Vy_o, Vy, Ω_o, Ω)


        # FIXME: ADD ADAPTIVE TIMESTEPPING HERE
        # @show dt = min(1.9*L/maximum(Vp), 0.1*time_year)
        # @show dt = min(dx*1e-5/maximum(Vp), 0.1*time_year)
        @show dt = min(1.9*dx/maximum(Vp), 0.1*time_year)

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
        
        _dt     = inv(dt)


        err  = 2εnl; iter = 0
        while err > εnl && iter < maxiter
            #==========================================#
            # pressure [x] correct
            @parallel compute_∇!(∇V, Vx, Vy, _dx, _dy)
            @parallel compute_residual_mass_law!(Rp, ∇V)
            # @parallel compute_residual_mass_law_inertia!(Rp, ∇V, Pr, Pr_o, _dt)
            @parallel compute_pressure!(Pr, Gdτ, Rp, r)

            #==========================================#            
            # visco-elastic stress update containing PT terms
            @parallel compute_ve_stress!(Exy, τxx, τyy, τxy, τxyc, τxx_o, τyy_o, τxy_o, τxyc_o, Vx, Vy, η_ve_τ, η_ve_τv, η_e, η_ev, Gdτ, Gdτv, _dx, _dy)

            # preventing race condition!
            @parallel compute_tensor!(τxy, η_ve_τv, Exy, τxy_o, η_ev, Gdτv)                        
            
            # slip rate [x] correct!
            # FIXME: change 1: depenent on Pr instead of Pt
            @parallel compute_slip_rate!(Vp, τii, a, b, Ω, γ, Pt, V0)


            # plastic correction [x] correct!
            @parallel compute_plastic_correction!(λ[:, h_index], Vp[:, h_index], τxx[:, h_index], τyy[:, h_index], τxy[:, h_index], τxyc[:, h_index], τii[:, h_index], η_ve_τ[:, h_index], η_ve_τv[:, h_index], _dx)

            # second invariant [x] correct!
            @parallel compute_second_invariant!(τii, τxx, τyy, τxyc)

            #==========================================#
            # velocity [x] correct!
            @parallel compute_residual_momentum_law!(Rx, Ry, τxx, τyy, τxy, Pr, ρ, ρg, Vx, Vx_o, Vy, Vy_o, _dx, _dy, _dt)
            @parallel compute_velocity!(Vx, Vy, Rx, Ry, dτ_ρ)
            @parallel compute_state_parameter!(Ω, Vp, dt, V0, L)

            #==========================================#
            # Boundary conditions [x] correct
            @parallel (1:nx+1) dirichlet_y!(Vx, -VL, VL)
            @parallel (1:nx)   dirichlet_y!(Vy, 0.0, 0.0)            
            @parallel (1:ny)   free_slip_x!(Vx)
            @parallel (1:ny+1) free_slip_x!(Vy)
            # @parallel (1:ny+1) dirichlet_x!(Vy, 0.0, 0.0)            

            if iter % nchk == 0
                norm_Rx = norm(Rx)/sqrt(length(Rx)); norm_Ry = norm(Ry)/sqrt(length(Ry)); norm_∇V = norm(∇V)/sqrt(length(∇V))
                err = maximum([norm_Rx, norm_Ry, norm_∇V])
                @printf("it = %d, iter = %d, err = %1.2e norm[Rx=%1.2e, Ry=%1.2e, ∇V=%1.2e] (F=%1.2e) \n", it, iter, err, norm_Rx, norm_Ry, norm_∇V, maximum(F))
            end
            iter += 1


        end

        @show max_Vp          = maximum(Vp)
        @show max_τii         = maximum(τii)


        # @parallel compute_state_parameter!(Ω, Vp, dt, V0, L)

        t += dt; push!(evo_t, t); push!(evo_τxx, maximum(τxx))
        
        # store evolution of physical properties wrt time
        push!(evo_t_year, t/time_year); push!(evo_Vp, max_Vp);

        # p1 = heatmap(xc,yc,log.(Array(Vp)/V0)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\log_{10}(\frac{V_P}{V_0})$")
        p1 = heatmap(xc,yc,log.(Array(Vp))',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\log_{10}(V_P)$")

        p2 = heatmap(xc,yc,Array(Pr)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$P$")
        p3 = heatmap(xc,yc,Array(τxx)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\tau_\mathrm{xx}$")
        p4 = heatmap(xc,yc,Array(τyy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\tau_\mathrm{yy}$")
        p5 = heatmap(xv,yv,Array(τxy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\tau_\mathrm{xy}$")
        p6 = heatmap(xc,yc,Array(τii)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\tau_\mathrm{II}$")
        p7 = heatmap(xv,yc,Array(Vx)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
        p8 = heatmap(xc,yv,Array(Vy)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")

        p9 = plot(X, Array(τxx)[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="Shear stress on the fault", framestyle=:box, markersize=3)
        p10 = plot(X, Array(Ω)[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="State variable", framestyle=:box, markersize=3)
        p11 = plot(X, Array(Vp)[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="Slip rate", framestyle=:box, markersize=3)
     
        # FIXME: changed here!
        p12 = plot(evo_t_year, evo_Vp; xlims=(0.0, 1000), ylims=(1.0e-13, 1.0e2), yaxis=:log, yticks =[1e-13, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], label="", color= :dodgerblue, framestyle= :box, linestyle= :solid, 
        seriesstyle= :path, title="Seismo-Mechanical Simulation (t = " * string(@sprintf("%.3f", t/time_year)) * " year )", 
        xlabel = "Time [year]", ylabel="Maximum Slip Rate [m/s]" )

        display(plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12; layout=(4,3))); frame(anim)
    end                


    gif(anim, "vizGPU_out/rsf_strikeslip_fault.gif", fps = 15)

    @show evo_τxx
    @show evo_Vp

    return
end

# action
Stokes2D_vep()
