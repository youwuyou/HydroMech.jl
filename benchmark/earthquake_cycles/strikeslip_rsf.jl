# Script derived from Stokes2D_vep_reg_IU.jl
# Physics: Incompressible stokes equation with VEP
#          - momentum equation with inertia effects included

# TODO:
# [x] add inertia to momentum equation
# [x] add rate-dependent friction
# [x] add adaptive time stepping
# [] move on to strike-slip fault setup
# [] add rate- and state- friction and try out with weak inclusion setup



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

DO_VIZ = true

# in case needed
# @views  av_xi(A) =  0.5*(A[2:end-2,2:end-1].+A[3:end-1,2:end-1])
# @views  av_yi(A) =  0.5*(A[2:end-1,2:end-2].+A[2:end-1,3:end-1])

# VISUALIZATION
if DO_VIZ
    default(size=(3200,1700),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=7mm)
    scalefontsizes(); scalefontsizes(1.35)

    ENV["GKSwstype"]="nul"; if isdir("viz10_out")==false mkdir("viz10_out") end; loadpath = "./viz10_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
end

# main function
@views function Stokes2D_vep()
    # numerics
    lx       = 40000.0  # [m] 40 km
    ly       = 20000.0  # [m] 20 km
    nx       = 320
    ny       = 160
    nt       = 3000
    # εnl      = 1e-1
    εnl      = 1e-6
    # maxiter = 100max(nx,ny)
    # maxiter = 10000
    maxiter = 30000


    nchk    = 20max(nx,ny)
    Re      = 5π
    r       = 1.0
    CFL     = 0.99/sqrt(2)

    # preprocessing
    @show dx,dy   = lx/nx,ly/ny
    max_lxy = max(lx,ly)
    vpdτ    = CFL*min(dx,dy)
    xc,yc   = LinRange(-(lx-dx)/2,(lx-dx)/2,nx),LinRange(-(ly-dy)/2,(ly-dy)/2,ny)
    xv,yv   = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1)
    @show h_index  = ceil(Int, (ny - 1) / 2) + 1 # row index where the properties are stored for the fault


    # phyics
    η_reg   = 0.0
    η0      = 2.0e23          # viscosity
    G0      = 25.0e9          # shear modulus
    # τ_c     = 4.5e7         # yield stress. If do_DP=true, τ_c stand for the cohesion: c*cos(ϕ)
    τ_c     = 3.0e7           # yield stress. If do_DP=true, τ_c stand for the cohesion: c*cos(ϕ)
    # τ_c     = 3.0e5         # yield stress. If do_DP=true, τ_c stand for the cohesion: c*cos(ϕ)

    Gi      = G0/3.0
    ebg     = 2.0e-9

    # dt      = η0/G0/4.0
    # dt      = η0/G0/4.0e8

    dt      = η0/G0/4.0e6


    # allocate arrays
    # FIXME: testing yield stress
    τ0      = zeros(nx, ny)    
    Pt      = 30.0e6                   # 30 MPa
    Pr      = fill(Pt, nx, ny)
    Pr_o    = zeros(nx  ,ny  )
    Rp      = zeros(nx  ,ny  )

    τxx     = zeros(nx  ,ny  )
    τyy     = zeros(nx  ,ny  )
    τxy     = zeros(nx+1,ny+1)
    τxyc    = zeros(nx  ,ny  )
    τii     = zeros(nx  ,ny  )
    λ       = zeros(nx  ,ny  )
    F       = zeros(nx  ,ny  )
    τxx_o   = zeros(nx  ,ny  )
    τyy_o   = zeros(nx  ,ny  )
    τxyc_o  = zeros(nx  ,ny  )
    τxy_o   = zeros(nx+1,ny+1)
    Vx      = zeros(nx+1,ny  )
    Vy      = zeros(nx  ,ny+1)
    Rx      = zeros(nx-1,ny  )
    Ry      = zeros(nx  ,ny-1)
    ∇V      = zeros(nx  ,ny  )

    
    
    # added inertia
    ρ0      = 2700.0
    # g       = 9.81998
    g       = 0.0

    ρ       = fill(ρ0, nx ,ny)
    Vx_o    = zeros(nx+1,ny  )
    Vy_o    = zeros(nx  ,ny+1)
    ρg      = fill(ρ0*g, nx, ny)

    # rate and state friction
    γ_eff       = fill(sind(30), nx, ny)
    Vp          = zeros(nx, ny)
    Dc          = dx


    # boundary condition
    VL        = 2.0e-9
    time_year = 365.25*24*3600



    
    # Parameters for rate-and-state dependent friction
    #            domain   fault
    a0        = [0.100    0.006]     # a-parameter of RSF
    b0        = [0.003    0.005]     # b-parameter of RSF
    # Ω0        = [10.0      -5.0]     # State variable from the preνious time step
    Ω0        = [5.0      5.0]     # State variable from the preνious time step

    L0        = [0.010    0.001]     # L-parameter of RSF (characteristic slip distance)
    V0        = 1.0e-9               # characteristic slip rate
    γ0        = 0.6                  # Reference Friction
    γ         = fill(γ0, nx,ny)

    a   = fill(a0[1],nx,ny)
    b   = fill(b0[1],nx,ny)
    Ω   = fill(Ω0[1],nx,ny)
    Ω_o = zeros(nx,ny)
    L   = fill(L0[1],nx,ny)

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
                    a[i,j]  = a0[1]
                    b[i,j]  = b0[1]
                    Ω[i,j]  = Ω0[1]
                    L[i,j]  = L0[1]
                end

                # assign fault value
                if x2 <= X[i] <= x3
                    a[i,j]  = a0[2]
                    b[i,j]  = b0[2]
                    Ω[i,j]  = Ω0[2]
                    L[i,j]  = L0[2]
                end

                # assign transition zone value (left)
                if x1 < X[i] < x2
                    a[i, j]    = a0[1] - (a0[1] - a0[2]) * ((X[i] - x1) / (x2 - x1))
                    b[i, j]    = b0[1] - (b0[1] - b0[2]) * ((X[i] - x1) / (x2 - x1))
                    Ω[i, j]    = Ω0[1] - (Ω0[1] - Ω0[2]) * ((X[i] - x1) / (x2 - x1))
                    L[i, j]    = L0[1] - (L0[1] - L0[2]) * ((X[i] - x1) / (x2 - x1))

                end

                if x3 < X[i] < x4
                    a[i, j]  = a0[2] - (a0[2] - a0[1]) * ((X[i] - x3) / (x4 - x3))
                    b[i, j]  = b0[2] - (b0[2] - b0[1]) * ((X[i] - x3) / (x4 - x3))
                    Ω[i, j]  = Ω0[2] - (Ω0[2] - Ω0[1]) * ((X[i] - x3) / (x4 - x3))
                    L[i, j]  = L0[2] - (L0[2] - L0[1]) * ((X[i] - x3) / (x4 - x3))

                end

            end

        end
    end


    # DEBUG
    # plotAA = plot(X, Ω[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="State variable", framestyle=:box, markersize=3)
    # display(plot(plotAA))
    # savefig("viz5_out/test_setup.png")


    Exy     = zeros(nx+1,ny+1)
    η_ve_τ  = zeros(nx  ,ny  )
    η_ve_τv = zeros(nx+1,ny+1)
    η_ve    = zeros(nx  ,ny  )
    η_vem   = zeros(nx  ,ny  )
    η_vev   = zeros(nx+1,ny+1)
    η_vevm  = zeros(nx+1,ny+1)
    dτ_ρ    = zeros(nx  ,ny  )
    dτ_ρv   = zeros(nx+1,ny+1)
    Gdτ     = zeros(nx  ,ny  )
    Gdτv    = zeros(nx+1,ny+1)

    # FIXME: adaptive time stepping dtvep
    Eii     =  zeros(nx, ny)
    η_vep   =  zeros(nx, ny)
    η_vepv  =  zeros(nx+1, ny+1)
    

    # init
    Y = LinRange(0.0, ly, ny-1)
    
    y1 = 0.0
    y2 = 10.0e3
    y3 = 20.0e3
    
    Vx0 = [VL 0 -VL]
    
    
    for i in 1:1:nx
        for j in 1:1:ny-1
            
            # assign transition zone value (upper)
            if y1 <= Y[j] <= y2
                Vx[i, j]    = Vx0[3] - (Vx0[3] - Vx0[2]) * ((Y[j] - y1) / (y2 - y1))
            end
            
            if y2 <= Y[j] <= y3
                Vx[i, j]    = Vx0[2] - (Vx0[2] - Vx0[1]) * ((Y[j] - y2) / (y3 - y2))
            end
            

        end
    end
    
    
    
    # along x-axis
    Vx[:,1]    .= -VL
    Vx[:,end]  .= VL
    
    # along y-axis
    # Vx[1,:]    .= Vx[2,:]
    # Vx[end,:]  .= Vx[end-1,:]
    
    # Vy = [ ebg*x for x ∈ xc, _ ∈ yv ]
    # Vy = [-VL*y for _ ∈ xc, y ∈ yv ]
    # Vy = [ ebg*x for x ∈ xc, _ ∈ yv ]
    # Vy = [-ebg*y for _ ∈ xc, y ∈ yv ]

    # Vy[:,1]    .= 0.0
    # Vy[:,end]  .= 0.0

    

    # FIXME: intialize stress such that frictional coefficient is of order 1e-2
    @. τxy   = 0.1*Pt
    @. τxyc  = 0.1*Pt

    # precomputation of τII as slip rate requires this
    @. τii   = sqrt(0.5*(τxx^2 + τyy^2) + τxyc*τxyc)   # TODO: new compute_second_invariant!()
    @. Vp = 2*V0*sinh((τii)/a/Pt)/exp((b*Ω + γ)/a)



    plota = heatmap(xv,yc,Vx',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
    plotb = heatmap(xc,yv,Vy',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")

    display(plot(plota, plotb))
    savefig("viz10_out/VxVy.png")


    η  = fill(η0,nx,ny); ηv = fill(η0,nx+1,ny+1)
    G  = fill(G0,nx,ny); Gv = fill(G0,nx+1,ny+1)

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

    @show vpdτ
    @show extrema(η_ve)
    @show extrema(Gdτ)
    @show extrema(dτ_ρ)
    @show extrema(η_ve_τ)

    
    # action
    t = 0.0; evo_t = Float64[]; evo_τxx = Float64[]

    # record evolution of time step size and slip rate
    evo_t_year = Float64[]; evo_Δt = Float64[]; evo_Vp = Float64[]; evo_Peff = Float64[]

    for it = 1:nt

        τxx_o .= τxx; τyy_o .= τyy; τxy_o .= τxy; τxyc_o .= τxyc

        Pr_o .= Pr

        # inertia
        Vx_o .= Vx; Vy_o .= Vy

        # update of aging law
        Ω_o .= Ω

        # FIXME: ADD ADAPTIVE TIMESTEPPING HERE
        @show dt = min(1.9*Dc/maximum(Vp), 0.1*time_year)
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



        err  = 2εnl; iter = 0
        while err > εnl && iter < maxiter
            #==========================================#
            # pressure
            ∇V    .= diff(Vx, dims=1)./dx .+ diff(Vy, dims=2)./dy    # ! compute _∇
            # @. Pr -= r*Gdτ*∇V                                        # ! compute residual mass law + ! compute pressure newdamping
            
            # pressure
            # Rp    .= -∇V - (Pr - Pr_o)/dt
            Rp    .= -∇V
            @. Pr = Pr + r*Gdτ*Rp                                        # ! compute residual mass law + ! compute pressure newdamping
            #==========================================#
            
            # new compute_ve_stress!()
            # viscoelastic pseudo-transient strain rates
            Exy[2:end-1,2:end-1] .= 0.5.*(diff(Vx[2:end-1,:], dims=2)./dy .+ diff(Vy[:,2:end-1], dims=1)./dx)
            

            # @. Vp = 2*V0*sinh((max(τii, 0.0)-τ_c)/a/Pr)/exp((b*Ω + γ)/a)
            # @. Vp = 2*V0*sinh((τii-τ_c)/a/Pt)/exp((b*Ω + γ)/a)
            @. Vp = 2*V0*sinh((τii)/a/Pt)/exp((b*Ω + γ)/a)
            @. λ[:, h_index]  = Vp[:, h_index]/2.0/dx
            
            # @. Vp = 2*V0*(max(τii, 0.0)/a/Pt)^2*exp(-(b*Ω+γ)/a)
            # @. λ[:, h_index]  = Vp[:, h_index]/2.0/dx

            # why not (τxx.- τxx_o)/2/η_e/dt ? for elastic part => embedded in the formulation already, can be derived
            # visco-elastic stress update containing PT terms
            τxx   .= 2.0.*η_ve_τ .* (diff(Vx, dims=1)./dx .+ τxx_o./2.0./η_e .+ τxx./2.0./Gdτ)
            τyy   .= 2.0.*η_ve_τ .* (diff(Vy, dims=2)./dy .+ τyy_o./2.0./η_e .+ τyy./2.0./Gdτ)
            @. τxy = 2.0*η_ve_τv  * (Exy + τxy_o/2.0/η_ev + τxy/2.0/Gdτv)
            τxyc  .= 2.0.*η_ve_τ .* (av(Exy) .+ τxyc_o./2.0./η_e + τxyc./2.0./Gdτ)


            @. τxx[:,h_index]    -= 2.0*η_ve_τ[:,h_index] *(λ[:,h_index]*0.5*τxx[:,h_index] /τii[:,h_index])
            @. τyy[:,h_index]    -= 2.0*η_ve_τ[:,h_index] *(λ[:,h_index]*0.5*τyy[:,h_index] /τii[:,h_index])
            @. τxyc[:,h_index]   -= 2.0*η_ve_τ[:,h_index] *(λ[:,h_index]*0.5*τxyc[:,h_index]/τii[:,h_index])
            τxy[2:end-1,h_index] .-= 2.0 .* η_ve_τv[2:end-1,h_index].*(0.5 .* av(λ.*τxyc./τii)[:,h_index])

            @. τii   = sqrt(0.5*(τxx^2 + τyy^2) + τxyc*τxyc)   # TODO: new compute_second_invariant!()



            # FIXME: maybe checking out if better ways exist to avoid redundant computations
            # @. Eii   = sqrt(0.5*((τxx/2.0/η_ve_τ)^2 + (τyy/2.0/η_ve_τ)^2) + (τxyc/2.0/η_ve_τ)^2)

            # # FIXME: ADAPTIVE TIMESTEPPING
            # @. η_vep  = τii /2.0 /Eii
            # η_vepv[2:end-1,2:end-1] .= av(η_vep); η_vepv[1,:].=η_vepv[2,:]; η_vepv[end,:].=η_vepv[end-1,:]; η_vepv[:,1].=η_vepv[:,2]; η_vepv[:,end].=η_vepv[:,end-1]


            #==========================================#
            # ! compute residual momentum law damping with inertia
            Rx .=  (diff(τxx, dims=1).-diff(Pr, dims=1))./dx .+ diff(τxy[2:end-1,:], dims=2)./dy .- av_xa(ρ) .* (Vx[2:end-1, :] .- Vx_o[2:end-1, :]) ./ dt
            Ry .=  (diff(τyy, dims=2) .-diff(Pr, dims=2))./dy  .+ diff(τxy[:,2:end-1], dims=1)./dx .+ av_ya(ρg) .- av_ya(ρ) .* (Vy[:, 2:end-1] .- Vy_o[:, 2:end-1]) ./ dt

            # ! compute velocity new damping
            Vx[2:end-1,:] .= Vx[2:end-1,:] .+ av_xa(dτ_ρ) .* Rx
            Vy[:,2:end-1] .= Vy[:,2:end-1] .+ av_ya(dτ_ρ) .* Ry
            
            #==========================================#

            # Boundary conditions
            # along x-axis
            Vx[:,1]    .= -VL
            Vx[:,end]  .= VL
            Vy[:,1]    .= 0.0
            Vy[:,end]  .= 0.0

            # along y-axis
            Vx[1,:]    .= Vx[2,:]
            Vx[end,:]  .= Vx[end-1,:]

            # TODO: check boundary conditions!
            Vy[1,:]    .= Vy[2,:]
            Vy[end,:]  .= Vy[end-1,:]
            # Vy[1,:]    .= 0.0
            # Vy[end,:]  .= 0.0
            

            if iter % nchk == 0
                norm_Rx = norm(Rx)/sqrt(length(Rx)); norm_Ry = norm(Ry)/sqrt(length(Ry)); norm_∇V = norm(∇V)/sqrt(length(∇V))
                err = maximum([norm_Rx, norm_Ry, norm_∇V])
                @printf("it = %d, iter = %d, err = %1.2e norm[Rx=%1.2e, Ry=%1.2e, ∇V=%1.2e] (F=%1.2e) \n", it, iter, err, norm_Rx, norm_Ry, norm_∇V, maximum(F))
            end
            iter += 1
        end

        @show max_Vp          = maximum(Vp)
        @show max_τii         = maximum(τii)

        # FIXME: rate-and-state dependent friction

        # TODO: plastic multiplier - dependent on stress and state
        

        # @. Vp = 2*V0*sinh(max(τii, 0.0)/a/Pr)*exp(-(b*Ω+γ)/a)
        # @. Vp = 2*V0*(max(τii, 0.0)/a/Pr)^2*exp(-(b*Ω+γ)/a)


        @. Ω  = Ω_o + dt * (V0*exp(-Ω_o) - Vp)/ Dc        # aging_law(Ω, Δt, V, Dc)

        # regularized rsf - turn this off to recover old rate-independent friction!
        # @. γ_eff = asinh(Vp/2/V0 * exp((b * Ω + γ) / a)) * a


        # ADAPTIVE TIMESTEPPING
        # δd         = 1.0e-5
        # @show dts   = dx*δd/max_Vp
        # @show dtd   = δd*min(abs(dx/minimum(Vx)),abs(dy/minimum(Vy)))
        # dt = min(dts, dtd)        


        # @show dtvep = 0.2*maximum(η_vep)/(G0/(1-0.25))

        # dt = min(dts, dtd, dtvep)        

        t += dt; push!(evo_t, t); push!(evo_τxx, maximum(τxx))
        
        # store evolution of physical properties wrt time
        push!(evo_t_year, t/time_year); push!(evo_Vp, max_Vp);

        p1 = heatmap(xc,yc,log.(Vp/V0)',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\log_{10}(\frac{V_P}{V_0})$")
        p2 = heatmap(xc,yc,Pr',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$P$")
        p3 = heatmap(xc,yc,τxx',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\tau_\mathrm{xx}$")
        p4 = heatmap(xc,yc,τyy',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\tau_\mathrm{yy}$")
        p5 = heatmap(xc,yc,τii',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title=L"$\tau_\mathrm{II}$")

        p6 = heatmap(xv,yc,Vx',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vx")
        p7 = heatmap(xc,yv,Vy',aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")

        p8 = plot(X, τxx[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="Shear stress on the fault", framestyle=:box, markersize=3)
        p9 = plot(X, Ω[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="State variable", framestyle=:box, markersize=3)
        p10 = plot(X, Vp[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="Slip rate", framestyle=:box, markersize=3)
        p11 = plot(X, γ_eff[:, h_index] , legend=false, xlabel="", xlims=(0.0,lx), title="Effective frictional coefficient", framestyle=:box, markersize=3)


        p12 = plot(evo_t_year, evo_Vp; xlims=(0.0, dt*nt/time_year), ylims=(1.0e-12, 1.0e2), yaxis=:log, yticks =[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], label="", color= :dodgerblue, framestyle= :box, linestyle= :solid, 
        seriesstyle= :path, title="Seismo-Mechanical Simulation (t = " * string(@sprintf("%.3f", t/time_year)) * " year )", 
        xlabel = "Time [year]", ylabel="Maximum Slip Rate [m/s]" )

        display(plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12; layout=(4,3))); frame(anim)
    end                


    gif(anim, "viz10_out/rsf_strikeslip_fault.gif", fps = 15)

    @show evo_τxx
    @show evo_Vp

    return
end

# action
Stokes2D_vep()
