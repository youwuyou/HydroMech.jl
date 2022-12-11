using HydroMech

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

using Statistics, Printf, LinearAlgebra


## visualization
@static if DO_VIZ
    using Plots
end

# testing
@static if SAVE_TEST
    using JLD
end


##################################################
@views function HydroMech2D_incompressible(;t_tot_)
    # Physics - scales
    ρfg      = 1.0             # fluid rho*g
    k_μf0    = 1.0             # reference permeability
    ηC0      = 1.0             # reference bulk viscosity
  
    # Physics - non-dimensional parameters
    η2μs     = 10.0            # bulk/shear viscosity ration
    R        = 500.0           # Compaction/decompaction strength ratio for bulk rheology
    nperm    = 3.0             # Carman-Kozeny exponent
    ϕ0       = 0.01            # reference porosity
    ra       = 2               # radius of initil porosity perturbation
    λ0       = 1.0             # standard deviation of initial porosity perturbation
    t_tot    = t_tot_          # total time
  
    # Physics - dependent scales
    ρsg      = 2.0*ρfg         # solid rho*g
    lx       = 20.0            # domain size x
    ly       = ra*lx           # domain size y
    ϕA       = 2*ϕ0            # amplitude of initial porosity perturbation
    λPe      = 0.01            # effective pressure transition zone
    dt       = 1e-5            # physical time-step
  
    # Numerics
    CN       = 0.5             # Crank-Nicolson CN=0.5, Backward Euler CN=0.0
    res      = 128
    nx, ny   = res-1, ra*res-1 # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf
    ε        = 1e-5            # non-linear tolerance
    iterMax  = 5e3             # max nonlinear iterations
    nout     = 200             # error checking frequency
    β_n      = 1.0             # numerical compressibility
    Vdmp     = 5.0             # velocity damping for momentum equations
    Pfdmp    = 0.8             # fluid pressure damping for momentum equations
    Vsc      = 2.0             # reduction of PT steps for velocity
    Ptsc     = 2.0             # reduction of PT steps for total pressure
    Pfsc     = 4.0             # reduction of PT steps for fluid pressure
    θ_e      = 9e-1            # relaxation factor for non-linear viscosity
    θ_k      = 1e-1            # relaxation factor for non-linear permeability
    dt_red   = 1e-3            # reduction of physical timestep
  
    # Derived physics
    μs       = ηC0*ϕ0/η2μs                       # solid shear viscosity
    λ        = λ0*sqrt(k_μf0*ηC0)                # initial perturbation width
    ρgBG     = ρfg*ϕ0 + ρsg*(1.0-ϕ0)             # Background density
  
    # Derived numerics
    dx, dy   = lx/(nx-1), ly/(ny-1)              # grid step in x, y
    min_dxy2 = min(dx,dy)^2
    dτV      = min_dxy2/μs/(1.0+β_n)/4.1/Vsc     # PT time step for velocity
    dτPt     = 4.1*μs*(1.0+β_n)/max(nx,ny)/Ptsc
    dampX    = 1.0-Vdmp/nx
    dampY    = 1.0-Vdmp/ny

    
    # Array allocations
    Phi_o    = @zeros(nx  ,ny  )
    Pt       = @zeros(nx  ,ny  )
    Pf       = @zeros(nx  ,ny  )
    Rog      = @zeros(nx  ,ny  )
    ∇V       = @zeros(nx  ,ny  )
    ∇V_o     = @zeros(nx  ,ny  )
    ∇qD      = @zeros(nx  ,ny  )
    dτPf     = @zeros(nx  ,ny  )
    RPt      = @zeros(nx  ,ny  )
    RPf      = @zeros(nx  ,ny  )
    τxx      = @zeros(nx  ,ny  )
    τyy      = @zeros(nx  ,ny  )
    σxy      = @zeros(nx-1,ny-1)
    dVxdτ    = @zeros(nx-1,ny-2)
    dVydτ    = @zeros(nx-2,ny-1)
    Rx       = @zeros(nx-1,ny-2)
    Ry       = @zeros(nx-2,ny-1)
    Vx       = @zeros(nx+1,ny  )
    Vy       = @zeros(nx  ,ny+1)
    qDx      = @zeros(nx+1,ny  )


    # Initial conditions
    qDy_cpu             =   zeros(nx  ,ny+1)
    Phi_cpu             = ϕ0*ones(nx  ,ny  )
    Radc                =   zeros(nx  ,ny  )
    Radc               .= [(((ix-1)*dx-0.5*lx)/λ/4.0)^2 + (((iy-1)*dy-0.25*ly)/λ)^2 for ix=1:size(Radc,1), iy=1:size(Radc,2)]
    Phi_cpu[Radc.<1.0] .= Phi_cpu[Radc.<1.0] .+ ϕA
    EtaC_cpu            = μs./Phi_cpu.*η2μs
    K_muf_cpu           = k_μf0.*(Phi_cpu./ϕ0)
    ϕ0bc                = mean.(Phi_cpu[:,end])
    qDy_cpu[:,[1,end]] .= (ρsg.-ρfg).*(1.0.-ϕ0bc).*k_μf0.*(ϕ0bc./ϕ0).^nperm
    Phi                 = PTArray(Phi_cpu)
    EtaC                = PTArray(EtaC_cpu)
    K_muf               = PTArray(K_muf_cpu)
    qDy                 = PTArray(qDy_cpu)
    t                   = 0.0
    it                  = 1

    # boundary condition
    freeslip = (freeslip_x=true, freeslip_y=true)

    # HPC precomputation
    _dx, _dy    = 1.0/dx, 1.0/dy
    _ϕ0         = 1.0/ϕ0
    length_Ry   = length(Ry)
    length_RPf  = length(RPf)
    data_size   = sizeof(eltype(Phi))
 
    # Preparation of visualisation
    if DO_VIZ
        ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
    end
    
  
    # Time loop
    while t<t_tot
        @parallel update_old!(Phi_o, ∇V_o, Phi, ∇V)
        err=2*ε; iter=1; niter=0
       
        while err > ε && iter <= iterMax
            if (iter==11)  global wtime0 = Base.time()  end


            # @parallel compute_params_∇!(EtaC, K_muf, Rog, ∇V, ∇qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, μs, η2μs, R, λPe, k_μf0, _ϕ0, nperm, θ_e, θ_k, ρfg, ρsg, ρgBG, _dx, _dy)
            # @parallel compute_RP!(dτPf, RPt, RPf, K_muf, ∇V, ∇qD, Pt, Pf, EtaC, Phi, Pfsc, Pfdmp, min_dxy2, _dx, _dy)

            # apply_free_slip!(freeslip, dτPf, nx, ny)
 
            # @parallel compute_P_τ!(Pt, Pf, τxx, τyy, σxy, RPt, RPf, dτPf, Vx, Vy, ∇V, dτPt, μs, β_n, _dx, _dy)
            # @parallel compute_res!(Ry, dVxdτ, dVydτ, τxx, τyy, σxy, Pt, Rog, dampX, dampY, _dx, _dy)

            # @parallel compute_update!(Vx, Vy, qDx, qDy, Phi, dVxdτ, dVydτ, K_muf, Pf, Phi_o, ∇V, ∇V_o, dτV, ρfg, ρgBG, CN, dt, _dx, _dy)
          
            # apply_free_slip!(freeslip, Vx, Vy, nx+1, ny+1)
            # apply_free_slip!(freeslip, qDx, qDy, nx+1, ny+1)

            compute!(EtaC, K_muf, Rog, ∇V, ∇qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, μs, η2μs, R, λPe, k_μf0, _ϕ0, nperm, θ_e, θ_k, ρfg, ρsg, ρgBG, _dx, _dy,
            dτPf, RPt, RPf, Pfsc, Pfdmp, min_dxy2,
            freeslip, nx, ny, τxx, τyy, σxy,dτPt, β_n,
            Rx, Ry, dVxdτ, dVydτ, dampX, dampY,
            Phi_o, ∇V_o, dτV, CN, dt
            )


            if mod(iter,nout)==0
                global norm_Ry, norm_RPf
                norm_Ry = norm(Ry)/length_Ry; norm_RPf = norm(RPf)/length_RPf; err = max(norm_Ry, norm_RPf)
                # @printf("iter = %d, err = %1.3e [norm_Ry=%1.3e, norm_RPf=%1.3e] \n", iter, err, norm_Ry, norm_RPf)
            end
            iter+=1; niter+=1
        end
   
        # Performance
        wtime    = Base.time() - wtime0
        A_eff    = (8*2)/1e9*nx*ny*data_size  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(niter-10)                     # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
        @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))


        # Visualisation
        if DO_VIZ
            default(size=(500,700))
            if mod(it,5)==0
                p1 = heatmap(X, Y,  Array(Phi)'  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="porosity")
                p2 = heatmap(X, Y,  Array(Pt-Pf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="effective pressure")
                p3 = heatmap(X, Yv, Array(qDy)'  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical Darcy flux")
                p4 = heatmap(X, Yv, Array(Vy)'   , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical velocity")
                display(plot(p1, p2, p3, p4)); frame(anim)
            end
        end

        # Time
        dt = dt_red/(1e-10+maximum(abs.(∇V)))
        t  = t + dt
        it+=1
    end
    
    # Visualization
    if DO_VIZ
        gif(anim, "HydroMech2D_incompressible.gif", fps = 15)
    end

    # Testing
    ## store data in case further testings needed
    if SAVE_TEST
        save("../../test/2D/incom_Peff.jld", "data", Array(Pt-Pf)')  # store case for reference testing
    end

    # return effective pressure at final time
    return Array(Pt-Pf)'
end


# if isinteractive()
#     HydroMech2D_incompressible(;t_tot_=0.02) # for reproducing porosity wave benchmark
#     # HydroMech2D_incompressible(;t_tot_=0.1) # for reproducing porosity wave benchmark
#     # HydroMech2D_incompressible(;t_tot_=0.0005) # for reproducing the test result
# end
