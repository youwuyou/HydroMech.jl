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

@views function PorosityWave2D_compressible(;t_tot_)

    # MESH
    ra       = 2               # radius of initil porosity perturbation
    lx       = 20.0            # domain size x
    ly       = ra*lx           # domain size y
    res      = 256
    nx, ny   = res-1, ra*res-1 # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf
    dx, dy   = lx/(nx-1), ly/(ny-1)              # grid step in x, y

    mesh     = PTGrid((nx,ny), (lx,ly),(dx,dy))

    # RHEOLOGY
    # i). viscous rheology

    # ii).Porosity dependent viscosity
    # η_ϕ = η_c ⋅ ɸ0/ɸ (1+ 1/2(1/R − 1)(1+tanh(−Pₑ/λₚ)))
    # ηc = μs/C/φ0
    
    # iii). power law permeability
    # k_ɸ = k0 (ɸ/ɸ0)^nₖ    
    
    # Physics - non-dimensional parameters
    R        = 1.0           # Compaction/decompaction strength ratio for bulk rheology
    nₖ       = 3.0             # Carman-Kozeny exponent
    ɸ0       = 0.01            # reference porosity
    ɸA       = 2*ɸ0            # amplitude of initial porosity perturbation
    C        = 0.1            # bulk/shear viscosity ratio
    ηC0      = 1.0             # reference bulk viscosity
    μˢ       = ηC0*ɸ0*C                      # solid shear viscosity
    µᶠ       = 1.0
    λp       = 0.01            # effective pressure transition zone
    k0       = 1.0             # reference permeability

    rheology = ViscousRheology(μˢ,µᶠ,C,R,λp,k0,ɸ0,nₖ)


    # TWO PHASE FLOW
    # forces
    ρfg      = 1.0             # fluid rho*g
    ρsg      = 2.0*ρfg         # solid rho*g
    ρgBG     = ρfg*ɸ0 + ρsg*(1.0-ɸ0)             # Background density
    
    # initial conditions
    λ0              = 1.0                            # standard deviation of initial porosity perturbation
    λ               = λ0*sqrt(k0*ηC0)                # initial perturbation wiΔth
    Radc            =   zeros(nx  ,ny  )
    Radc           .= [(((ix-1)*dx-0.5*lx)/λ/4.0)^2 + (((iy-1)*dy-0.25*ly)/λ)^2 for ix=1:size(Radc,1), iy=1:size(Radc,2)]
    𝝫              = ɸ0*ones(nx  ,ny  )
    𝝫[Radc.<1.0]  .= 𝝫[Radc.<1.0] .+ ɸA

    𝞅0bc            = mean.(𝝫[:,end])
    qDy             =   zeros(nx  ,ny+1)
    qDy[:,[1,end]] .= (ρsg.-ρfg).*(1.0.-𝞅0bc).*k0.*(𝞅0bc./ɸ0).^nₖ
    
    𝞰ɸ              = μˢ./𝝫./C
    𝐤ɸ_µᶠ           = k0.*(𝝫./ɸ0)
    

    flow              = TwoPhaseFlow2D(mesh, (ρfg, ρsg, ρgBG))
    flow.qD.y         = PTArray(qDy)
    flow.𝝫            = PTArray(𝝫)
    flow.𝞰ɸ           = PTArray(𝞰ɸ)
    flow.𝐤ɸ_µᶠ        = PTArray(𝐤ɸ_µᶠ)

    # PHYSICS FOR COMPRESSIBILITY
    µ   = 25.0
    ν   = 0.25      # Poisson ratio
    Ks  = 50.0 
    βs  = 0.25
    βf  = 0.04

    compressibility = Compressibility(mesh, µ, ν, Ks, βs, βf)


    # PT COEFFICIENT
    pt = PTCoeff(OriginalDamping, mesh, μˢ)
    
    # BOUNDARY CONDITIONS
    freeslip = (freeslip_x=true, freeslip_y=true)
    
    
    # Preparation of visualisation
    if DO_VIZ
        ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
    end
    
    
    # Time loop
    t_tot    = t_tot_          # total time
    Δt       = 1e-5            # physical time-step
    Δt_red   = 1e-3            # reduction of physical timestep
    t        = 0.0
    it       = 1

    while t<t_tot

        # Pseudo-time loop solving
        solve!(flow, compressibility, rheology, mesh, freeslip, pt, Δt, it)
   
        # Visualisation
        if DO_VIZ
            default(size=(500,700))
            if mod(it,5)==0
                p1 = heatmap(X, Y,  Array(flow.𝝫)'  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="porosity")
                p2 = heatmap(X, Y,  Array(flow.Pt-flow.Pf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="effective pressure")
                p3 = heatmap(X, Yv, Array(flow.qD.y)'  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical Darcy flux")
                p4 = heatmap(X, Yv, Array(flow.V.y)'   , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical velocity")
                display(plot(p1, p2, p3, p4)); frame(anim)
            end
        end

        # Time
        Δt = Δt_red/(1e-10+maximum(abs.(flow.∇V)))
        t  = t + Δt
        it+=1
    end
    
    # Visualization
    if DO_VIZ
        gif(anim, "PorosityWave2D_compressible.gif", fps = 15)
    end

    # Testing
    # store data in case further testings needed
    if SAVE_TEST
        save("../../test/2D/com_Peff.jld", "data", Array(flow.Pt-flow.Pf)')  # store case for reference testing
    end

    # return effective pressure at final time
    return Array(flow.Pt-flow.Pf)'

end


# if isinteractive()
    PorosityWave2D_compressible(;t_tot_=0.02) # for reproducing porosity wave benchmark
    # PorosityWave2D_compressible(;t_tot_=0.03) # for R=1
#     # PorosityWave2D_compressible(;t_tot_=0.1) # for reproducing porosity wave benchmark
    # PorosityWave2D_compressible(;t_tot_=0.0005) # for reproducing the test result
# end
