using HydroMech

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)

using Statistics, Printf, LinearAlgebra


# visualization
@static if DO_VIZ
    using Plots
end


const COMPRESSIBLE    = true
const FLUID_INJECTION = true
const ADAPTIVE        = false



@static if FLUID_INJECTION
    using JLD
end

##################################################
@views function fluid_injection(;t_tot_)


    # MESH
    if FLUID_INJECTION
        lx       = 100.0 # [m]
        ly       = 20.0  # [m]
    else
        lx       = 100000.0               # domain size x - 100 km
        ly       = 20000.0                # domain size y - 20  km
    end
    
    nx       = 1001
    ny       = 201                       # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf

    @show dx, dy   = lx/(nx-1), ly/(ny-1)   # grid step in x, y

    rows    = 201
    cols    = 1001

    @show mesh = PTGrid((rows,cols), (lx,ly), (dx,dy))

    
    # index for accessing the corresponding row of the interface
    h_index = Int((ny - 1) / 2) # row index where the properties are stored for the fault

    @show h_index

    # RHEOLOGY
    # i). viscous rheology

    # ii).Porosity dependent viscosity
    # η_ϕ = η_c ⋅ ɸ0/ɸ (1+ 1/2(1/R − 1)(1+tanh(−Pₑ/λₚ)))
    # ηc = μs/C/φ0
    
    # iii). power law permeability
    # k_ɸ = k0 (ɸ/ɸ0)^nₖ = k0 (ɸ/ɸ0)^3
    

    # in order to recover formulation in Dal Zilio (2022)
    C        = 1.0             # bulk/shear viscosity ratio
    R        = 1.0             # Compaction/decompaction strength ratio for bulk rheology

    # from table 1
    ɸ0       = 0.01            # reference porosity   -> 1%
    k0       = 1e-16           # reference permeability [m²]
    μˢ       = 1e23            # solid shear viscosity [Pa·s]
    µᶠ       = 1e-3            # fluid viscosity

    #====================#
        
    rheology = ViscousRheology(μˢ,µᶠ,C,R,k0,ɸ0)

    # TWO PHASE FLOW
    # forces
    ρf       = 1.0e3                    # fluid density 1000 kg/m^3
    ρs       = 2.7e3                    # solid density 2700 kg/m^3


    if FLUID_INJECTION
        g        = 0.0
    else
        g        = 9.81998                  # gravitational acceleration [m/s^2]
    end
    
    ρfg      = ρf * g                   # force fluid
    ρsg      = ρs * g                   # force solid
    ρgBG     = ρfg*ɸ0 + ρsg*(1.0-ɸ0)    # Background density
    
    #====================#

    # Initial conditions
    𝝫                     =  ɸ0 *ones(rows, cols)
    𝞰ɸ                    =   μˢ./C./𝝫
    flow                  = TwoPhaseFlow2D(mesh, (ρfg, ρsg, ρgBG))
    
    if FLUID_INJECTION
        kɸ_fault          = 1e-16                         # domain with high permeability
        kɸ_domain         = 1e-23                         # domain with low permeability


        𝐤ɸ_µᶠ             = kɸ_domain/µᶠ * ones(rows,cols)    # porosity-dependent permeability
        𝐤ɸ_µᶠ[h_index,:] .= kɸ_fault/µᶠ                   # 1e-14
        
        pf                = 5.0e6                         # [Pa] = 5MPa Pf at t = 0
        Pf                = pf * ones(rows,cols)


        pt                = 20.0e6                        #  [Pa] = 20MPa
        Pt                = pt * ones(rows,cols)

    else 
        𝐤ɸ_µᶠ             = k0.*(𝝫./ɸ0)            # porosity-dependent permeability
        pf                = 40.0e+6               # 40 [MPa]
        Pf                = pf*ones(rows,cols)


        pt                = 1.0e+7    # [Pa]
        Pt                = pt*ones(rows,cols)


    end
    
    flow.𝝫               = PTArray(𝝫)
    flow.𝞰ɸ              = PTArray(𝞰ɸ)
    flow.𝐤ɸ_µᶠ           = PTArray(𝐤ɸ_µᶠ)
    flow.Pf              = PTArray(Pf)
    flow.Pt              = PTArray(Pt)

    # bc for fluid injection benchmark
    peff                = pt - pf


    if COMPRESSIBLE

        # PHYSICS FOR COMPRESSIBILITY
        µ   = 25.0e+9   # shear modulus 25 GPa
        ν   = 0.25      # Poisson ratio
        Ks  = 50.0e+9   # bulk modulus  50 GPa

        # Table 1
        βs = 2.5e-11      # solid compressibility  # [1/Pa]
        βf = 4.0e-10      # fluid compressibility  # [1/Pa]

        compressibility   = Compressibility(mesh, µ, ν, Ks, βs, βf)

    end
    

    # PT COEFFICIENT
    # scalar shear viscosity μˢ = 1.0 was used in porosity wave benchmark to construct dτPt
    # NOTE: μˢ = 1e23, other factors are used for time step size reduction
    if COMPRESSIBLE
        pt = PTCoeff(OriginalDamping, mesh, 1e23, Pfᵣ = 1.0e7, Ptᵣ = 1.0e25, Vᵣ = 0.825, dampPf = 1.0, dampV  = 0.5)        # choose this norm does not get smaller than norm_Rx=5.261e-01, dt = 15
    else
        pt = PTCoeff(ConstantStepDamping, mesh, 1e23, Pfᵣ = 1.0e7, Ptᵣ = 1.0e20, Vᵣ = 2.0)  # norm_Rx fluctuates around 1.743e-03 at it = 209000
    end

    
    # BOUNDARY CONDITIONS
    freeslip = (freeslip_x=true, freeslip_y=true)
    
    
    # Preparation of visualisation
    if DO_VIZ
        ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
        Xv          = (-dx/2):dx:(lx+dx/2)
    end
    
  
    # Time loop
    t_tot    = t_tot_          # total time
    Δt       = 5.0             # physical time-step
    t        = 0.0
    it       = 1
    
    while t<t_tot

        # Pseudo-time loop solving
        if COMPRESSIBLE
            solve!("StrikeSlip", h_index, flow, compressibility, rheology, mesh, freeslip, pt, Δt, it, Peff=peff, iterMax = 5.0e+5)
        else
            solve!("StrikeSlip", h_index, flow, rheology, mesh, freeslip, pt, Δt, it, Peff=peff)
        end

        # store fluid pressure for wanted time points
        if FLUID_INJECTION && mod(it, 1) == 0
            save("fluid_injection/Pf_fault" * string(it) * ".jld", "data", Array(flow.Pf[h_index,:])')   # store the fluid pressure along the fault for fluid injection benchmark
        end


        # Visualisation
        if DO_VIZ

            # y_flip is for correct orientation
            default(size=(1600,1400), y_flip= true)

            l = @layout [a b ; c d; e f]

            if mod(it,5) == 0
                # orientation
                p1 = heatmap(Array(flow.Pf)    , aspect_ratio=1, c=:bwr, title="fluid pressure")
                p2 = heatmap(Array(flow.Pt)    , aspect_ratio=1, c=:bwr, title="solid pressure")
               
                p3 = heatmap(Array(flow.qD.x)  , aspect_ratio=1, c=:bwr, title="horizontal Darcy flux")               
                p5 = heatmap(Array(flow.V.x)   , aspect_ratio=1, c=:bwr, title="horizontal velocity")
                
                p4 = heatmap(Array(flow.qD.y)  , aspect_ratio=1, c=:bwr, title="vertical Darcy flux")
                p6 = heatmap(Array(flow.V.y)   , aspect_ratio=1, c=:bwr, title="vertical velocity")


                display(plot(p1, p2, p3, p4, p5, p6, layout=l)); frame(anim)
            end



            
        end
        
        
        
        # Debug
        # @show it
        # @show t   = t + Δt

        # @show pt.Δτᵥ         
        
        # @show pt.Δτₚᶠ[1,1]            # domain
        # @show pt.Δτₚᶠ[h_index, 10]    # along fault
        # @show pt.Δτₚᶠ[h_index, 20]    # along fault

        # @show pt.Δτₚᵗ        


        # @show flow.Pf[h_index, 10]
        # @show flow.Pf[h_index, 20]
        # @show flow.Pf[h_index, 30]


        # Time        
        if ADAPTIVE
            δd       = 1e-5     # maximum grid fraction            
            Δts      = dx * δd / maximum(abs.(flow.V.x))  # constraint slip acceleration on fault
            Δtd      = δd * min(abs(dx / maximum(flow.V.x)), abs(dy / maximum(flow.V.y)))
            
            if COMPRESSIBLE
                ξ        = 0.2      # fraction to capture relaxation time scale
                ηvep     = 1.0      # TODO: change it?
                Δtvep    = ξ * ηvep / (compressibility.µ/ (1-compressibility.ν))
                @show Δt = min(Δts, Δtd, Δtvep)
            else 
                @show Δt = min(Δts, Δtd)
            end
        end



        it += 1
    end
    
    # Visualization
    if DO_VIZ
        gif(anim, "fault1D_injection_incompressible.gif", fps = 1)
    end

    # return effective pressure at final time
    return Array(flow.Pt - flow.Pf)'
end


# if isinteractive()
    fluid_injection(;t_tot_= 4000) # for reproducing fluid injection benchmark
# end
