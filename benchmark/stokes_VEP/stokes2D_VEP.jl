# Visco-elasto-plastic Stokes solver based on
# Stokes2D_vep.jl benchmark in: https://github.com/PTsolvers/Stokes2D_simpleVEP
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
else
        @init_parallel_stencil(Threads, Float64, 2)
end

const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
const DO_VIZ  = true


# Initialisation
using Plots, Plots.Measures, Printf, Statistics, LinearAlgebra
Dat = Float64  # Precision (double=Float64 or single=Float32)


# NOTE: set VISCO_ELASTIC_PREDICTOR to false for formulation as in Dal Zilio et al. 2022
const VISCO_ELASTIC_PREDICTOR = false   
const WITH_INERTIA       = false
const ADAPTIVE           = false
const COMPRESSIBLE       = false



# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])

# 2D Stokes routine
"""Stokes 2D solver with visco-elasto-plastic rheology
"""
@views function Stokes2D_vep()

        # MESH
        lx, ly        = 1.0, 1.0           # domain size
        nx, ny        = 121, 121             # numerical grid resolution
        @show dx, dy  = lx/(nx-1), ly/(ny-1)

        # PLASTICITY
        do_DP    = true              # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
        θ        = 30                 # define the friction angle 30°
        sinθ     = sind(θ)*do_DP     # sinus of the friction angle

        G0       = 1.0                # elastic shear modulus
        Gi       = G0/(8.0-6.0*do_DP) # elastic shear modulus perturbation
        ε0       = 1.0                # background strain-rate
        η0       = 1.0                # solid viscosity

        # Array initialisation
        Pt       = zeros(Dat, nx  ,ny  )
        ∇V       = zeros(Dat, nx  ,ny  )
        Vx       = zeros(Dat, nx+1,ny  )
        Vy       = zeros(Dat, nx  ,ny+1)
        ɛ̇xx_vis  = zeros(Dat, nx  ,ny  )
        ɛ̇yy_vis  = zeros(Dat, nx  ,ny  )
        ɛ̇xyv_vis = zeros(Dat, nx+1,ny+1)
        ɛ̇xx_ve   = zeros(Dat, nx  ,ny  )
        ɛ̇yy_ve   = zeros(Dat, nx  ,ny  )
        ɛ̇xy_ve   = zeros(Dat, nx  ,ny  )
        ɛ̇xyv_ve  = zeros(Dat, nx+1,ny+1)
        σxxʼ     = zeros(Dat, nx  ,ny  )
        σyyʼ     = zeros(Dat, nx  ,ny  )
        σxyʼ     = zeros(Dat, nx  ,ny  )
        σxyvʼ    = zeros(Dat, nx+1,ny+1)
        σxx_oʼ   = zeros(Dat, nx  ,ny  )
        σyy_oʼ   = zeros(Dat, nx  ,ny  )
        σxy_oʼ   = zeros(Dat, nx  ,ny  )
        σxyv_oʼ  = zeros(Dat, nx+1,ny+1)
        σII      = zeros(Dat, nx  ,ny  )
        ɛ̇II_plastic      = zeros(Dat, nx  ,ny  )

        # TODO: new!
        Vx_o     = zeros(Dat, nx+1,ny  )
        Vy_o     = zeros(Dat, nx  ,ny+1)
        V        = zeros(Dat, nx  ,ny  )     # slip rate
        τ0       =  zeros(Dat, nx  ,ny  )
        
        if VISCO_ELASTIC_PREDICTOR
                τ_yield = 1.6
        else
                τ_yield  =  zeros(Dat, nx  ,ny  )
        end
        
        Z        = zeros(Dat, nx  ,ny  )
        Zv       = zeros(Dat, nx+1,ny+1)
        F        = zeros(Dat, nx  ,ny  )
        Pla      = zeros(Dat, nx  ,ny  )
        χ        = zeros(Dat, nx  ,ny  )
        ηm       = zeros(Dat, nx  ,ny  )   # matrix viscosity
        η_vp     = zeros(Dat, nx  ,ny  )   # effective visco-plastic viscosity
        ηv_vp    = zeros(Dat, nx+1  ,ny+1  )   # effective visco-plastic viscosity
        

        # FIXME: formulation as in Dal Zilio et al. 2022
        f = 0.0   # static friction coeff
        c = 0.001   # cohesion
        γ = 1.0  # rate-stengthening exponent
        _γ = 1/γ  # power 
        λ  = -29  # porosity-weakening coefficient
        wh  = dx


        # PT specific
        dampV    = 4.0
        dampVx   = (1-dampV/nx)
        dampVy   = (1-dampV/ny)
        Vᵣ       = 4.0                  # iterative time step limiter original
        Ptᵣ      = 8.0                  # iterative time step limiter original
        ∂Q_∂σxxʼ  = zeros(Dat, nx  ,ny  )
        ∂Q_∂σyyʼ  = zeros(Dat, nx  ,ny  )
        ∂Q_∂σxyʼ  = zeros(Dat, nx  ,ny  )
        fᵛˣ      = zeros(Dat, nx-1,ny  )
        fᵛʸ      = zeros(Dat, nx  ,ny-1)
        gᵛˣ      = zeros(Dat, nx-1,ny  )
        gᵛʸ      = zeros(Dat, nx  ,ny-1)
        Δτₚᵗ     = zeros(Dat, nx  ,ny  )
        Δτᵥ₁     = zeros(Dat, nx-1,ny  )
        Δτᵥ₂     = zeros(Dat, nx  ,ny-1)


        # INITIAL CONDITIONS
        # physics
        # g       = 9.81998
        g       = 0.0
        ρt      = 0.1
        ρg      = fill(ρt*g, nx  ,ny  )
        G       = fill(G0, nx, ny)      # shear modulus [Pa]
        Gv      = fill(G0, nx+1, ny+1)

        # viscosity
        ηs       = fill(η0, nx, ny)
        ηsv      = fill(η0, nx+1, ny+1)
        η_e      = zeros(nx, ny)       # [Pa·s]
        η_ev     = zeros(nx+1, ny+1)
        η_ve     = ones(Dat, nx, ny)
        η_vep    = ones(Dat, nx, ny)
        η_vepv   = ones(Dat, nx+1, ny+1)


        xc, yc  = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
        xc, yc  = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
        xv, yv  = LinRange(0.0, lx, nx+1), LinRange(0.0, ly, ny+1)
        (Xvx,Yvx) = ([x for x=xv,y=yc], [y for x=xv,y=yc])
        (Xvy,Yvy) = ([x for x=xc,y=yv], [y for x=xc,y=yv])
        radc      = (xc.-lx./2).^2 .+ (yc'.-ly./2).^2
        radv      = (xv.-lx./2).^2 .+ (yv'.-ly./2).^2
        radi    = 0.01               # inclusion radius

        # shear modulus
        G[radc.<radi] .= Gi
        Gv[radv.<radi].= Gi

        # effective elastic viscosity
        @show Δt  = η0/G0/4.0 # assumes Maxwell time of 4
        _Δt       = inv(Δt)

        η_e            = Δt*G
        η_ev           = Δt*Gv
        η_ve          .= (1.0./η_e + 1.0./ηs).^-1

        # setting initial conditions for shearing
        Vx            .=   ε0.*Xvx
        Vy            .= .-ε0.*Yvy


        # VISUALIZATION
        if DO_VIZ
                default(size=(2000,1000), margin=1mm)
                ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
                println("Animation directofᵛʸ: $(anim.dir)")
        end
        

        # Time loop
        ε         = 1e-7               # nonlinear tolerence
        iterMax   = 5e3               # max number of iters
        nout      = 50                # check frequency
        nt        = 15                 # number of time steps
        t=0.0; evo_t=[]; evo_σxxʼ=[]; niter = 0
        for it = 1:nt
                iter=1; err=2*ε; err_evo1=[]; err_evo2=[]

                @. Vx_o    = Vx
                @. Vy_o    = Vy
                @. σxx_oʼ  = σxxʼ
                @. σyy_oʼ  = σyyʼ
                   σxy_oʼ .= av(σxyvʼ)
                @. σxyv_oʼ = σxyvʼ
                @. χ       = 0.0


                # UNCOMMENT FOR ADAPTIVE TIMESTEPPING!
                # η_e is defined as ηe = GΔt
                # @. η_e       = Δt*G0
                # @. η_ev      = Δt*G0

                # η_e[radc.<radi]  .= Δt*Gi
                # η_ev[radv.<radi] .= Δt*Gi
        

                local itg
                
                while (err>ε && iter<=iterMax)
                        # divergence - pressure
                        ∇V     = diff(Vx, dims=1)./dx .+ diff(Vy, dims=2)./dy
                        @. Pt  = Pt - Δτₚᵗ * ∇V

                        ###############################
 
                        # VISCO-ELASTO-PLASTIC RHEOLOGY
                        # ɛ̇ᵢⱼʼ = [ɛ̇ᵢⱼʼ]ᵥ + [ɛ̇ᵢⱼʼ]ₑ + [ɛ̇ᵢⱼʼ]ₚ

                        # i). viscous 
                        # [ɛ̇ᵢⱼʼ]ᵥ = 1/2μˢ ·σᵢⱼ' = (1/2 (∇ᵢvⱼˢ + ∇ⱼvᵢˢ) - 1/3 δᵢⱼ ∇ₖvₖˢ)
                        ɛ̇xx_vis                   .= diff(Vx, dims=1)./dx .- 1.0/3.0 * ∇V
                        ɛ̇yy_vis                   .= diff(Vy, dims=2)./dy .- 1.0/3.0 * ∇V
                        ɛ̇xyv_vis[2:end-1,2:end-1] .= 0.5.*(diff(Vx[2:end-1,:], dims=2)./dy .+ diff(Vy[:,2:end-1], dims=1)./dx)

                        # ii). elastic
                        # [ɛ̇ᵢⱼʼ]ₑ = 1/(2·ηe) D̃τᵢⱼ/D̃t
                        # ηe [Pa·s]
                        # compute visco-elasticity factor
                        if VISCO_ELASTIC_PREDICTOR
                                
                                @. ɛ̇xx_ve   = ɛ̇xx_vis  + σxx_oʼ /2.0 / η_e
                                @. ɛ̇yy_ve   = ɛ̇yy_vis  + σyy_oʼ /2.0 / η_e
                                @. ɛ̇xyv_ve  = ɛ̇xyv_vis + σxyv_oʼ/2.0 / η_ev
                                ɛ̇xy_ve   .= av(ɛ̇xyv_vis)  .+ σxy_oʼ ./ 2.0 ./η_e
                                
                                
                                # trial stress i) + ii) (stress containing viscous and elastic components)
                                @. σxxʼ     = 2.0 * η_ve * ɛ̇xx_ve
                                @. σyyʼ     = 2.0 * η_ve * ɛ̇yy_ve
                                @. σxyʼ     = 2.0 * η_ve * ɛ̇xy_ve
                                
                                # second stress invariant i) + ii) σII = √(1/2 σᵢⱼ'²)
                                @. σII     = sqrt(0.5 * (σxxʼ^2 + σyyʼ^2) + σxyʼ^2)
                                
                                # second invariant of plastic strain rate
                                @. ɛ̇II_plastic    = sqrt(0.5*(ɛ̇xx_ve^2 + ɛ̇yy_ve^2) + ɛ̇xy_ve^2)


                        else       
 
                                # using formulation as in Dal Zilio et al. 2022
                                @. Z = (η_e) / (η_e + ηs)

                                @. σxxʼ    = 2.0 * ηs * Z * ɛ̇xx_vis + σxx_oʼ * (1 - Z)
                                @. σyyʼ    = 2.0 * ηs * Z * ɛ̇yy_vis + σyy_oʼ * (1 - Z)
                                σxyʼ   .= 2.0 .* ηs .* Z.* av(ɛ̇xyv_vis) .+ σxy_oʼ .* (1 .- 0.5 .* Z)
                                @. σxyvʼ   = 2.0 * ηsv * Zv * ɛ̇xyv_vis + σxyv_oʼ * (1 - 0.5 * Zv)

                                @. Zv = (η_ev) / (η_ev + ηsv)


                                # second stress invariant i) + ii) σII = √(1/2 σᵢⱼ'²)
                                @. σII     = sqrt(0.5 * (σxxʼ^2 + σyyʼ^2) + σxyʼ^2)

                                # second strain rate invariant of the deviatoric plastic strain rate
                                # τ0 = c + f * (pt - pf)  TODO: change this for two-phase flow
                                # @. τ0             = 1.0 + f * Pt
                                # @. ɛ̇II_plastic    = ε0 * (σII / τ0)^_γ
                                # @. τ_yield        = τ0 * (ɛ̇II_plastic / ε0)^γ

                                @. τ_yield        = 1.6
                                @. ɛ̇II_plastic    = sqrt(0.5*(ɛ̇xx_ve^2 + ɛ̇yy_ve^2) + ɛ̇xy_ve^2)
                        end

                        # compute slip rate
                        @. V = 2 * ɛ̇II_plastic * wh


                        # iii). plasticity
                        # [ɛ̇ᵢⱼʼ]ₚ = χ·∂Q/∂σij'
                        #         = χ·σij'/σII                         

                        # yield function
                        # F = τII - τyield - Pt sinθ > 0 <=> set Pla to true if τII > τyield
                        @. F       = σII - τ_yield - Pt * sinθ
                        @. Pla     = F > 0.0
                        
                        if VISCO_ELASTIC_PREDICTOR
                                # Plastic multiplier
                                @. χ       = Pla * F / η_ve   # Duretz
        
                                @. ∂Q_∂σxxʼ = 0.5 * σxxʼ/σII
                                @. ∂Q_∂σyyʼ = 0.5 * σyyʼ/σII
                                @. ∂Q_∂σxyʼ =       σxyʼ/σII

                                # plastic corrections
                                @. σxxʼ    = 2.0 * η_ve * (ɛ̇xx_ve - χ * ∂Q_∂σxxʼ)
                                @. σyyʼ    = 2.0 * η_ve * (ɛ̇yy_ve - χ * ∂Q_∂σyyʼ)
                                @. σxyʼ    = 2.0 * η_ve * (ɛ̇xy_ve - 0.5 * χ * ∂Q_∂σxyʼ)
                                @. σII     = sqrt(0.5 * (σxxʼ^2 + σyyʼ^2) + σxyʼ^2)

                                @. η_vep  = σII /2.0 /ɛ̇II_plastic
                                η_vepv[2:end-1,2:end-1] .= av(η_vep)
                                @. η_vepv[1,:]              = η_vepv[2,:]
                                @. η_vepv[end,:]            = η_vepv[end-1,:]
                                @. η_vepv[:,1]              = η_vepv[:,2]
                                @. η_vepv[:,end]            = η_vepv[:,end-1]

                                @. σxyvʼ   = 2.0 * η_vepv * ɛ̇xyv_ve # (Duretz)
                        else
                                
                                # compute matrix viscosity
                                # @. ηm = ηs * exp(λ * 0.01)   # change it to ηm = ηs * exp(λɸ) with ɸ the porosity
                                @. ηm = ηs

                                # conditional assignment of effective visco-plastic viscosity
                                @. η_vp = (1.0 - Pla) * ηm + Pla * ηm * σII/(2.0ηm * ɛ̇II_plastic + σII)
                                @. Z = (η_e) / (η_e + η_vp)

                                # corrected stress
                                @. σxxʼ    = 2.0 * η_vp * Z * ɛ̇xx_vis + σxx_oʼ * (1 - Z)
                                @. σyyʼ    = 2.0 * η_vp * Z * ɛ̇yy_vis + σyy_oʼ * (1 - Z)
                                   σxyʼ   .= 2.0 .* η_vp .* Z.* av(ɛ̇xyv_vis) .+ σxy_oʼ .* (1 .- 0.5 .* Z)


                                # conditional assignment of effective visco-plastic viscosity
                                @. η_vep  = Pla * ηm + (1 - Pla) * ηm * (σII / (2.0 * ηm * ɛ̇II_plastic + σII))

                                # compute for vertices
                                η_vepv[2:end-1,2:end-1]    .= av(η_vep)
                                @. η_vepv[1,:]              = η_vepv[2,:]
                                @. η_vepv[end,:]            = η_vepv[end-1,:]
                                @. η_vepv[:,1]              = η_vepv[:,2]
                                @. η_vepv[:,end]            = η_vepv[:,end-1]


                                # compute for vertices
                                @. Zv = (η_ev) / (η_ev + η_vepv)
                                @. σxyvʼ   = 2.0 * η_vepv * Zv * ɛ̇xyv_vis + σxyv_oʼ * (1 - 0.5 * Zv)


                        end


                        ###############################
                        
                        # PT timestep
                        Δτᵥ₁   .= min(dx,dy)^2.0./av_xa(η_vep)./4.1./Vᵣ
                        Δτᵥ₂   .= min(dx,dy)^2.0./av_ya(η_vep)./4.1./Vᵣ
                        Δτₚᵗ   .= 4.1.*η_vep./max(nx,ny)./Ptᵣ
                        
                        # VELOCITIES
                        # NOTE: we use the geological coordinates here!
                        if WITH_INERTIA
                                fᵛˣ    .= (diff(σxxʼ, dims=1) .-diff(Pt, dims=1))./dx .+ diff(σxyvʼ[2:end-1,:], dims=2)./dy .- ρt .* (Vx[2:end-1,:] .- Vx_o[2:end-1,:]) ./ Δt
                                fᵛʸ    .= (diff(σyyʼ, dims=2) .-diff(Pt, dims=2))./dy .+ diff(σxyvʼ[:,2:end-1], dims=1)./dx .+ av_ya(ρg) .- ρt .* (Vy[:,2:end-1] .- Vy_o[:,2:end-1]) ./ Δt
                        else
                                fᵛˣ    .= (diff(σxxʼ, dims=1) .-diff(Pt, dims=1))./dx .+ diff(σxyvʼ[2:end-1,:], dims=2)./dy
                                fᵛʸ    .= (diff(σyyʼ, dims=2) .-diff(Pt, dims=2))./dy .+ diff(σxyvʼ[:,2:end-1], dims=1)./dx .+ av_ya(ρg)
                        end

                        @. gᵛˣ    = gᵛˣ * dampVx + fᵛˣ
                        @. gᵛʸ    = gᵛʸ * dampVy + fᵛʸ
                        @. Vx[2:end-1,:] = Vx[2:end-1,:] + gᵛˣ * Δτᵥ₁
                        @. Vy[:,2:end-1] = Vy[:,2:end-1] + gᵛʸ * Δτᵥ₂

                        
                        # CONVERGENCE CHECK
                        if mod(iter, nout)==0
                                norm_fᵛˣ = norm(fᵛˣ)/length(fᵛˣ); norm_fᵛʸ = norm(fᵛʸ)/length(fᵛʸ); norm_∇V = norm(∇V)/length(∇V)
                                err = maximum([norm_fᵛˣ, norm_fᵛʸ, norm_∇V])
                                push!(err_evo1, err); push!(err_evo2, itg)
                                @printf("it = %d, iter = %d, err = %1.2e norm[fᵛˣ=%1.2e, fᵛʸ=%1.2e, ∇V=%1.2e] (F = %1.2e) \n", it, itg, err, norm_fᵛˣ, norm_fᵛʸ, norm_∇V, maximum(F))
                        end
                        iter+=1; itg=iter; niter += 1
                end


                t += Δt
                push!(evo_t, t); push!(evo_σxxʼ, maximum(σxxʼ))


                # DEBUG
                @show extrema(ɛ̇II_plastic)
                @show extrema(τ_yield)
                @show extrema(σII)

                @show sum(Pla)


                # ADAPTIVE TIME STEPPING
                if ADAPTIVE
                        # δd       = 1e-5     # maximum grid fraction            
                        δd       = 5.0     # maximum grid fraction            

                        Δts      = dx * δd / maximum(abs.(Vx))  # constraint slip acceleration on fault
                        Δtd      = δd * min(abs(dx / maximum(Vx)), abs(dy / maximum(Vy)))
                        
                        if COMPRESSIBLE
                            ξ        = 0.2      # fraction to capture relaxation time scale
                            ηvep     = 1.0      # TODO: change it?
                            Δtvep    = ξ * ηvep / (compressibility.µ/ (1-compressibility.ν))
                            @show Δt = min(Δts, Δtd, Δtvep)
                        else 
                            @show Δt = min(Δts, Δtd)
                        end
                end
            


                if DO_VIZ
                        # Plotting
                        p1 = heatmap(xv, yc, Vx' , aspect_ratio=1, xlims=(0, lx), ylims=(dy/2, ly-dy/2), c=:inferno, title="Vx")
                        p2 = heatmap(xc, yv, Vy' , aspect_ratio=1, xlims=(dx/2, lx-dx/2), ylims=(0, ly), c=:inferno, title="Vy")
                        p3 = heatmap(xc, yc, η_vep' , aspect_ratio=1, xlims=(dx/2, lx-dx/2), ylims=(0, ly), c=:inferno, title="η_vep")
                        p4 = heatmap(xc, yc, V' , aspect_ratio=1, xlims=(dx/2, lx-dx/2), ylims=(0, ly), c=:inferno, title="Slip rate V")
                        p5 = heatmap(xc, yc, σII' , aspect_ratio=1, xlims=(dx/2, lx-dx/2), ylims=(0, ly), c=:inferno, title="τii")
                        p6 = plot(evo_t, evo_σxxʼ , legend=false, xlabel="time", ylabel="max(τxx)", linewidth=0, markershape=:circle, framestyle=:box, markersize=3)
                                plot!(evo_t, 2.0.*ε0.*η0.*(1.0.-exp.(.-evo_t.*G0./η0)), linewidth=2.0) # analytical solution for VE loading
                                plot!(evo_t, 2.0.*ε0.*η0.*ones(size(evo_t)), linewidth=2.0)            # viscous flow stress
                                if !do_DP plot!(evo_t, τ_yield*ones(size(evo_t)), linewidth=2.0) end        # von Mises yield stress
                        display(plot(p1, p2, p3, p4, p5, p6; layout = (2,3))); frame(anim)
                end
        end # end of physical loop

        # Visualization
        if DO_VIZ
                gif(anim, "stokes2D_VEP_inertia.gif", fps = 3)
        end

        println(niter) # monitoring convergence speed
        return
end

Stokes2D_vep()
