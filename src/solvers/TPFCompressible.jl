# Two-phase flow compressible solver
# this source file contains update routines needed for the compressible solver 

@inbounds @parallel function assign!(𝝫_o::Data.Array, ∇V_o::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, 𝝫::Data.Array, ∇V::Data.Array,  Pt::Data.Array, Pf::Data.Array)
    @all(𝝫_o)  = @all(𝝫)
    @all(∇V_o)  = @all(∇V)

    # use the value from last physical iteration throughout PT iterations
    @all(Pt_o)  = @all(Pt)
    @all(Pf_o)  = @all(Pf)
    return
end


@inbounds @parallel function compute_Kd!(𝗞d::Data.Array, 𝗞ɸ::Data.Array, 𝝫::Data.Array, _Ks::Data.Number, µ::Data.Number, ν::Data.Number)

    # compute effective bulk modulus for the pores
    # Kɸ = 2m/(1+m)µ*/ɸ =  µ/(1-ν)/ɸ (m=1)
    @all(𝗞ɸ) = µ / (1.0 - ν) / @all(𝝫)

    # compute drained bulk modulus
    # Kd = (1-ɸ)(1/Kɸ + 1/Ks)⁻¹
    @all(𝗞d) = (1.0 - @all(𝝫)) / (1.0 /@all(𝗞ɸ) + _Ks)

    return nothing

end


@inbounds @parallel function compute_ɑ!(𝝰::Data.Array, 𝝱d::Data.Array, 𝗞ɸ::Data.Array, 𝝫::Data.Array, βs::Data.Number)

    # compute solid skeleton compressibility
    # 𝝱d = (1+ βs·Kɸ)/(Kɸ-Kɸ·ɸ) = (1+ βs·Kɸ)/Kɸ/(1-ɸ)
    @all(𝝱d) = (1.0 + βs * @all(𝗞ɸ)) / @all(𝗞ɸ) / (1.0 - @all(𝝫))

    # compute Biot Willis coefficient
    @all(𝝰)  = 1.0 - βs / @all(𝝱d)
    
    return nothing
end


@inbounds @parallel function compute_B!(B::Data.Array, 𝝫::Data.Array, 𝝱d::Data.Array, βs::Data.Number, βf::Data.Number)

    # compute skempton coefficient
    # B = (𝝱d - βs)/(𝝱d - βs + ɸ(βf - βs))
    @all(B) = (@all(𝝱d) - βs) / (@all(𝝱d) - βs + @all(𝝫) * (βf - βs))

    return nothing
end



@inbounds function solve!(

    #==== Governing flow ====#    
    flow::TwoPhaseFlow2D,

    # new for compressible flow
    comp::Compressibility,

    #==== Rheology ====#
    rheology::ViscousRheology,

    #==== mesh properties ====#
    mesh::PTGrid,
    
    #====  boundary condition ====#
    freeslip, 
    
    #====  iteration specific ====#
    pt::PTCoeff,

    Δt,
    it;
    
    ε = 1e-5, 
    iterMax = 5e3,
    nout = 200, 
    CN = 0.5
    )


    # unpack
    nx, ny = mesh.ni
    dx, dy = mesh.di

    # precomputation
    _dx, _dy    = inv.(mesh.di)
    min_dxy2    = min(dx,dy)^2

    length_Rx  = length(flow.R.Vx)
    length_Ry  = length(flow.R.Vy)
    length_RPf = length(flow.R.Pf)
    length_RPt = length(flow.R.Pt)
    
    _C          = inv(rheology.C)
    _ɸ0         = inv(rheology.ɸ0)

    # for the compressibility
    _Ks         = inv(comp.Ks)


    @parallel assign!(flow.𝝫_o, flow.∇V_o, comp.Pt_o, comp.Pf_o, flow.𝝫, flow.∇V, flow.Pt, flow.Pf)
    err=2*ε; iter=1; niter=0
    
    while err > ε && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end

        # involve the compressible TPF solver
        @parallel compute_params_∇!(flow.𝞰ɸ, flow.𝐤ɸ_µᶠ, flow.𝞀g, flow.∇V, flow.∇qD, flow.𝝫, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.k0, _ɸ0, rheology.nₖ, rheology.θ_e, rheology.θ_k, flow.ρfg, flow.ρsg, flow.ρgBG, _dx, _dy)
        
        #  parameters computation for compressible case!
        @parallel compute_Kd!(comp.𝗞d, comp.𝗞ɸ, flow.𝝫, _Ks, comp.µ, comp.ν)
        @parallel compute_ɑ!(comp.𝝰, comp.𝝱d, comp.𝗞ɸ, flow.𝝫, comp.βs)
        @parallel compute_B!(comp.𝗕, flow.𝝫, comp.𝝱d, comp.βs, comp.βf)
        
        @parallel compute_residual_mass_law!(pt.Δτₚᶠ, flow.R.Pt, flow.R.Pf, flow.𝐤ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, flow.𝝫, comp.𝗞d, comp.𝝰, comp.Pt_o, comp.Pf_o, comp.𝗕, pt.Pfᵣ, pt.dampPf, min_dxy2, Δt)

        apply_free_slip!(freeslip, pt.Δτₚᶠ, nx, ny)
        
        @parallel compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.Δτₚᶠ, pt.Δτₚᵗ)
        @parallel compute_tensor!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, rheology.μˢ, pt.ηb, _dx, _dy)
        
        # velocity update from the conservation of momentum equations
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.gᵛˣ, pt.gᵛʸ, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, pt.dampVx, pt.dampVy, _dx, _dy)
        @parallel compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.gᵛˣ, pt.gᵛʸ, flow.𝐤ɸ_µᶠ, flow.Pf, pt.Δτᵥ, flow.ρfg, flow.ρgBG, _dx, _dy)
        
        apply_free_slip!(freeslip, flow.V.x, flow.V.y, nx+1, ny+1)
        apply_free_slip!(freeslip, flow.qD.x, flow.qD.y, nx+1, ny+1)
    
        # update the porosity
        @parallel compute_porosity!(flow.𝝫, flow.𝝫_o, flow.∇V, flow.∇V_o, CN, Δt)
        if mod(iter,nout)==0
            global norm_Rx, norm_Ry, norm_RPf, norm_RPt
            norm_Rx  = norm(flow.R.Vx)/length_Rx
            norm_Ry  = norm(flow.R.Vy)/length_Ry
            norm_RPf = norm(flow.R.Pf)/length_RPf
            norm_RPt = norm(flow.R.Pt)/length_RPt
            
            err = max(norm_Rx, norm_Ry, norm_RPf, norm_RPt)

            @printf("iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_RPf=%1.3e, norm_RPt=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_RPf, norm_RPt)
        end
        iter+=1; niter+=1
    end

    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (8*2)/1e9*nx*ny*sizeof(eltype(flow.𝝫))  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(niter-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))
    
end



#=======temporary solver function========#
@inbounds function solve!(

    StrikeSlip_2D::String,

    h_index::Integer,

    #==== Governing flow ====#    
    flow::TwoPhaseFlow2D,

    # new for compressible flow
    comp::Compressibility,

    #==== Rheology ====#
    rheology::ViscousRheology,

    #==== mesh properties ====#
    mesh::PTGrid,
    
    #====  boundary condition ====#
    freeslip, 
    
    #====  iteration specific ====#
    pt::PTCoeff,

    Δt,
    it;

    # TODO: check if correct values used!
    p₀f     = 5.0e6,            # initial fluid pressure 5 MPa
    Δpf     = 5.0e6,            # constant amount of fluid to be injected 5 MPa
    Vpl     = 1.9977e-9,       # loading rate [m/s] = 6.3 cm/yr
    p⁻      = -1.0e-12,        # BC top - outward flux [m/s]
    p⁺      = 1.0e-12,         # BC bottom - inward flux [m/s]
    Peff    = 3.0e+7,          # constant effective pressure [Pa] -> 30MPa 

    ε       = 1e-5,
    # ε       = 1e-20,
    iterMax = 5e5,          # 5e3 for porosity wave, 5e5 previously
    # iterMax = 5e4,          # 5e3 for porosity wave, 5e5 previously
    # nout    = 200,
    nout    = 1,
    CN      = 0.5
    )


    # unpack
    # nx, ny = mesh.ni
    ny, nx     = mesh.ni
    rows, cols = mesh.ni  # FIXME: fix for later!
    dx, dy     = mesh.di

    # precomputation
    _dx, _dy    = inv.(mesh.di)
    min_dxy2    = min(dx,dy)^2

    length_Rx  = length(flow.R.Vx)
    length_Ry  = length(flow.R.Vy)
    length_RPf = length(flow.R.Pf)
    length_RPt = length(flow.R.Pt)
    
    _C          = inv(rheology.C)
    _ɸ0         = inv(rheology.ɸ0)

    # for the compressibility
    _Ks         = inv(comp.Ks)


    @parallel assign!(flow.𝝫_o, flow.∇V_o, comp.Pt_o, comp.Pf_o, flow.𝝫, flow.∇V, flow.Pt, flow.Pf)
    err=2*ε; iter=1; niter=0
    
    while err > ε && iter <= iterMax
        # if (iter==11)  global wtime0 = Base.time()  end
        if (iter==1)  global wtime0 = Base.time()  end

        # @parallel compute_params_∇!(flow.𝞰ɸ, flow.𝐤ɸ_µᶠ, flow.𝞀g, flow.∇V, flow.∇qD, flow.𝝫, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.k0, _ɸ0, rheology.nₖ, rheology.θ_e, rheology.θ_k, flow.ρfg, flow.ρsg, flow.ρgBG, _dx, _dy)

        # not updating flow.𝐤ɸ_µᶠ
        @parallel compute_params_∇!(flow.𝞰ɸ, flow.∇V, flow.∇qD, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.θ_e,  _dx, _dy)


        # parameters computation for compressible case!
        @parallel compute_Kd!(comp.𝗞d, comp.𝗞ɸ, flow.𝝫, _Ks, comp.µ, comp.ν)
        @parallel compute_ɑ!(comp.𝝰, comp.𝝱d, comp.𝗞ɸ, flow.𝝫, comp.βs)
        @parallel compute_B!(comp.𝗕, flow.𝝫, comp.𝝱d, comp.βs, comp.βf)
        
        # pressure update from the conservation of mass flow
        @parallel compute_residual_mass_law!(pt.Δτₚᶠ, flow.R.Pt, flow.R.Pf, flow.𝐤ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, flow.𝝫, comp.𝗞d, comp.𝝰, comp.Pt_o, comp.Pf_o, comp.𝗕, pt.Pfᵣ, pt.dampPf, min_dxy2, Δt)
        
        # FIXME: monday experiment, trying to bring up pressure -> experiment result, seems like these are not needed!
        # @parallel (1:cols) free_slip_x!(pt.Δτₚᶠ)
        # @parallel (1:rows) free_slip_y!(pt.Δτₚᶠ)
        
        @parallel compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.Δτₚᶠ, pt.Δτₚᵗ)
        @parallel compute_tensor!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, rheology.μˢ, pt.ηb, _dx, _dy)
        
        # velocity update from the conservation of momentum equations
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.gᵛˣ, pt.gᵛʸ, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, pt.dampVx, pt.dampVy, _dx, _dy)
        @parallel compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.gᵛˣ, pt.gᵛʸ, flow.𝐤ɸ_µᶠ, flow.Pf, pt.Δτᵥ, flow.ρfg, flow.ρgBG, _dx, _dy)
        
        
        # left/right boundary
        # @parallel (1:ny+1)   free_slip_y!(flow.V.x)
        # @parallel (1:ny)     free_slip_y!(flow.V.y)
        # @parallel (1:rows)   free_slip_y!(flow.qD.y)

        # top & bottom boundary
        # @parallel (1:cols)   dirichlet_x!(flow.V.x, 0.5 * Vpl, -0.5 * Vpl)
        @parallel (1:cols)       dirichlet_x!(flow.V.x, 0.0, 0.0)
        @parallel (1:rows+1)     dirichlet_y!(flow.V.x, 0.0, 0.0)
        @parallel (1:cols+1)     dirichlet_x!(flow.V.y, 0.0, 0.0)
        @parallel (1:rows)       dirichlet_y!(flow.V.y, 0.0, 0.0)

        # @parallel (1:nx)     constant_flux_x!(flow.qD.y, p⁻, p⁺)
        # @parallel (1:cols)   constant_effective_pressure_x!(flow.Pt, flow.Pf, Peff)
        # @parallel (1:cols)   constant_effective_pressure_x!(flow.Pt, flow.Pf, Peff)
        @parallel (1:cols) constant_effective_pressure_x!(flow.Pf, flow.Pt, Peff)    


        # used for fluid injection benchmark! Otherwise not!
        flow.Pf[h_index, 1] = p₀f + Δpf      # constant fluid injection to the leftmost injection point on the fault

        if mod(iter,nout)==0
            global norm_Rx, norm_Ry, norm_RPf, norm_RPt
            norm_Rx  = norm(flow.R.Vx)/length_Rx
            norm_Ry  = norm(flow.R.Vy)/length_Ry
            norm_RPf = norm(flow.R.Pf)/length_RPf
            norm_RPt = norm(flow.R.Pt)/length_RPt
            
            err = max(norm_Rx, norm_Ry, norm_RPf, norm_RPt)


            # FIXME: recover this!
            # if mod(iter,nout*100) == 0
                @printf("iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_RPf=%1.3e, norm_RPt=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_RPf, norm_RPt)
            # end

        end


        iter+=1; niter+=1
    end

    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (8*2)/1e9*nx*ny*sizeof(eltype(flow.𝝫))   # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(niter-10)                         # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                           # Effective memory throughput [GB/s]
    @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))
    
end
