# Two-phase flow compressible solver
# this source file contains update routines needed for the compressible solver 

@inbounds @parallel function assign!(𝝫_o::Data.Array, ∇V_o::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, 𝝫::Data.Array, ∇V::Data.Array,  Pt::Data.Array, Pf::Data.Array)
    @all(𝝫_o)   = @all(𝝫)
    @all(∇V_o)  = @all(∇V)
    @all(Pt_o)  = @all(Pt)
    @all(Pf_o)  = @all(Pf)
    return
end


@inbounds @parallel function compute_Kd!(𝗞d::Data.Array, 𝗞ɸ::Data.Array, 𝝫::Data.Array, _Ks::Data.Number, µ::Data.Number)


    # compute effective bulk modulus for the pores
    # Kɸ = 2m/(1+m)µ/ɸ =  µ/ɸ (m=1)
    @all(𝗞ɸ) = µ / @all(𝝫)

    # compute drained bulk modulus
    # Kd = (1-ɸ)(1/Kɸ + 1/Ks)⁻¹
    @all(𝗞d) = (1.0 -@all(𝝫)) / (1.0 /@all(𝗞ɸ) + _Ks)

    return nothing

end


@inbounds @parallel function compute_ɑ!(𝝰::Data.Array, 𝝱d::Data.Array, 𝗞ɸ::Data.Array, 𝝫::Data.Array, βs::Data.Number)

    # compute solid skeleton compressibility
    # 𝝱d = (1+ βs·Kɸ)/(Kɸ-Kɸ·ɸ) = (1+ βs·Kɸ)/Kɸ/(1-ɸ)
    @all(𝝱d) = (1.0 + βs * @all(𝗞ɸ)) / @all(𝗞ɸ) / (1-@all(𝝫))
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

    length_RVy  = length(flow.R.Vy) 
    length_RPf  = length(flow.R.Pf)
    
    _C          = inv(rheology.C)
    _ɸ0         = inv(rheology.ɸ0)

    # for the compressibility
    _Ks         = inv(comp.Ks)


    @parallel assign!(flow.𝝫_o, flow.∇V_o, comp.Pt_o, comp.Pf_o, flow.𝝫, flow.∇V, flow.Pt, flow.Pf)
    err=2*ε; iter=1; niter=0
    
    while err > ε && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end

        # involve the compressible TPF solver
        @parallel compute_params_∇!(flow.𝞰ɸ, flow.𝗞ɸ_µᶠ, flow.𝞀g, flow.∇V, flow.∇qD, flow.𝝫, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.k0, _ɸ0, rheology.nₖ, rheology.θ_e, rheology.θ_k, flow.ρfg, flow.ρsg, flow.ρgBG, _dx, _dy)
        
        #  parameters computation for compressible case!
        @parallel compute_Kd!(comp.𝗞d, comp.𝗞ɸ, flow.𝝫, _Ks, comp.µ)
        @parallel compute_ɑ!(comp.𝝰, comp.𝝱d, comp.𝗞ɸ, flow.𝝫, comp.βs)
        @parallel compute_B!(comp.𝗕, flow.𝝫, comp.𝝱d, comp.βs, comp.βf)
        
        @parallel compute_residual_mass_law!(pt.dτPf, flow.R.Pt, flow.R.Pf, flow.𝗞ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, flow.𝝫, comp.𝗞d, comp.𝝰, comp.Pt_o, comp.Pf_o, comp.𝗕, pt.Pfsc, pt.Pfdmp, min_dxy2, _dx, _dy, Δt)

        apply_free_slip!(freeslip, pt.dτPf, nx, ny)
        
        @parallel compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.dτPf, pt.dτPt)
        @parallel compute_tensor!(flow.𝞃.xx, flow.𝞃.yy, flow.𝞃.xy, flow.V.x, flow.V.y, flow.∇V, flow.R.Pt, rheology.μˢ, pt.βₚₜ, _dx, _dy)
        
        # velocity update from the conservation of momentum equations
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.dVxdτ, pt.dVydτ, flow.𝞃.xx, flow.𝞃.yy, flow.𝞃.xy, flow.Pt, flow.𝞀g, pt.dampX, pt.dampY, _dx, _dy)
        @parallel compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.dVxdτ, pt.dVydτ, flow.𝗞ɸ_µᶠ, flow.Pf, pt.dτV, flow.ρfg, flow.ρgBG, _dx, _dy)
        apply_free_slip!(freeslip, flow.V.x, flow.V.y, nx+1, ny+1)
        apply_free_slip!(freeslip, flow.qD.x, flow.qD.y, nx+1, ny+1)
    
        # update the porosity
        @parallel compute_porosity!(flow.𝝫, flow.𝝫_o, flow.∇V, flow.∇V_o, CN, Δt)
        if mod(iter,nout)==0
            global norm_RVy, norm_RPf
            norm_RVy = norm(flow.R.Vy)/length_RVy; norm_RPf = norm(flow.R.Pf)/length_RPf; err = max(norm_RVy, norm_RPf)
            # @printf("iter = %d, err = %1.3e [norm_flow.R.Vy=%1.3e, norm_flow.R.Pf=%1.3e] \n", iter, err, norm_flow.R.Vy, norm_flow.R.Pf)
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





