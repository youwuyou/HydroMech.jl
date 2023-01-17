# Two-phase flow incompressible solver
# this source file contains update routines needed for the incompressible solver 

@inbounds @parallel function assign!(𝝫_o::Data.Array, ∇V_o::Data.Array, 𝝫::Data.Array, ∇V::Data.Array)
    @all(𝝫_o) = @all(𝝫)
    @all(∇V_o)  = @all(∇V)
    return
end


@inbounds function solve!(

    #==== Governing flow ====#    
    flow::TwoPhaseFlow2D,

    #==== Rheology ====#
    rheology::ViscousRheology,  
    
    #==== mesh properties ====#
    mesh::PTGrid,   # ni, di
    
    #====  boundary condition ====#
    freeslip,
    
    #====  iteration specific ====#
    pt::PTCoeff,
    Δt, 
    it;
    
    ε       = 1e-5,      # nonlinear tolerance 
    iterMax = 5e3,       # max nonlinear iterations
    nout    = 200,       # error checking frequency 
    CN      = 0.5        # Crank-Nicolson CN=0.5, Backward Euler CN=0.0
)

    # unpack
    nx, ny = mesh.ni
    dx, dy = mesh.di

    # precomputation
    _dx, _dy   = inv.(mesh.di)
    _ɸ0        = inv(rheology.ɸ0)
    length_Ry  = length(flow.R.Vy)
    length_RPf = length(flow.R.Pf)
    min_dxy2   = min(dx,dy)^2
    _C         = inv(rheology.C)


    @parallel assign!(flow.𝝫_o, flow.∇V_o, flow.𝝫, flow.∇V)
    err=2*ε; iter=1; niter=0
    
    while err > ε && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end

        # involve the incompressible TPF solver
        @parallel compute_params_∇!(flow.𝞰ɸ, flow.𝗞ɸ_µᶠ, flow.𝞀g, flow.∇V, flow.∇qD, flow.𝝫, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.k0, _ɸ0, rheology.nₖ, rheology.θ_e, rheology.θ_k, flow.ρfg, flow.ρsg, flow.ρgBG, _dx, _dy)

        # pressure update from the conservation of mass flow
        @parallel compute_residual_mass_law!(pt.dτPt, pt.dτPf, flow.R.Pt, flow.R.Pf, flow.𝗞ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, flow.𝝫, pt.Pfsc, pt.Pfdmp, min_dxy2, _dx, _dy)
        apply_free_slip!(freeslip, pt.dτPf, nx, ny)
        @parallel compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.dτPf, pt.dτPt)
        @parallel compute_tensor!(flow.𝞃.xx, flow.𝞃.yy, flow.𝞃.xy, flow.V.x, flow.V.y,  flow.∇V, flow.R.Pt, rheology.μˢ, pt.βₚₜ, _dx, _dy)

    
        # velocity update from the conservation of momentum flow
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.dVxdτ, pt.dVydτ, flow.𝞃.xx, flow.𝞃.yy, flow.𝞃.xy, flow.Pt, flow.𝞀g, pt.dampX, pt.dampY, _dx, _dy)
        @parallel compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.dVxdτ, pt.dVydτ, flow.𝗞ɸ_µᶠ, flow.Pf, pt.dτV, flow.ρfg, flow.ρgBG, _dx, _dy)
        apply_free_slip!(freeslip, flow.V.x, flow.V.y, nx+1, ny+1)
        apply_free_slip!(freeslip, flow.qD.x, flow.qD.y, nx+1, ny+1)
    
        # update the porosity
        @parallel compute_porosity!(flow.𝝫, flow.𝝫_o, flow.∇V, flow.∇V_o, CN, Δt)


        if mod(iter,nout)==0
            global norm_Ry, norm_RPf
            norm_Ry = norm(flow.R.Vy)/length_Ry; norm_RPf = norm(flow.R.Pf)/length_RPf; err = max(norm_Ry, norm_RPf)
            # @printf("iter = %d, err = %1.3e [norm_Ry=%1.3e, norm_RPf=%1.3e] \n", iter, err, norm_Ry, norm_RPf)
        end
        iter+=1; niter+=1
    end

    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (8*2)/1e9*nx*ny*sizeof(eltype(flow.𝝫))  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(niter-10)                        # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]
    @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))
    
end
