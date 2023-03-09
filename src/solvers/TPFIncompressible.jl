# Two-phase flow incompressible solver
# this source file contains update routines needed for the incompressible solver 

@inbounds @parallel function assign!(𝝫_o::Data.Array, ∇V_o::Data.Array, 𝝫::Data.Array, ∇V::Data.Array)
    @all(𝝫_o)   = @all(𝝫)
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
        @parallel compute_params_∇!(flow.𝞰ɸ, flow.𝐤ɸ_µᶠ, flow.𝞀g, flow.∇V, flow.∇qD, flow.𝝫, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.k0, _ɸ0, rheology.nₖ, rheology.θ_e, rheology.θ_k, flow.ρfg, flow.ρsg, flow.ρgBG, _dx, _dy)

        # pressure update from the conservation of mass flow
        @parallel compute_residual_mass_law!(pt.Δτₚᵗ, pt.Δτₚᶠ, flow.R.Pt, flow.R.Pf, flow.𝐤ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, flow.𝝫, pt.Pfᵣ, pt.dampPf, min_dxy2, _dx, _dy)
        apply_free_slip!(freeslip, pt.Δτₚᶠ, nx, ny)
        @parallel compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.Δτₚᶠ, pt.Δτₚᵗ)
        @parallel compute_tensor!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y,  flow.∇V, flow.R.Pt, rheology.μˢ, pt.ηb, _dx, _dy)

        
    
        # velocity update from the conservation of momentum flow
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.gᵛˣ, pt.gᵛʸ, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, pt.dampVx, pt.dampVy, _dx, _dy)
        @parallel compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.gᵛˣ, pt.gᵛʸ, flow.𝐤ɸ_µᶠ, flow.Pf, pt.Δτᵥ, flow.ρfg, flow.ρgBG, _dx, _dy)
        apply_free_slip!(freeslip, flow.V.x, flow.V.y, nx+1, ny+1)
        apply_free_slip!(freeslip, flow.qD.x, flow.qD.y, nx+1, ny+1)
    
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


#========temporary solver function=========#
@inbounds function solve!(

    StrikeSlip_2D::String,

    h_index::Integer,

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
    

    # TODO: check if correct values used!
    p₀f  = 5.0e6,       # initial fluid pressure 5 MPa
    Δpf  = 5.0e6,       # constant amount of fluid to be injected 5 MPa 
    Vpl  = 1.9977e-9, # loading rate [m/s] = 6.3 cm/yr
    p⁻   = -1.0e-12,     # BC top - outward flux [m/s]
    p⁺   = 1.0e-12,      # BC bottom - inward flux [m/s]
    Peff = 3.0e+7,     # constant effective pressure [Pa] -> 30MPa 

    
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
        @parallel compute_params_∇!(flow.𝞰ɸ, flow.𝐤ɸ_µᶠ, flow.𝞀g, flow.∇V, flow.∇qD, flow.𝝫, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.k0, _ɸ0, rheology.nₖ, rheology.θ_e, rheology.θ_k, flow.ρfg, flow.ρsg, flow.ρgBG, _dx, _dy)

        # pressure update from the conservation of mass flow
        @parallel compute_residual_mass_law!(pt.Δτₚᵗ, pt.Δτₚᶠ, flow.R.Pt, flow.R.Pf, flow.𝐤ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, flow.𝝫, pt.Pfᵣ, pt.dampPf, min_dxy2, _dx, _dy)
        apply_free_slip!(freeslip, pt.Δτₚᶠ, nx, ny)
        @parallel compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.Δτₚᶠ, pt.Δτₚᵗ)
        @parallel compute_tensor!(flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.V.x, flow.V.y,  flow.∇V, flow.R.Pt, rheology.μˢ, pt.ηb, _dx, _dy)

    
        # velocity update from the conservation of momentum flow
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.gᵛˣ, pt.gᵛʸ, flow.𝞂ʼ.xx, flow.𝞂ʼ.yy, flow.𝞂ʼ.xy, flow.Pt, flow.𝞀g, pt.dampVx, pt.dampVy, _dx, _dy)
        @parallel compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.gᵛˣ, pt.gᵛʸ, flow.𝐤ɸ_µᶠ, flow.Pf, pt.Δτᵥ, flow.ρfg, flow.ρgBG, _dx, _dy)
        
        # left/right boundary
        @parallel (1:ny+1) free_slip_y!(flow.V.x)
        @parallel (1:ny)   free_slip_y!(flow.V.y)
        @parallel (1:ny)   free_slip_y!(flow.qD.y)

        # top & bottom boundary
        @parallel (1:nx)   dirichlet_x!(flow.V.x, 0.5 * Vpl, -0.5 * Vpl)
        @parallel (1:nx+1) dirichlet_x!(flow.V.y, 0.0, 0.0)
        @parallel (1:nx)   constant_flux_x!(flow.qD.y, p⁻, p⁺)
        @parallel (1:nx)   constant_effective_pressure_x!(flow.Pt, flow.Pf, Peff)

        
        # used for fluid injection benchmark! Otherwise not!
        flow.Pf[h_index, 1] = p₀f + Δpf      # constant fluid injection to the leftmost injection point on the fault
        
        # FIXME: not updating porosity in Dal Zilio (2022)
        # @parallel compute_porosity!(flow.𝝫, flow.𝝫_o, flow.∇V, flow.∇V_o, CN, Δt)


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
