function model_problem()

    # 1. set up the mesh
    nx, ny = ...
    lx, ly = ...

    mesh = Grid(...)

    # 2. define rheology, relationships
    # accordingly, visco_elastic_rheology!
    # where the type of the returned struct object is different
    # for later multiplt-dispatch
    
    # constitutive law - for the rheology
    viscous = Viscosity(k0...)
    
    # porosity-dependent viscosity
    
    
    # empirical law - Carman-Kozeny
    #  power law dependent permeability
    k0    = ...
    ϕ0    = ...
    nperm = ...

    empirical_law = Carman_Kozeny(k0, ...)
    
    # relaxation factors
    relax = (θ_e, θ_k)



    rheology = Rheology(viscous, empirical_law, relax)
    


    # 3. write down the equations
    a = ()       # representing some params
    flow = TwoPhaseFlow2D(mesh, a)

    # 4. define I.C.
    qDy_cpu        = zeros(nx,ny)
    qDy_cpu[1:end] = ...
    flow.qDy = PTArray(qDy_cpu)

    
    # 5. apply PT method, augment the equation
    # choose between :diffusion and :damped_wave
    # for the scheme of adding the pseudo-terms
    # variable equations is passed by reference here,
    # only used for type dispatch
    dampX, dampY  = ...
    PTCoeff = augment!(flow, :diffusion, dampX, dampY)
    # PT specific arrays
    # PT specific coefficients
    
    # 6. define B.C.
    freeslip = (freeslip_x=true, freeslip_y=true)

    # set up physical time loop
    t_tot = ...
    Δt    = ...
    t     = 0.0

    while t < t_tot 
        # call the solver routine
        solve!(flow, rheology, PTCoeff, mesh, freeslip, Δt)
   
        # callback functions here
        # eg. visu, performance, array storage

        # advance in time
        Δt    = ...
    end


end