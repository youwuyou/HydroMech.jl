struct PS_Setup{B,C}
    device::Symbol

    function PS_Setup(device::Symbol, precision::DataType, nDim::Integer)
        return new{precision,nDim}(device)
    end
end

function environment!(model::PS_Setup{T,N}) where {T,N}
    gpu = model.device == :gpu ? true : false

    # environment variable for XPU
    @eval begin
        const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : $gpu
    end

    # call appropriate FD module
    Base.eval(@__MODULE__, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D"))
    Base.eval(Main, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D"))

    # using const PTArray to avoid type instability
    if model.device == :gpu
        eval(:(@init_parallel_stencil(CUDA, $T, $N)))
        Base.eval(Main, Meta.parse("using CUDA"))
        eval(:(const PTArray = CUDA.CuArray{$T,$N}))
    else
        @eval begin
            @init_parallel_stencil(Threads, $T, $N)
            const PTArray = Array{$T,$N}
        end
    end


    # TODO: add the creation for array structs!
    make_vector_struct!(N)             # PTVector
    make_symmetrictensor_struct!(N)    # PTSymmetricTensor

    make_twophase_residual_struct!(N)  # Residuals for twophase flow equations
    make_twophase_struct!()            # TwoPhaseFlowEquations2D
    make_pt_struct!()

    
    # includes and exports
    @eval begin
        export USE_GPU, PTArray

        #====== MetaJustRelax.jl =======#
        export PTVector, PTSymmetricTensor
        export TwoPhaseResidual, TwoPhaseFlow2D
        export PTCoeff
        
        Adapt.@adapt_structure PTVector
        Adapt.@adapt_structure PTSymmetricTensor
        Adapt.@adapt_structure TwoPhaseFlow2D
        Adapt.@adapt_structure TwoPhaseResidual
        Adapt.@adapt_structure PTCoeff

        #===Type dispatch for PTCoeff===#
        export OriginalDamping



        #==============  BOUNDARY CONDITION ================#
        include(joinpath(@__DIR__, "boundary_conditions/BoundaryConditions.jl"))
        export free_slip_x!, free_slip_y!, apply_free_slip!

        #==============  DISCRETE EVOLUTION OPERATORS FOR CONSERVATION LAWS ================#
        include(joinpath(@__DIR__, "evolution_operators/MassConservation.jl"))
        export compute_residual_mass_law!, compute_pressure!, compute_tensor!, compute_porosity!

        include(joinpath(@__DIR__, "evolution_operators/MomentumConservation.jl"))
        export compute_residual_momentum_law!, compute_velocity!

        #=================== SOLVERS ======================#
        # Two-phase flow solvers
        include(joinpath(@__DIR__, "solvers/TPFCommon.jl"))
        export compute_params_∇!

        include(joinpath(@__DIR__, "solvers/TPFIncompressible.jl"))
        export update_old!, solve!

        include(joinpath(@__DIR__, "solvers/TPFCompressible.jl"))
        export update_old!, compute_Kd!, compute_ɑ!, compute_B!, solve!

    end
end

function ps_reset!()
    Base.eval(Main, ParallelStencil.@reset_parallel_stencil)
    Base.eval(@__MODULE__, ParallelStencil.@reset_parallel_stencil)
    return nothing
end