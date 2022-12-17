struct PS_Setup{B,C}
    device::Symbol

    function PS_Setup(device::Symbol, precission::DataType, nDim::Integer)
        return new{precission,nDim}(device)
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

    # includes and exports
    @eval begin
        export USE_GPU, PTArray

        #==============  BOUNDARY CONDITION ================#
        include(joinpath(@__DIR__, "boundaryconditions/BoundaryConditions.jl"))
        export free_slip_x!, free_slip_y!, apply_free_slip!

        #==============  CONSERVATION LAWS ================#
        include(joinpath(@__DIR__, "equations/MassConservation.jl"))
        export compute_residual_mass_law!, compute_pressure!, compute_tensor!, compute_porosity!

        include(joinpath(@__DIR__, "equations/MomentumConservation.jl"))
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