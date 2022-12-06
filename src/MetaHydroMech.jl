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

    # start ParallelStencil
    # NOTE: the use of const PTArray boosts the performance significantly
    #       which avoids the type instability
    global PTArray
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

        include(joinpath(@__DIR__, "boundaryconditions/BoundaryConditions.jl"))
        export free_slip_x!, free_slip_y!, apply_free_slip!

        include(joinpath(@__DIR__, "incompressible/Temp.jl"))
        export update_old!, compute_params_∇!, compute_RP!, compute_P_τ!, compute_res!, compute_update!, compute!
    end
end

function ps_reset!()
    Base.eval(Main, ParallelStencil.@reset_parallel_stencil)
    Base.eval(@__MODULE__, ParallelStencil.@reset_parallel_stencil)
    return nothing
end