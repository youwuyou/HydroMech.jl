"""  BasicTypes.jl
contains basic types that is commonly used in geophysical modeling
associated with PTArray which is a wrapper for the multi-xPU arrays of the ParallelStencil.jl package

1. Field Variables    (Scalar, Vector, Tensor)
2. Mesh topology      (staggered grid)
3. Equations          (for representing PDEs to be solved)

NOTE: other types can be customly added but once we use routines of ParallelStencil.jl package, we need
to also have them exported in MetaHydroMech.jl script under src
"""


#======= 1. Field Variables ========#
"""
In continuum mechanics, we use three major types of field variables (scalar, vector, tensor) to describe the physical properties of a continuum.
In order to provide an adequate level of abstraction, we create two new basic types that correspond to PTVector and PTSymmetricTensor. 
Both types are based on the PTArray type which we defined in the MetaHydroMech.jl. The so-called PTArrays are just a high-level abstraction for multi-xPU arrays, 
which do not necessarily need to be used for only the PT method. We follow the same naming convention to avoid naming duplicance with other packages with the "PT" prefix here.
"""


function make_vector_struct!(ndim::Integer; name::Symbol=:PTVector)
    dims = (:x, :y, :z)
    fields = [:($(dims[i])::T) for i in 1:ndim]
    @eval begin
        # using mutable in order to be able to set I.C.
        # perhaps some hack around it for better performance in the future!
        mutable struct $(name){T}
            $(fields...)

            function $(name)(ni::NTuple{2,T}) where {T}
                return new{$PTArray}(@zeros(ni[1]...), @zeros(ni[2]...))
            end # end of 2D vector

        end
    end
end


function make_symmetrictensor_struct!(nDim::Integer; name::Symbol=:PTSymmetricTensor)
    
    # NOTE: the naming of the fields differs slightly from the naming convention in JustRelax.jl
    #      we added the symbol "T" representing a generic tensor. 
    dims = (:x, :y, :z)
    fields = [:($(Symbol((dims[i]), (dims[j])))::T) for i in 1:nDim, j in 1:nDim if j ≥ i]

    @eval begin
        struct $(name){T}
            $(fields...)
            # TII::T         FIXME: maybe add the second invariant to the class

            function $(name)(ni::NTuple{3,T}) where {T}
                return new{$PTArray}(
                    @zeros(ni[1]...),                # xx
                    @zeros(ni[2]...),                # xy
                    @zeros(ni[3]...),                # yy
                    # @zeros(ni...)                  # II (second invariant)
                )
            end # end of 2D tensor
        end
    end
end




#======= 2. Mesh Topology  =========#

"""
PTMesh{nDim}
An abstract supertype of specific mesh types such as `PTGrid`
The type parameters encode the number of spatial dimensions (`nDim`).
"""
abstract type PTMesh{nDim} end

# Staggered grid


"""
PTGrid{nDim}
A concrete type as subtype of PTMesh abstract type.
Encode minimal information on a well-defined tensor-structure grid.
"""
struct PTGrid{nDim} <: PTMesh{nDim}
    ni::NTuple{nDim, Integer}
    li::NTuple{nDim, Float64}
    di::NTuple{nDim, Float64}
end



#==== 3. Equations  ====#



