# Boundary Conditions

# i). Dirichlet boundary condition
@inline @inbounds @parallel_indices (iy) function dirichlet_x!(A::Data.Array, val_top::Data.Number, val_bottom::Data.Number)
    A[1, iy]   = val_top
    A[end, iy] = val_bottom 

    return nothing
end

@inline @inbounds @parallel_indices (ix) function dirichlet_y!(A::Data.Array, val_left::Data.Number, val_right::Data.Number)
    A[ix, 1]   = val_left
    A[ix, end] = val_right

    return nothing
end

# eg. pt  = pf + peff
# @inline @inbounds @parallel_indices (iy) function constant_effective_pressure_x!(A::Data.Array, B::Data.Array, val::Data.Number)
#     A[1,iy]   = B[1, iy] + val
#     A[end,iy] = B[end, iy] + val

#     return nothing
# end

# eg. pf = pt - peff
@inline @inbounds @parallel_indices (iy) function constant_effective_pressure_x!(A::Data.Array, B::Data.Array, val::Data.Number)
    A[1,iy]   = B[1, iy] - val
    A[end,iy] = B[end, iy] - val

    return nothing
end



# ii). Neumann boundary condition

# CONSTANT FLUX ∂V/∂x = C for some constant

# apply constant flux condition along x-axis
@inline @inbounds @parallel_indices (iy) function constant_flux_x!(A::Data.Array, val_top::Data.Number, val_bottom::Data.Number)
    A[1, iy]   = 2 * val_top - A[2, iy]         # constant flux at top
    A[end, iy] = 2 * val_bottom - A[end-1, iy]  # constant flux at bottom
    return nothing
end



# FREE SLIP
@inline @inbounds @parallel_indices (iy) function free_slip_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return nothing
end

@inline @inbounds @parallel_indices (ix) function free_slip_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return nothing
end


@inline @inbounds function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx::Data.Array, Vy::Data.Array, size_Vx_x, size_Vy_y) where {T}
    freeslip_x, freeslip_y = freeslip

    # free slip boundary conditions
    freeslip_x && (@parallel (1:size_Vy_y) free_slip_x!(Vy))  # applied along x-axis, A[1, iy] = A[2, iy]
    freeslip_y && (@parallel (1:size_Vx_x) free_slip_y!(Vx))  # applied along y-axis  A[ix,1]  = A[ix,2]

    return nothing
end


@inline @inbounds function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, C::Data.Array, size_C_x, size_C_y) where {T}
    freeslip_x, freeslip_y = freeslip

    # free slip boundary conditions
    freeslip_x && (@parallel (1:size_C_y) free_slip_x!(C))
    freeslip_y && (@parallel (1:size_C_x) free_slip_y!(C))

    return nothing
end
