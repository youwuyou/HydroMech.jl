# Boundary Conditions
# (1) Free slip
# (2) no slip,
# (3) free surface,
# (4) fast erosion,
# (5) infinity-like (external free slip, external no slip, Winkler basement),
# (6) prescribed velocity (moving boundary)

# 2D kernel
@inline @inbounds @parallel_indices (iy) function free_slip_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@inline @inbounds @parallel_indices (ix) function free_slip_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end


@inline @inbounds function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx::Data.Array, Vy::Data.Array, size_Vx_x, size_Vy_y) where {T}
    freeslip_x, freeslip_y = freeslip

    # free slip boundary conditions
    freeslip_x && (@parallel (1:size_Vy_y) free_slip_x!(Vy))
    freeslip_y && (@parallel (1:size_Vx_x) free_slip_y!(Vx))

    return nothing
end


@inline @inbounds function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, C::Data.Array, size_C_x, size_C_y) where {T}
    freeslip_x, freeslip_y = freeslip

    # free slip boundary conditions
    freeslip_x && (@parallel (1:size_C_y) free_slip_x!(C))
    freeslip_y && (@parallel (1:size_C_x) free_slip_y!(C))

    return nothing
end
