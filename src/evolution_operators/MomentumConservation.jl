# Momentum Conservation Law
# Compute kernel for update of physical properties for the momentum conservation
# i). without inertia => stokes equation with assumption Re << 1
# ii). with inertia   => naiver stokes


#=================== RESIDUAL UPDATE ======================#

# i).without inertia

# compute residual for stokes equation
@inbounds @parallel function compute_residual_momentum_law!(fᵛˣ::Data.Array, fᵛʸ::Data.Array, gᵛˣ::Data.Array, gᵛʸ::Data.Array, σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Pt::Data.Array, 𝞀g::Data.Array, dampVx::Data.Number, dampVy::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # compute residual f_vᵢⁿ for total momentum
    
    # FIXME: (gpu) common Cartesian coordinates with y-axis positive pointing upwards
    @all(fᵛˣ)    = (@d_xi(σxxʼ)- @d_xi(Pt))* _dx + @d_ya(σxyʼ)* _dy 
    @all(fᵛʸ)    = (@d_yi(σyyʼ)- @d_yi(Pt))* _dy + @d_xa(σxyʼ)* _dx - @av_yi(𝞀g)

    # geological coordinates y-axis positive pointing downwards
    # @all(fᵛˣ)    = (@d_xi(σxxʼ)- @d_xi(Pt))* _dx + @d_ya(σxyʼ)* _dy 
    # @all(fᵛʸ)    = (@d_yi(σyyʼ)- @d_yi(Pt))* _dy + @d_xa(σxyʼ)* _dx + @av_yi(𝞀g)

    # apply damping terms for the residual
    @all(gᵛˣ) = dampVx * @all(gᵛˣ) + @all(fᵛˣ)
    @all(gᵛʸ) = dampVy * @all(gᵛʸ) + @all(fᵛʸ)
    return
end


# ii).with inertia





#================== PHYSICAL PROPERTIES =================#

# velocities update
# @inbounds @parallel function compute_velocity!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, gᵛˣ::Data.Array, gᵛʸ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, Pf::Data.Array, Δτᵥ::Data.Number, ρfg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)
@inbounds @parallel function compute_velocity!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, gᵛˣ::Data.Array, gᵛʸ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, Pf::Data.Array, Δτᵥ::Data.Number, ρfg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # i). total momentum, velocity update
    # vᵢⁿ = vᵢⁿ⁻¹ + Δτ_vᵢ g_vᵢⁿ for i in x,y
    @inn(Vx)  =  @inn(Vx) + Δτᵥ* @all(gᵛˣ)
    @inn(Vy)  =  @inn(Vy) + Δτᵥ* @all(gᵛʸ)

    # ii). fluid momentum, velocity update
    # qDᵢⁿ = - k^ɸ/ µ^f (∇Pf - ρ·g)

    # (gpu)
    @inn(qDx) = -@av_xi(𝐤ɸ_µᶠ)*(@d_xi(Pf)* _dx)
    @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy + (ρfg - ρgBG))

    # geological coords
    # @inn(qDx) = -@av_xi(𝐤ɸ_µᶠ)*(@d_xi(Pf)* _dx)
    # @inn(qDy) = -@av_yi(𝐤ɸ_µᶠ)*(@d_yi(Pf)* _dy - (ρfg - ρgBG))
    
    return
end
