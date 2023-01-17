# Momentum Conservation Law
# Compute kernel for update of physical properties for the momentum conservation
# i). without inertia => stokes equation with assumption Re << 1
# ii). with inertia   => naiver stokes


#=================== RESIDUAL UPDATE ======================#

# i).without inertia

# compute residual for stokes equation
@inbounds @parallel function compute_residual_momentum_law!(RVx::Data.Array, RVy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Pt::Data.Array, 𝞀g::Data.Array, dampX::Data.Number, dampY::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # compute residual f_vᵢⁿ for total momentum 
    @all(RVx)    = (@d_xi(τxx)- @d_xi(Pt))* _dx + @d_ya(τxy)* _dy 
    @all(RVy)    = (@d_yi(τyy)- @d_yi(Pt))* _dy + @d_xa(τxy)* _dx - @av_yi(𝞀g)

    # apply damping terms for the residual
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(RVx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(RVy)
    return
end

# ii).with inertia





#================== PHYSICAL PROPERTIES =================#

# velocities update
# @inbounds @parallel function compute_velocity!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, 𝗞ɸ_µᶠ::Data.Array, Pf::Data.Array, dτV::Data.Number, ρfg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)
@inbounds @parallel function compute_velocity!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, 𝗞ɸ_µᶠ::Data.Array, Pf::Data.Array, dτV::Data.Number, ρfg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # i). total momentum, velocity update
    # vᵢⁿ = vᵢⁿ⁻¹ + Δτ_vᵢ f_vᵢⁿ for i in x,y    
    @inn(Vx)  =  @inn(Vx) + dτV*@all(dVxdτ)
    @inn(Vy)  =  @inn(Vy) + dτV*@all(dVydτ)

    # ii). fluid momentum, velocity update
    # qDᵢⁿ = - k^ɸ/ µ^f (∇Pf - Pf·g)
    @inn(qDx) = -@av_xi(𝗞ɸ_µᶠ)*(@d_xi(Pf)* _dx)
    @inn(qDy) = -@av_yi(𝗞ɸ_µᶠ)*(@d_yi(Pf)* _dy + (ρfg - ρgBG))
    
    return
end