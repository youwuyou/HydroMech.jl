# Momentum Conservation Law
# Compute kernel for update of physical properties for the momentum conservation
# i). without inertia => stokes equation with assumption Re << 1
# ii). with inertia   => naiver stokes


#=================== RESIDUAL UPDATE ======================#

# i).without inertia

# compute residual for stokes equation
@inbounds @parallel function compute_residual_momentum_law!(Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, τxx::Data.Array, τyy::Data.Array, σxy::Data.Array, Pt::Data.Array, Rhog::Data.Array, dampX::Data.Number, dampY::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # compute residual f_vᵢⁿ for total momentum 
    @all(Rx)    = (@d_xi(τxx)- @d_xi(Pt))* _dx + @d_ya(σxy)* _dy 
    @all(Ry)    = (@d_yi(τyy)- @d_yi(Pt))* _dy + @d_xa(σxy)* _dx - @av_yi(Rhog)

    # apply damping terms for the residual
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

# ii).with inertia





#================== PHYSICAL PROPERTIES =================#

# velocities update
@inbounds @parallel function compute_velocity!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, K_muf::Data.Array, Pf::Data.Array, dτV::Data.Number, ρfg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)
    # i). total momentum, velocity update
    # vᵢⁿ = vᵢⁿ⁻¹ + Δτ_vᵢ f_vᵢⁿ for i in x,y    
    @inn(Vx)  =  @inn(Vx) + dτV*@all(dVxdτ)
    @inn(Vy)  =  @inn(Vy) + dτV*@all(dVydτ)

    # ii). fluid momentum, velocity update
    # qDᵢⁿ = - k^ɸ/ µ^f (∇Pf - Pf·g)
    @inn(qDx) = -@av_xi(K_muf)*(@d_xi(Pf)* _dx)
    @inn(qDy) = -@av_yi(K_muf)*(@d_yi(Pf)* _dy + (ρfg - ρgBG))
    
    return
end