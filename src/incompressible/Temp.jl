# Compute kernel of the incompressible two-phase flow solver
@inbounds @parallel function update_old!(Phi_o::Data.Array, ∇V_o::Data.Array, Phi::Data.Array, ∇V::Data.Array)
    @all(Phi_o) = @all(Phi)
    @all(∇V_o)  = @all(∇V)
    return
end



@inbounds @parallel function compute_params_∇!(EtaC::Data.Array, K_muf::Data.Array, Rog::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Phi::Data.Array, Pf::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, μs::Data.Number, η2μs::Data.Number, R::Data.Number, λPe::Data.Number, k_μf0::Data.Number, _ϕ0::Data.Number, nperm::Data.Number, θ_e::Data.Number, θ_k::Data.Number, ρfg::Data.Number, ρsg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @all(EtaC)  = (1.0-θ_e)*@all(EtaC)  + θ_e*( μs/@all(Phi)*η2μs*(1.0+0.5*(1.0/R-1.0)*(1.0+tanh((@all(Pf)-@all(Pt))/λPe))) )
    @all(K_muf) = (1.0-θ_k)*@all(K_muf) + θ_k*( k_μf0 * (@all(Phi)* _ϕ0)^nperm )
    @all(Rog)   = ρfg*@all(Phi) + ρsg*(1.0-@all(Phi)) - ρgBG
    @all(∇V)    = @d_xa(Vx)* _dx  + @d_ya(Vy)* _dy
    @all(∇qD)   = @d_xa(qDx)* _dx + @d_ya(qDy)* _dy

    return
end


@inbounds @parallel function compute_RP!(dτPf::Data.Array, RPt::Data.Array, RPf::Data.Array, K_muf::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, EtaC::Data.Array, Phi::Data.Array, Pfsc::Data.Number, Pfdmp::Data.Number, min_dxy2::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @inn(dτPf) = min_dxy2/@maxloc(K_muf)/4.1/Pfsc
    @all(RPt)  =                 - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(EtaC)*(1.0-@all(Phi)))
    @all(RPf)  = @all(RPf)*Pfdmp - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(EtaC)*(1.0-@all(Phi)))

    return
end



@inbounds @parallel function compute_P_τ!(Pt::Data.Array, Pf::Data.Array, τxx::Data.Array, τyy::Data.Array, σxy::Data.Array, RPt::Data.Array, RPf::Data.Array, dτPf::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, dτPt::Data.Number, μs::Data.Number, β_n::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @all(Pt)  = @all(Pt) +      dτPt *@all(RPt)
    @all(Pf)  = @all(Pf) + @all(dτPf)*@all(RPf)
    @all(τxx) = 2.0*μs*( @d_xa(Vx)* _dx - 1.0/3.0*@all(∇V) - β_n*@all(RPt) )
    @all(τyy) = 2.0*μs*( @d_ya(Vy)* _dy - 1.0/3.0*@all(∇V) - β_n*@all(RPt) )
    @all(σxy) = 2.0*μs*(0.5*( @d_yi(Vx)* _dy + @d_xi(Vy)* _dx ))
    return
end

@inbounds @parallel function compute_res!(Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, τxx::Data.Array, τyy::Data.Array, σxy::Data.Array, Pt::Data.Array, Rog::Data.Array, dampX::Data.Number, dampY::Data.Number, _dx::Data.Number, _dy::Data.Number)
    # @all(Rx)    = @d_xi(τxx)* _dx + @d_ya(σxy)* _dy - @d_xi(Pt)* _dx
    # @all(Rx)    = (@d_xi(τxx)- @d_xi(Pt))* _dx + @d_ya(σxy)* _dy 
    # @all(Ry)    = @d_yi(τyy)* _dy + @d_xa(σxy)* _dx - @d_yi(Pt)* _dy - @av_yi(Rog)
    @all(Ry)    = (@d_yi(τyy)- @d_yi(Pt))* _dy + @d_xa(σxy)* _dx - @av_yi(Rog)

    # @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVxdτ) = dampX*@all(dVxdτ) + (@d_xi(τxx)- @d_xi(Pt))* _dx + @d_ya(σxy)* _dy
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end


@inbounds @parallel function compute_update!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, Phi::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, K_muf::Data.Array, Pf::Data.Array, Phi_o::Data.Array, ∇V::Data.Array, ∇V_o::Data.Array, dτV::Data.Number, ρfg::Data.Number, ρgBG::Data.Number, CN::Data.Number, dt::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @inn(Vx)  =  @inn(Vx) + dτV*@all(dVxdτ)
    @inn(Vy)  =  @inn(Vy) + dτV*@all(dVydτ)
    @inn(qDx) = -@av_xi(K_muf)*(@d_xi(Pf)* _dx)                     # fluid momentum x
    @inn(qDy) = -@av_yi(K_muf)*(@d_yi(Pf)* _dy + (ρfg - ρgBG))      # fluid momentum y
    @all(Phi) =  @all(Phi_o) + (1.0-@all(Phi))*(CN*@all(∇V_o) + (1.0-CN)*@all(∇V))*dt

    return
end