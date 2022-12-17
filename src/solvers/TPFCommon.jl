# Two-phase flow solvers
# this source file contains the common kernel that is shared by solvers for TPF problem 

@inbounds @parallel function compute_params_∇!(EtaC::Data.Array, K_muf::Data.Array, Rhog::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Phi::Data.Array, Pf::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, μs::Data.Number, η2μs::Data.Number, R::Data.Number, λPe::Data.Number, k_μf0::Data.Number, _ϕ0::Data.Number, nperm::Data.Number, θ_e::Data.Number, θ_k::Data.Number, ρfg::Data.Number, ρsg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @all(EtaC)  = (1.0-θ_e)*@all(EtaC)  + θ_e*( μs/@all(Phi)*η2μs*(1.0+0.5*(1.0/R-1.0)*(1.0+tanh((@all(Pf)-@all(Pt))/λPe))) )
    @all(K_muf) = (1.0-θ_k)*@all(K_muf) + θ_k*( k_μf0 * (@all(Phi)* _ϕ0)^nperm )
    @all(Rhog)   = ρfg*@all(Phi) + ρsg*(1.0-@all(Phi)) - ρgBG
    
    # compute gradient 2D
    @all(∇V)    = @d_xa(Vx)* _dx  + @d_ya(Vy)* _dy
    @all(∇qD)   = @d_xa(qDx)* _dx + @d_ya(qDy)* _dy

    return
end