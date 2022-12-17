# Two-phase flow compressible solver
# this source file contains update routines needed for the compressible solver 

@inbounds @parallel function update_old!(Phi_o::Data.Array, ∇V_o::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, Phi::Data.Array, ∇V::Data.Array,  Pt::Data.Array, Pf::Data.Array)
    @all(Phi_o) = @all(Phi)
    @all(∇V_o)  = @all(∇V)
    @all(Pt_o)  = @all(Pt)
    @all(Pf_o)  = @all(Pf)
    return
end


@inbounds @parallel function compute_Kd!(Kd::Data.Array, Kphi::Data.Array, Phi::Data.Array, _Ks::Data.Number, µ::Data.Number)


    # compute effective bulk modulus for the pores
    # Kɸ = 2m/(1+m)µ/ɸ =  µ/ɸ (m=1)
    @all(Kphi) = µ / @all(Phi)

    # compute drained bulk modulus
    # Kd = (1-ɸ)(1/Kɸ + 1/Ks)⁻¹
    @all(Kd) = (1.0 -@all(Phi)) / (1.0 /@all(Kphi) + _Ks)

    return nothing

end


@inbounds @parallel function compute_ɑ!(ɑ::Data.Array, βd::Data.Array, Kphi::Data.Array, Phi::Data.Array, βs::Data.Number)

    # compute solid skeleton compressibility
    # βd = (1+ βs·Kɸ)/(Kɸ-Kɸ·ɸ) = (1+ βs·Kɸ)/Kɸ/(1-ɸ)
    @all(βd) = (1.0 + βs * @all(Kphi)) / @all(Kphi) / (1-@all(Phi))
    @all(ɑ)  = 1.0 - βs / @all(βd)
    
    return nothing
end


@inbounds @parallel function compute_B!(B::Data.Array, Phi::Data.Array, βd::Data.Array, βs::Data.Number, βf::Data.Number)

    # compute skempton coefficient
    # B = (βd - βs)/(βd - βs + ɸ(βf - βs))
    @all(B) = (@all(βd) - βs) / (@all(βd) - βs + @all(Phi) * (βf - βs))

    return nothing
end



@inbounds function solve!(EtaC, K_muf, Rhog, ∇V, ∇qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, μs, η2μs, R, λPe, k_μf0, _ϕ0, nperm, θ_e, θ_k, ρfg, ρsg, ρgBG, _dx, _dy,
    dτPf, RPt, RPf, Pfsc, Pfdmp, min_dxy2,
    freeslip, nx, ny, τxx, τyy, σxy,dτPt, β_n,
    Rx, Ry, dVxdτ, dVydτ, dampX, dampY,
    Phi_o, ∇V_o, dτV, CN, dt,
    Kd, Kphi, _Ks, µ, ɑ, βd, βs, βf, B, Pt_o, Pf_o
    )

    @parallel compute_params_∇!(EtaC, K_muf, Rhog, ∇V, ∇qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, μs, η2μs, R, λPe, k_μf0, _ϕ0, nperm, θ_e, θ_k, ρfg, ρsg, ρgBG, _dx, _dy)
            
    #  parameters computation for compressible case!
    @parallel compute_Kd!(Kd, Kphi, Phi, _Ks, µ)
    @parallel compute_ɑ!(ɑ, βd, Kphi, Phi, βs)
    @parallel compute_B!(B, Phi, βd, βs, βf)
   
    @parallel compute_residual_mass_law!(dτPf, RPt, RPf, K_muf, ∇V, ∇qD, Pt, Pf, EtaC, Phi, Kd, ɑ, Pt_o, Pf_o, B, Pfsc, Pfdmp, min_dxy2, _dx, _dy, dt)
    
    apply_free_slip!(freeslip, dτPf, nx, ny)
   
    @parallel compute_pressure!(Pt, Pf, RPt, RPf, dτPf, dτPt)
    @parallel compute_tensor!(τxx, τyy, σxy, Vx, Vy, ∇V, RPt, μs, β_n, _dx, _dy)
    
    # velocity update from the conservation of momentum equations
    # for both fluid and solid
    @parallel compute_residual_momentum_law!(Rx, Ry, dVxdτ, dVydτ, τxx, τyy, σxy, Pt, Rhog, dampX, dampY, _dx, _dy)
    @parallel compute_velocity!(Vx, Vy, qDx, qDy, dVxdτ, dVydτ, K_muf, Pf, dτV, ρfg, ρgBG, _dx, _dy)
    apply_free_slip!(freeslip, Vx, Vy, nx+1, ny+1)
    apply_free_slip!(freeslip, qDx, qDy, nx+1, ny+1)

    # update the porosity
    @parallel compute_porosity!(Phi, Phi_o, ∇V, ∇V_o, CN, dt)
end





