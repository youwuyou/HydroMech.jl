# Compute kernel of the compressible two-phase flow solver
# included in the package by MetaHydroMech.jl

@inbounds @parallel function update_old_compressible!(Phi_o::Data.Array, ∇V_o::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, Phi::Data.Array, ∇V::Data.Array,  Pt::Data.Array, Pf::Data.Array)
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


@inbounds @parallel function compute_RP_compressible!(dτPf::Data.Array, RPt::Data.Array, RPf::Data.Array, K_muf::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, EtaC::Data.Array, Phi::Data.Array, Kd::Data.Array, ɑ::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, B::Data.Array, Pfsc::Data.Number, Pfdmp::Data.Number, min_dxy2::Data.Number, _dx::Data.Number, _dy::Data.Number, dt::Data.Number)
   # FIXME: perhaps changes needed here
    @inn(dτPf) = min_dxy2/@maxloc(K_muf)/4.1/Pfsc

    # residual f_pt for compressible solid mass
    @all(RPt)  = - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(EtaC)*(1.0-@all(Phi))) - 
                 1/@all(Kd)/dt * (@all(Pt)- @all(Pt_o) + @all(ɑ)* (@all(Pf_o) - @all(Pf)))

    #  residual f_pf for compressible fluid mass 
    @all(RPf)  = @all(RPf)*Pfdmp - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(EtaC)*(1.0-@all(Phi))) + 
                 @all(ɑ)/@all(Kd)/dt * (@all(Pt) - 1/@all(B) * @all(Pf)) - (@all(Pt_o)-1/@all(B)*@all(Pf_o))

    return
end





