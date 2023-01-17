# Mass Conservation Law
# Compute kernel for update of physical properties for the mass conservation
# i). incompressible
# ii). compressible

#=================== RESIDUAL UPDATES ======================#
# compute continuity equation for single phase flow problem


# compute mass conservation residual for two phase flow problem (incompressible)
@inbounds @parallel function compute_residual_mass_law!(dτPt::Data.Number, dτPf::Data.Array, RPt::Data.Array, RPf::Data.Array, 𝗞ɸ_µᶠ::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, 𝞰ɸ::Data.Array, 𝝫::Data.Array, Pfsc::Data.Number, Pfdmp::Data.Number, min_dxy2::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @inn(dτPf) = min_dxy2/@maxloc(𝗞ɸ_µᶠ)/4.1/Pfsc

    # residual f_pt for incompressible solid mass
    @all(RPt)  =                 - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫)))
    
    #  residual f_pf for incompressible fluid mass 
    @all(RPf)  = @all(RPf)*Pfdmp - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫)))

    return
end


# compute mass conservation residual for two phase flow problem (compressible)
@inbounds @parallel function compute_residual_mass_law!(dτPf::Data.Array, RPt::Data.Array, RPf::Data.Array, 𝗞ɸ_µᶠ::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, 𝞰ɸ::Data.Array, 𝝫::Data.Array, Kd::Data.Array, ɑ::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, B::Data.Array, Pfsc::Data.Number, Pfdmp::Data.Number, min_dxy2::Data.Number, _dx::Data.Number, _dy::Data.Number, Δt::Data.Number)
     @inn(dτPf) = min_dxy2/@maxloc(𝗞ɸ_µᶠ)/4.1/Pfsc
 
     # residual f_pt for compressible solid mass
     @all(RPt)  = - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫))) - 
                  1/@all(Kd)/Δt * (@all(Pt)- @all(Pt_o) + @all(ɑ)* (@all(Pf_o) - @all(Pf)))
 
     #  residual f_pf for compressible fluid mass 
     @all(RPf)  = @all(RPf)*Pfdmp - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫))) + 
                  @all(ɑ)/@all(Kd)/Δt * (@all(Pt) - 1/@all(B) * @all(Pf)) - (@all(Pt_o)-1/@all(B)*@all(Pf_o))
 
     return
 end



#================== PHYSICAL PROPERTIES =================#

# compute residual for fluid and solid mass conservation eq
@inbounds @parallel function compute_pressure!(Pt::Data.Array, Pf::Data.Array, RPt::Data.Array, RPf::Data.Array, dτPf::Data.Array, dτPt::Data.Number)

    # i). incompressible solid mass, total pressure update
    # pⁿ = pⁿ⁻¹ + Δτ_pt f_pⁿ    
    @all(Pt)  = @all(Pt) +      dτPt *@all(RPt)
    
    # ii). incompressible fluid mass, fluid pressure update
    # pfⁿ = pfⁿ⁻¹ + Δτ_pf f_pfⁿ
    @all(Pf)  = @all(Pf) + @all(dτPf)*@all(RPf)
    
    return nothing
end

# compute stress update
@inbounds @parallel function compute_tensor!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, RPt::Data.Array, μˢ::Data.Number, βₚₜ::Data.Number, _dx::Data.Number, _dy::Data.Number)

    @all(τxx) = 2.0*μˢ*( @d_xa(Vx)* _dx - 1.0/3.0*@all(∇V) - βₚₜ*@all(RPt) )
    @all(τyy) = 2.0*μˢ*( @d_ya(Vy)* _dy - 1.0/3.0*@all(∇V) - βₚₜ*@all(RPt) )
    @all(τxy) = 2.0*μˢ*(0.5*( @d_yi(Vx)* _dy + @d_xi(Vy)* _dx ))

    return nothing
end


# compute porosity update
@inbounds @parallel function compute_porosity!(𝝫::Data.Array, 𝝫_o::Data.Array, ∇V::Data.Array, ∇V_o::Data.Array, CN::Data.Number, Δt::Data.Number)
    # ∂ɸ/∂t = (1-ɸ) ∇ₖvₖ^s
    @all(𝝫) =  @all(𝝫_o) + (1.0-@all(𝝫))*(CN*@all(∇V_o) + (1.0-CN)*@all(∇V))*Δt

    return nothing
end
