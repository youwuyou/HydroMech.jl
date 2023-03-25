# Mass Conservation Law
# Compute kernel for update of physical properties for the mass conservation
# i).  incompressible
# ii). compressible


# TODO: use the 2nd order scheme for total pressure in the following updates!

#=================== RESIDUAL UPDATES ======================#
# compute continuity equation for single phase flow problem


# compute mass conservation residual for two phase flow problem (incompressible)
@inbounds @parallel function compute_residual_mass_law!(Δτₚᵗ::Data.Number, Δτₚᶠ::Data.Array, fᴾᵗ::Data.Array, fᴾᶠ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, 𝞰ɸ::Data.Array, 𝝫::Data.Array, Pfsc::Data.Number, dampPf::Data.Number, min_dxy2::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @inn(Δτₚᶠ) = min_dxy2/@maxloc(𝐤ɸ_µᶠ)/4.1/Pfsc

    # residual f_pt for incompressible solid mass
    @all(fᴾᵗ)  =                 - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫)))
    
    #  residual f_pf for incompressible fluid mass 
    @all(fᴾᶠ)  = @all(fᴾᶠ) * dampPf - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫)))
    return
end


# compute mass conservation residual for two phase flow problem (compressible)
@inbounds @parallel function compute_residual_mass_law!(Δτₚᶠ::Data.Array, fᴾᵗ::Data.Array, fᴾᶠ::Data.Array, 𝐤ɸ_µᶠ::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, 𝞰ɸ::Data.Array, 𝝫::Data.Array, 𝗞d::Data.Array, 𝝰::Data.Array, Pt_o::Data.Array, Pf_o::Data.Array, 𝗕::Data.Array, Pfsc::Data.Number, dampPf::Data.Number, min_dxy2::Data.Number, Δt::Data.Number)
     @inn(Δτₚᶠ) = min_dxy2/@maxloc(𝐤ɸ_µᶠ)/4.1/Pfsc

     # residual f_pt for compressible solid mass
    #  + @all(𝝰) ... and + 1/@all(B) here to avoid subtraction operation due to performance
     @all(fᴾᵗ)  =  - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫))) -
                         1.0 /@all(𝗞d)/Δt * (@all(Pt)- @all(Pt_o) + @all(𝝰)* (@all(Pf_o) - @all(Pf)))

     #  residual f_pf for compressible fluid mass 
     @all(fᴾᶠ)  = @all(fᴾᶠ)*dampPf - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(𝞰ɸ)*(1.0-@all(𝝫))) + 
                        @all(𝝰)/@all(𝗞d)/Δt * (@all(Pt) - @all(Pt_o) + 1.0/@all(𝗕)* (@all(Pf_o) - @all(Pf)))
 
     return
 end



#================== PHYSICAL PROPERTIES =================#

# compute residual for fluid and solid mass conservation eq
@inbounds @parallel function compute_pressure!(Pt::Data.Array, Pf::Data.Array, fᴾᵗ::Data.Array, fᴾᶠ::Data.Array, Δτₚᶠ::Data.Array, Δτₚᵗ::Data.Number)

    # i). incompressible solid mass, total pressure update
    # ptⁿ = ptⁿ⁻¹ + Δτ_pt f_ptⁿ    
    @all(Pt)  = @all(Pt) +      Δτₚᵗ *@all(fᴾᵗ)
    
    # ii). incompressible fluid mass, fluid pressure update
    # pfⁿ = pfⁿ⁻¹ + Δτ_pf f_pfⁿ
    @all(Pf)  = @all(Pf) + @all(Δτₚᶠ)*@all(fᴾᶠ)
    
    return nothing
end


# compute residual for fluid and solid mass conservation eq but with constant Δτₚᶠ
@inbounds @parallel function compute_pressure!(Pt::Data.Array, Pf::Data.Array, fᴾᵗ::Data.Array, fᴾᶠ::Data.Array, Δτₚᶠ::Data.Number, Δτₚᵗ::Data.Number)

    # i). incompressible solid mass, total pressure update
    # ptⁿ = ptⁿ⁻¹ + Δτ_pt f_ptⁿ    
    @all(Pt)  = @all(Pt) + Δτₚᵗ * @all(fᴾᵗ)
    
    # ii). incompressible fluid mass, fluid pressure update
    # pfⁿ = pfⁿ⁻¹ + Δτ_pf f_pfⁿ
    @all(Pf)  = @all(Pf) + Δτₚᶠ * @all(fᴾᶠ)
    
    return nothing
end


# compute stress update
@inbounds @parallel function compute_tensor!(σxxʼ::Data.Array, σyyʼ::Data.Array, σxyʼ::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, fᴾᵗ::Data.Array, μˢ::Data.Number, ηb::Data.Number, _dx::Data.Number, _dy::Data.Number)

    # TODO: add the plasticity and elasticity!
   
    # @all(Pt)  = @all(Pt) +      dτPt *@all(RPt)
    # @all(Pf)  = @all(Pf) + @all(dτPf)*@all(RPf)
    # @all(τxx) = 2.0*μs*( @d_xa(Vx)/dx - 1.0/3.0*@all(∇V) - β_n*@all(RPt) )
    # @all(τyy) = 2.0*μs*( @d_ya(Vy)/dy - 1.0/3.0*@all(∇V) - β_n*@all(RPt) )
    # @all(σxy) = 2.0*μs*(0.5*( @d_yi(Vx)/dy + @d_xi(Vy)/dx ))
    
    
    # General formula for viscous creep shear rheology
    # μˢ <-> solid shear viscosity
    # σᵢⱼ' = 2μˢ · ɛ̇ᵢⱼ = 2μˢ · (1/2 (∇ᵢvⱼˢ + ∇ⱼvᵢˢ) - 1/3 δᵢⱼ ∇ₖvₖˢ)
    
    # σxxʼ = 2μˢ · ɛ̇xx = 2μˢ · (∂Vx/∂x - 1/3 δᵢⱼ ∇ₖvₖˢ)
    @all(σxxʼ) = 2.0*μˢ*( @d_xa(Vx)* _dx - 1.0/3.0*@all(∇V) - ηb*@all(fᴾᵗ) )
    @all(σyyʼ) = 2.0*μˢ*( @d_ya(Vy)* _dy - 1.0/3.0*@all(∇V) - ηb*@all(fᴾᵗ) )
    

    # compute the xy component of the deviatoric stress
    # σxy' = 2μˢ · ɛ̇xy = 2μˢ · 1/2 (∂Vx/∂y + ∂Vy/∂x) =  μˢ · (∂Vx/∂y + ∂Vy/∂x)     
    @all(σxyʼ) = 2.0*μˢ*(0.5*( @d_yi(Vx)* _dy + @d_xi(Vy)* _dx ))

    return nothing
end


# compute porosity update
@inbounds @parallel function compute_porosity!(𝝫::Data.Array, 𝝫_o::Data.Array, ∇V::Data.Array, ∇V_o::Data.Array, CN::Data.Number, Δt::Data.Number)
    # ∂ɸ/∂t = (1-ɸ) ∇ₖvₖ^s
    @all(𝝫) =  @all(𝝫_o) + (1.0-@all(𝝫))*(CN*@all(∇V_o) + (1.0-CN)*@all(∇V))*Δt

    return nothing
end
