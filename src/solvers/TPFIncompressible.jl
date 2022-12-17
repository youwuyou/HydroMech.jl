# Two-phase flow incompressible solver
# this source file contains update routines needed for the incompressible solver 


@inbounds @parallel function update_old!(Phi_o::Data.Array, ∇V_o::Data.Array, Phi::Data.Array, ∇V::Data.Array)
    @all(Phi_o) = @all(Phi)
    @all(∇V_o)  = @all(∇V)
    return
end



@inbounds function solve!(EtaC, K_muf, Rhog, ∇V, ∇qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, μs, η2μs, R, λPe, k_μf0, _ϕ0, nperm, θ_e, θ_k, ρfg, ρsg, ρgBG, _dx, _dy,
                  dτPf, RPt, RPf, Pfsc, Pfdmp, min_dxy2,
                  freeslip, nx, ny, τxx, τyy, σxy,dτPt, β_n,
                  Rx, Ry, dVxdτ, dVydτ, dampX, dampY,
                  Phi_o, ∇V_o, dτV, CN, dt
                  )
    @parallel compute_params_∇!(EtaC, K_muf, Rhog, ∇V, ∇qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, μs, η2μs, R, λPe, k_μf0, _ϕ0, nperm, θ_e, θ_k, ρfg, ρsg, ρgBG, _dx, _dy)
    
    # pressure update from the conservation of mass equations
    @parallel compute_residual_mass_law!(dτPt, dτPf, RPt, RPf, K_muf, ∇V, ∇qD, Pt, Pf, EtaC, Phi, Pfsc, Pfdmp, min_dxy2, _dx, _dy)
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

