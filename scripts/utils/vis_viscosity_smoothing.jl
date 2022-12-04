using GLMakie
ENV["GKSwstype"]="nul"

# lambda expression for the ηᵩ 
# as defined in Räss et al. for the hydro-mech solver
# => a smooth viscosity drop by tanh term
ηϕ( (;ηc, ϕ0, ϕ, λp), R, Pe ) = @. ηc * ϕ0 / ϕ * (1 + 0.5*(1/R-1) *(1 + tanh(-Pe/λp)))


function visualize()
    # prescribe values to the effective pressure
    Pe = LinRange(-0.1, 0.1,100)
    
    # prescribe values to be pass to the lambda expr.
    vars = (; ηc=1, ϕ0=1e-2, ϕ=1e-2, λp=1e-2)
    
    # visualization
    fig=Figure()
    
    ax=Axis(fig[1,1], xlabel="effective pressure", ylabel="ηϕ" )
    
    for R in (1e0, 1e1, 1e2, 1e3)
    
        vars = (; ηc=1, ϕ0=1e-2, ϕ=1e-2, λp=1e-2)
        visc = ηϕ(vars, R, Pe)
        lines!(ax, Pe, visc, label="R=$R")
        
    end
    
    axislegend(ax)
    
    save("smoothed_drop_cosh.png", fig)
    
    return fig

end

# ↓ uncomment to enable the visu
# visualize()
