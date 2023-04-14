using Plots, Plots.Measures
using LaTeXStrings


"""
Implements the Dieterich law which relates the state variable via
the aging law, the regularized version of it, and the invariant reformulation
for the regularized RSF law.
"""
function rsf_comparison()

    # Visualization
    default(size=(1400,700),fontfamily="Computer Modern", linewidth=3, framestyle=:box, margin=10.0mm)
    scalefontsizes(); scalefontsizes(1.35)
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")



    # variables
    a  = 0.01
    b  = 0.015
    V0 = 1.0         # 1 γm/s
    Dc = 20.0        # 10 γm
    k  = 0.01        # 0.01 γm⁻¹
    θ  = 0.0
    γ0 = 0.05


    # for regularized law
    L  = 20.0

    # setting sliding velocity
    # Define the sliding velocity
    V             = fill(1.0, 4000)  # Fill an array of length t with 1.0
    V[1001:2000] .= 10.0            # Set elements 1001:2000 to 10.0


    # time
    # we solve for 40 sec at 100Hz
    Δt    = 0.01         # 1 s
    t_tot = 40.0
    t     = 0.0
    iter  = 1


    # evolution wrt time
    evo_t = Float64[]; evo_γ1 = Float64[]; evo_γ2 = Float64[]; evo_γ3 = Float64[]

    while t < t_tot



        # aging_law(θ, Δt, V, Dc)
        θ = (θ + Δt) / (1 + Δt * V[iter]/Dc)

        # rsf_friction_evolution(γ, γ0, a, V, V0, b, θ, Dc)
        γ1 = γ0 + a * log(V[iter]/V0) + b * log(V0*θ/Dc)

        
        # regularized rsf
        γ2 = asinh( V[iter]/2/V0 * exp(b/a * log(θ*V0/L) + γ0/a)) * a


        # invariant reformulation rsf
        Ω  = log(θ*V0/L)
        γ3 = asinh( V[iter]/2/V0 * exp((b * Ω + γ0) / a)) * a


        # store current state
        push!(evo_t, t)
        push!(evo_γ1, γ1)
        push!(evo_γ2, γ2)
        push!(evo_γ3, γ3)



        if mod(iter, 5) == 0

            p1 = plot(evo_t, evo_γ1; xlims=(0.0, t_tot), ylims=(0.0, 0.07), label="Dieterich law", color= :dodgerblue, framestyle= :box, linestyle= :solid, seriesstyle= :path, 
                        title="Comparison of Different Formulations for RSF", 
                        xlabel = "Time", ylabel="Effective Friction Coefficient "* L"\gamma_\mathrm{eff}")

            plot!(p1, evo_t, evo_γ2, label="Regularized law", linestyle= :dash, linewidth = 4)
            plot!(p1, evo_t, evo_γ3, label="Invariant reformulation of regularized law")


            display(plot(p1)); frame(anim)
        end


        # advancing in time
        t    += Δt
        @show iter += 1

    end



    gif(anim, "rsf_friction.gif", fps = 15)



end


rsf_comparison()

