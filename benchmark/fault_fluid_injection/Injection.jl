# this script is used to plot the fluid injection benchmark result at selected physical time
# against analytical solution

using SpecialFunctions
using JLD
using Statistics, Printf, LinearAlgebra, Plots, Plots.Measures, LaTeXStrings


const NUMERICAL = true




# visualize the analytical solution
""" Uniform pore fluid pressure distribution
Reference: Self-similar fault slip in repsponse to fluid injection (Viesca 2008)

- where ld(t) = √(ɑ * t) is the diffusivity length scale
"""
function P(x::Float64, t::Float64)

    P₀  = 5e6         # [Pa] initial pore pressure                   p(x,0)    = P₀ 
    Δpf = 5e6         # [Pa] injection proceeds at constant pressure p(0, t>0) = Δp
    ηf  = 1e-3        # [Pa·s] viscosity of the permeating fluid
    kᵩ  = 1e-15       # [m²]   Darcy permeability of the layer    (table 1 value) -> calcuated from kᵩ = k* (φ)
    
    # calculated from values in table 1 with βs = 2.5e-11, βf = 4.0e-10 (see Dal Zilio et al. 2022)
    βd  = 2.5555555555555557e-11   # drained compressibility of the porous medium
        
    ɑₕ = kᵩ / (ηf * βd)# hydraulic diffusivity
    ɑ = 4 * ɑₕ
    

    return P₀ + Δpf * erfc(norm(x) / sqrt(ɑ * t))
end

# calculate for 25 grid points spanning from x ∈ [0, 50]
nx = 1001                      # this has to be identical to grid resolution
X  =  LinRange(0,50,nx)

function injection_benchmark()

    # setup for plots
    default(size=(1200,830), margin=8mm, linewidth=2)
    scalefontsizes(); scalefontsizes(1.20)

    plot(;xlims=(0, 50), ylims = (4,10.8), framestyle= :box, seriesstyle= :path, aspect_ratio = 5.0, title="Fluid injection benchmark (BP1) " * L"(\Delta t = 5 s)", 
            xlabel = "Distance from the injection point [m]", ylabel="Fluid Pressure [MPa]")

    T     = [500.0, 1000.0, 2000.0, 4000.0]
    label = ["500 sec", "1000 sec", "2000 sec", "4000 sec"]


    #======ANALYTICAL======#
    for i in 1:4  # unit: [s]
        Pressure = []
        t = T[i]

        Pressure = P.(X,t)
        Pressure /= 1e6

        # plotting with unit [MPa x m]
        plot!(X, Pressure; linestyle= :dash, label = label[i])


    end


    # #======NUMERICAL======#
    # # read the data from the experiment
    if NUMERICAL
        # label    = ["500 sec", "1000 sec", "2000 sec", "4000 sec"]
        iter     = [500, 1000, 2000, 4000]
        @. iter /=  5     # 
        color=[:skyblue,:orange,:green,:purple]

        # iterate over 4 different results obtained at different time points
        for i in 1:4
            Pf_fault = load("fluid_injection/Pf_fault" * string(iter[i]) * ".jld")["data"]
            Pf_fault /= 1e6


            X_selected        = []
            Pf_fault_selected = []

            for j in 1:size(Pf_fault)[2]

                if mod(j, 30) == 0
                    push!(X_selected, X[j])
                    push!(Pf_fault_selected, Pf_fault[j])

                end

            end            

            # plotting with unit [MPa x m]
            plot!(X_selected, Pf_fault_selected; seriestype= :scatter, label="", markercolor = color[i])

        end

    end


    # save the figure
    savefig("fluid_injection_benchmark.png")

end




injection_benchmark()