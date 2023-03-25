# NOTE: kᵩ  = 1e-17  [m²] was given in fig description but does not reproduce the curve when testing with analytical solution
# thus the value kᵩ  = 1e-16 calculated from the table 1 value shall be used instead

using SpecialFunctions
using JLD
using Statistics, Printf, LinearAlgebra, Plots, Plots.Measures



const NUMERICAL = true


P₀  = 5e6         # [Pa] initial pore pressure                   p(x,0)    = P₀ 
Δpf = 5e6         # [Pa] injection proceeds at constant pressure p(0, t>0) = Δp
ηf  = 1e-3        # [Pa·s] viscosity of the permeating fluid
kᵩ  = 1e-16       # [m²]   Darcy permeability of the layer    (table 1 value) -> calcuated from kᵩ = k* (φ)


# calculated from values in table 1  
# βd = ɸ * (βs + βf) = 0.01 * (2.5e-11 + 4.0e-10) | with βs = 2.5e-11, βf = 4.0e-10
βd  = 4.25e-12   # drained compressibility of the porous medium




ɑₕ = kᵩ / (ηf * βd)# hydraulic diffusivity
@show ɑ = 4 * ɑₕ


# visualize the analytical solution
""" Uniform pore fluid pressure distribution
Reference: Self-similar fault slip in repsponse to fluid injection (Viesca 2008)

- where ld(t) = √(ɑ * t) is the diffusivity length scale
"""
function P(x::Float64, t::Float64)
    return P₀ + Δpf * erfc(norm(x) / sqrt(ɑ * t))
end


function injection_benchmark()

    # calculate for 25 grid points spanning from x ∈ [0, 50]
    X =  LinRange(0,50,100)

    # setup for plots
    default(size=(1200,800), margin=8mm)
    plot(;xlims=(0, 50), ylims = (4,10.8), framestyle= :box, seriesstyle= :path, aspect_ratio = 5.0, title="Fluid injection benchmark (BP1) (dt = 5 s)", 
            xlabel = "Distance from the injection point [m]", ylabel="Fluid Pressure [MPa]")

    T     = [500.0, 1000.0, 2000.0, 4000.0]
    label = ["500 sec", "1000 sec", "2000 sec", "4000 sec"]


    #======ANALYTICAL======#
    for i in 1:4  # unit: [s]
        Pressure = []
        t = T[i]

        for x in X
            p = P(x, t)          # obtain analytical solution for specified x and t
            push!(Pressure, p)        
        end

        Pressure /= 1e6

        # plotting with unit [MPa x m]
        plot!(X, Pressure; linestyle= :dash, label = label[i])


    end


    # #======NUMERICAL======#
    # # read the data from the experiment
    if NUMERICAL
        @show X  = LinRange(0, 50, 500)
        label = ["500 sec", "1000 sec", "2000 sec", "4000 sec"]
        iter = [500, 1000, 2000, 4000]
        color=[:skyblue,:orange,:green,:purple]

        # iterate over 4 different results obtained at different time points
        for i in 1:4
            Pf_fault = load("fluid_injection/Pf_fault" * string(iter[i]) * ".jld")["data"]
            Pf_fault /= 1e6


            # DEBUG
            # @show Pf_fault[1:500]
            # println(size(Pf_fault))


            # plotting with unit [MPa x m]
            plot!(X, Pf_fault[1:500]; seriestype= :scatter, label = label[i], markercolor = color[i])

        end

    end


    # save the figure
    savefig("fluid_injection_benchmark.png")

end




injection_benchmark()