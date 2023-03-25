using Plots, Plots.Measures, JLD, ElasticArrays
using LaTeXStrings

default(size=(600,1000),fontfamily="Computer Modern", linewidth=4, framestyle=:box, margin=5mm)
scalefontsizes(); scalefontsizes(1.1)


function sensitivity_analysis()

    # varying system size for 4 experiments in all
    # nx_array = exp2.(range(6,11,step=1))
    # ny_array = exp2.(range(7,12,step=1))


    # x-axis - plot against time
    # for computing dimensionless time τc for nondimensionalize of the time
    ηc = 1.0
    ρs = 2.0
    ρf = 1.0
    g  = 1.0

    k0 = 1.0
    µf = 1.0

    δc = √(k0 * ηc/µf)       
    pc = (ρs - ρf) * g * δc  # characteristic pressure/stress

    τc = ηc / pc             # dimensionless time


    # data are array of tuples, each tuple has the storage format (t,ɸmax, Δh(ɸmax))   
    experiments = Vector{ElasticArray}(undef, 4)
    experiments[1] = load("2D/sen_ex01.jld")["data"]
    experiments[2] = load("2D/sen_ex02.jld")["data"]
    experiments[3] = load("2D/sen_ex03.jld")["data"]
    # experiments[4] = load("2D/sen_ex04.jld")["data"]
    # experiments[5] = load("2D/sen_ex05.jld")["data"]

    labels = ["PT 64x128" "PT 128x256" "PT 256x512" "PT 512x1024" "PT 1024x2048"]
    
    p1 = plot(title="Normalized maximal porosity",xlabel="t/τc", ylabel= L"\frac{\Phi_{max}}{\Phi_0}")
    p2 = plot(title="Vertical distance from initial pertubation", xlabel="t/τc", ylabel=L"\Delta h (\Phi_{max})")
    
    for number in 1:3
        
        # x-axis
        time_array     = experiments[number][1,:]
    
        # y-axis
        porosity_array = experiments[number][2,:]
        height_array   = experiments[number][3,:]
    
        time_array /= τc
    
    
        # y-axis plot1 the normalized maximal porosity
        ɸ0   = 0.01  # reference porosity
    
        # read from the results
        porosity_array /= ɸ0      # maximal porosity
    
    
        # y-axis plot2 the vertical distance from initial pertubation
        plot!(p1, time_array, porosity_array; label=labels[number], marker= :circle, markerstrokestyle= :dash)     
        plot!(p2, time_array, height_array;  label=labels[number], marker= :circle, markerstrokestyle= :dash)    

    end


    display(plot(p1, p2; layout=(2, 1)))
    savefig("sensitivity_analysis.png")


end


if isinteractive()
    sensitivity_analysis()
end