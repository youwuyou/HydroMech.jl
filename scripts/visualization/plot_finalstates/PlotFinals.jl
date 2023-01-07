using Plots, Plots.Measures, JLD, ElasticArrays, LinearAlgebra
using LaTeXStrings



function plot_finalstates(;res_)
    
    res        = res_                         # resolution used
    states     = 5                            # number of final states to be plotted
    ra         = 2.0                          # radius of initial porosity perturbation
    lx         = 20.0                         # domain size x
    ly         = ra * lx                      # domain size y
    
    nx     = Vector{Int32}(undef, states)
    ny     = Vector{Int32}(undef, states)
    
    dx     = Vector{Number}(undef, states)
    dy     = Vector{Number}(undef, states)
    
    X     = Vector{ElasticArray}(undef, states)
    Y     = Vector{ElasticArray}(undef, states)
    
    
    for i in 1:states
        
        if i <= 2
            res = 2^(i+5)
        else
            res = 2^(i+6)
        end
        
        nx[i], ny[i]   = res-1, ra*res-1 # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf
        dx[i], dy[i] =  lx/(nx[i]-1), ly/(ny[i]-1)
        X[i] , Y[i]  =  0:dx[i]:lx, 0:dy[i]:ly 
        
    end
    
    
    # data are array of tuples, each tuple has the storage format (t,ɸmax, Δh(ɸmax))   
    # precomputation of data
    ɸ0 = 0.01 # reference porosity

    
    # labels = ["PT 63x127" "PT 127x255" "PT 511x1023" "PT 1023x2047" "PT 2047x4095"]
    p = plot(layout = (1,5))   

    default(size=(3200,1100),
            fontfamily="Computer Modern", 
            # clims=(-1.1,1.4),
            clims = (0.0, 1.2),
            c = cgrad(:turbo, scale = :exp),
            linewidth=4, 
            framestyle=:box,
            xlims=(X[1][1],X[1][end]), 
            ylims=(Y[1][1],Y[1][end]),
            top_margin=3mm)           
    
    scalefontsizes(); scalefontsizes(2.1)
    
    experiments = Vector{ElasticArray}(undef, states)
    filenames   = Vector{String}(undef, states)
    annotations = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    
    for i in 1:states
        filenames[i] = "results/final" * string(i, pad=2) * ".jld"
        experiments[i]  = log10.(load(filenames[i])["data"] / ɸ0)
        # experiments[i]  = load(filenames[i])["data"] / ɸ0


        # title_ = "nx = " * string(nx[i]) * ", ny = " * string(ny[i])
        annotations[i] *= "  nx = " * string(nx[i]) * ", ny = " * string(ny[i])

        heatmap!(p[i], X[i], Y[i],  experiments[i], aspect_ratio=1, annotationcolor= :white, annotationfontsize = 25, annotation= (10,3,annotations[i]), showaxis= false, colorbar = false)
    end

    # specify layout => 
    l = @layout [grid(1,1) a{0.1w}]

    colorbar = scatter([0,0], [0,1], zcolor=[0,3], xlims=(1,1.1), axis=false, label="", grid=false)

    Plots.gr_cbar_width[] = 0.01
    
    plot(p, colorbar, layout=l, plot_title="Normalized Porosity at time " * L"t = 0.02")

    savefig("final_states.png")



end

if isinteractive()
    plot_finalstates(res_=64)
end