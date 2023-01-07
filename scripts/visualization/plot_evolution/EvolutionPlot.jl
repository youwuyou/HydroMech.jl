using Plots, Plots.Measures, JLD, ElasticArrays, LinearAlgebra
using LaTeXStrings


const WITH_COLORBAR = true


function plot_finalstates(;res_)
    
    res        = res_                         # resolution used
    states     = 6                            # number of final states to be plotted
    ra         = 2.0                          # radius of initial porosity perturbation
    lx         = 20.0                         # domain size x
    ly         = ra * lx                      # domain size y
    
    nx,ny = res-1, ra*res-1 # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf

    dx, dy   = lx/(nx-1), ly/(ny-1)              # grid step in x, y

    X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)

    
    # data are array of tuples, each tuple has the storage format (t,ɸmax, Δh(ɸmax))   
    # precomputation of data
    ɸ0 = 0.01 # reference porosity

    
    # labels = ["PT 63x127" "PT 127x255" "PT 511x1023" "PT 1023x2047" "PT 2047x4095"]
    p = plot(layout = (2,3))   

    default(size=(1000,1000),
            fontfamily="Computer Modern", 
            # clims=(-1.15,1.4), 
            clims = (0.0, 1.2),
            c = cgrad(:turbo, scale = :exp),
            linewidth=4, 
            framestyle=:box,
            xlims=(X[1],X[end]), 
            ylims=(Y[1],Y[end]),
            margin=(3,:mm))           
    
    scalefontsizes(); scalefontsizes(1.1)
    
    experiments = Vector{ElasticArray}(undef, states)
    filenames   = Vector{String}(undef, states)

    iterations = [3, 360, 660, 960, 2160, 3294]

    annotations = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


    for i in 1:states
        filenames[i] = "results/porosity" * string(iterations[i], pad=5) * ".jld"
        experiments[i]  = log10.(load(filenames[i])["data"] / ɸ0)

        title_ = "it = " * string(iterations[i])

        heatmap!(p[i], X, Y,  experiments[i], aspect_ratio=1, annotationcolor= :white, annotationfontsize = 22, annotation= (3,3,annotations[i]), showaxis= false, title= title_, colorbar = false)
    end

    if WITH_COLORBAR
        # specify layout
        l = @layout [grid(1,1) a{0.1w}]

        # use the colorbar hack
        colorbar = scatter([0,0], [0,1], zcolor=[0,3], xlims=(1,1.1), axis=false, label="", grid=false)
        Plots.gr_cbar_width[] = 0.02
        plot(p, colorbar, layout=l)
    else
        # without colorbar
        l = @layout [a b c; d e f] 
        plot(p, layout=l)
    end


    savefig("PW_evolution.png")
end

if isinteractive()
    plot_finalstates(res_=2048)    # plot final state for the very-high resolution case
end