using Plots, Plots.Measures, JLD, Printf, LinearAlgebra
using LaTeXStrings

default(size=(4000,5600),fontfamily="Computer Modern", linewidth=4, framestyle=:box, margin=20mm)
scalefontsizes(); scalefontsizes(5.0)


const ONLY_POROSITY = true


function plot_intermediate(;iterations_, res_)
    ENV["GKSwstype"]="nul"; if isdir("high_reso")==false mkdir("high_reso") end; loadpath = "./high_reso/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")

    iterations = iterations_   # number of iterations performed
    res        = res_          # resolution used

    ra = 2.0         # radius of initial porosity perturbation
    lx = 20.0        # domain size x
    ly = ra * lx     # domain size y

    nx,ny = res-1, ra*res-1 # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf

    dx, dy   = lx/(nx-1), ly/(ny-1)              # grid step in x, y

    X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)


    # precomputation of data
    ɸ0 = 0.01 # reference porosity


    for it in 1:iterations
        println(String("Iteration: $(it)"))

        if mod(it, 3) == 0

            filename1 = "intermediate/2D/poro/porosity" * string(it, pad=5) * ".jld"
            Phi  = load(filename1)["data"]
            Phi  = log10.(Phi / ɸ0)
            
            if ONLY_POROSITY == false
                filename2 = "intermediate/2D/pressure/effective_pressure" * string(it, pad=5) * ".jld"
                filename3 = "intermediate/2D/darcy/darcy_vert" * string(it, pad=5) * ".jld"
                filename4 = "intermediate/2D/velo/velo_vert" * string(it, pad=5) * ".jld"
                
                Peff = load(filename2)["data"]
                qDy  = load(filename3)["data"]
                Vy   = load(filename4)["data"]
                
                # plot all four stored values as subplots
                p1 = heatmap(X, Y,  Phi  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=::default, title="porosity")
                p2 = heatmap(X, Y,  Peff,  aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="effective pressure")
                p3 = heatmap(X, Yv, qDy  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical Darcy flux")
                p4 = heatmap(X, Yv, Vy   , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical velocity")
                display(plot(p1, p2, p3, p4)); frame(anim)
            else 
                # plotting only the porosity wave
                display(heatmap(X, Y,  Phi, aspect_ratio=1, ylabel=L"\log_{10}\left( \frac{\phi}{\phi_0} \right)", xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), title="Normalized Porosity Distribution")); frame(anim)
            end


            

        end

    end


    gif(anim, "PW_high_resolution.gif", fps=15)
    
end



plot_intermediate(;iterations_=2065, res_=512)