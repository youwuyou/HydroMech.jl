using HydroMech

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)


# NOTE: despite of using the package we initialize here again because 
# we need to use the type Data.Array, Data.Number for argument types
const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using Plots, Plots.Measures


@views function domain_setup()


    # MESH
    lx       = 40000.0  # [m] 40 km
    ly       = 20000.0  # [m] 20 km
    nx       = 321
    ny       = 161

    dx, dy   = lx/(nx-1), ly/(ny-1)   # grid step in x, y
    @show mesh     = PTGrid((nx,ny), (lx,ly), (dx,dy))
    _dx, _dy = inv.(mesh.di)
    max_nxy  = max(nx,ny)
    min_dxy2 = min(dx,dy)^2
        
    # index for accessing the corresponding row of the interface
    @show h_index  = Int((ny - 1) / 2) + 1 # row index where the properties are stored for the fault

    
    # calculate for 25 grid points spanning from x ∈ [0, 50]
    X = LinRange(0.0, lx, nx-1)
    @show size(X)

    Y = LinRange(0.0, ly, ny-1)
    @show size(Y)


    # X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
    # Xv          = (-dx/2):dx:(lx+dx/2)


    # RATE_AND_STATE_FRICTION
    Vp          = @zeros(nx-1, ny-1)
    F           = @zeros(nx-1, ny-1)
    σyield      = @zeros(nx-1, ny-1)
    ɛ̇II_plastic = @zeros(nx-1, ny-1)
    Bool_cpu    = fill(false, nx-1, ny-1)


    # Parameters for rsf
    #            domain   fault
    a0        = [0.018    0.008]     # a-parameter of RSF
    b0        = [0.001    0.016]     # b-parameter of RSF
    Ω0        = [15           1]     # State variable from the preνous time step
    L0        = [0.012    0.012]     # L-parameter of RSF (characteristic slip distance)
    V0        = 1e-9                 # Reference slip velocity of RSF, m/s
    γ0        = 0.6                  # Ref. Static Friction
    Wh        = dy                   # fault width
    σyieldmin = 1e-3

    # assign along fault [:, h_index] for rate-strengthing/weakening regions        
    #    0km   4km    6km                 34km    36km  40km
    #    x0    x1     x2                   x3     x4    x5
    #    |*****|xxxxxx|                    |xxxxxx|*****|  
    #    -----------------------------------------------

    x0   = 0.0
    x1   = 4.0e3
    x2   = 6.0e3
    x3   = 34.0e3
    x4   = 36.0e3
    x5   = 40.0e3

    # Use the velocity strengthening parameters as a-b>0 on the left and right extreme of the fault, for a length of 4 km each:
    # from 0 to 4 km in x-direction
    # from 36 to 40 km in x-direction
    a_cpu       = fill(a0[1], nx-1, ny-1)
    b_cpu       = fill(b0[1], nx-1, ny-1)
    Ω_cpu       = fill(Ω0[1], nx-1, ny-1)

    # L0 is identical on both regions
    L_cpu       = fill(L0[1], nx-1, ny-1)



    for i in 1:1:nx-1
        for j in 1:1:ny-1

            # if along the fault
            if j == h_index

                # assign domain value
                if x0 <= X[i] <= x1 || x4 <= X[i] <= x5
                    a_cpu[i,j] = a0[1]
                    b_cpu[i,j] = b0[1]
                    Ω_cpu[i,j] = Ω0[1]
                end

                # assign fault value
                if x2 <= X[i] <= x3
                    a_cpu[i,j] = a0[2]
                    b_cpu[i,j] = b0[2]
                    Ω_cpu[i,j] = Ω0[2]
                end

                # assign transition zone value (left)
                if x1 < X[i] < x2
                    a_cpu[i, j] = a0[1] - (a0[1] - a0[2]) * ((X[i] - x1) / (x2 - x1))
                    b_cpu[i, j] = b0[1] - (b0[1] - b0[2]) * ((X[i] - x1) / (x2 - x1))
                    Ω_cpu[i, j] = Ω0[1]
                end

                if x3 < X[i] < x4
                    a_cpu[i, j] = a0[2] - (a0[2] - a0[1]) * ((X[i] - x3) / (x4 - x3))
                    b_cpu[i, j] = b0[2] - (b0[2] - b0[1]) * ((X[i] - x3) / (x4 - x3))
                    Ω_cpu[i, j] = Ω0[1]
                end

            end

        end
    end

    a           = PTArray(a_cpu)
    b           = PTArray(b_cpu)
    Bool        = PTArray(Bool_cpu)
    Ω           = PTArray(Ω_cpu)
    L           = PTArray(L_cpu)


    # plotting properties
    default(size=(1200,1000),fontfamily="Computer Modern", linewidth=2, framestyle=:box, margin=8mm)
    scalefontsizes(); scalefontsizes(1.35)

    p1 = plot(X/1e3, Array(a[:, h_index]); xlims=(0,40), ylims=(0.0, 0.03), yticks=[0.008, 0.018], title="Parameter a of RSF")
    p2 = plot(X/1e3, Array(b[:, h_index]); xlims=(0,40), ylims=(0.0, 0.03), yticks=[0.001, 0.016], title="Parameter b of RSF")
    p3 = plot(X/1e3, Array(Ω[:, h_index]); xlims=(0,40), ylims=(0.0, 20), yticks=[1.0, 15.0], title="Parameter omega of RSF")
    p4 = plot(X/1e3, Array(L[:, h_index]); xlims=(0,40), ylims=(0.0, 0.02), yticks=[0.0, 0.012], title="Parameter b of RSF")


    display(plot(p1, p2, p3, p4; layout=(4,1)))


    savefig("domain.png")
end  




domain_setup()