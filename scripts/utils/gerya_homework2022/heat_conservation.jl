using LinearAlgebra, Plots, SparseArrays, BasicLU

ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
println("Animation directory: $(anim.dir)")

DO_VIZ = true



function heat_conservation()

    xsize = 100000
    ysize = 100000
    Nx    = 35
    Ny    = 45

    dx    = xsize/(Nx-1)
    dy    = ysize/(Ny-1)
    min_dx_dy = min(dx,dy)

    xpr = -dx/2:dx:xsize+dx/2
    ypr = -dy/2:dy:ysize+dy/2

    T0     = zeros(Ny+1, Nx+1)
    Tdt    = zeros(Ny+1, Nx+1)
    RHOcp  = zeros(Ny+1, Nx+1)

    # physics
    # constant thermal conductivity
    k = 3.                   # [W/m/K]
    T_int = 1773             # [K]
    T_ext = 1573             # [K]
    rhocp_int = 1.1 * 3.2e6  # [J/[K·m³]]
    rhocp_ext = 3.3e6        # [J/[K·m³]]
    R = 20000                # [m] = 20 [km]

    # properties defined on the pressure nodes
    for j in 1:1:Nx+1
        for i in 1:1:Ny+1
            r = ((xpr[j]-xsize/2)^2 + (ypr[i]-ysize/2)^2)^0.5

            if r < R
                # println["internal"]
                T0[i,j] = T_int
                RHOcp[i,j] = rhocp_int
            else
                # println["external"]
                T0[i,j] = T_ext
                RHOcp[i,j] = rhocp_ext
            end

        end
    end



    #===============TIME STEPPING START===============#

    # index of matrix
    n = (Nx+1) * (Ny+1)
    L = zeros(n,n)
    R = zeros(n)

    # time stepping
    ntimesteps = 10
    dt         = min_dx_dy^2 / (4 * k / findmin(RHOcp)[1])



    for _ in 1:1:ntimesteps

        # composition of the matrix
        for j in 1:1:Nx+1
            for i in 1:1:Ny+1

                # define global index g
                g = (j-1) * (Ny+1)+ i

                if i==1 || i==Ny+1 || j==1 || j==Nx+1
                    if j == 1 && i > 1 && i < Ny+1
                        # external nodes at the left boundary [Neumann]
                        L[g,g]        = 1
                        L[g,g+(Ny+1)] = -1
                        R[g]          = 0
                    end
                    
                    if j==Nx+1 && i > 1 && i < Ny+1
                        # external nodes at the right boundary [Neumann]
                        L[g,g]        = 1
                        L[g,g-(Ny+1)] = -1
                        R[g]          = 0
                    end

                    if i == 1
                        # external nodes at the upper boundary [Dirichlet]
                        # ERROR !!!: wrong BC
                        #  L[g,g]       = 1
                        L[g,g]       = 1/2
                        L[g,g+1]     = 1/2
                        R[g]         = T_ext
                    end

                    if i == Ny+1
                        # external nodes at the lower boundary [Dirichlet]
                        # ERROR !!!: wrong BC
                        #  L[g,g]       = 1
                        L[g,g]       = 1/2
                        L[g,g-1]     = 1/2
                        R[g]         = T_ext
                    end
            
                else
                    # internal: heat conservation equation
                    #  ρCp· DT/Dt = k· [∂²/∂x² T + ∂²/∂y² T]
                            
                    #             [i-1,j]
                    #             T2 
                    #             |
                    #             |
                    #  [i,j-1]   [i,j]     [i,j+1] 
                    #  T1--------T3--------T5
                    #            |
                    #            |
                    #            [i+1,j]
                    #           T4  

                    # Timestepping formulation
                    #  T3/dt - k/ρCp [ [T1 - 2T3 + T5]/dx²  +  [T2 - 2T3 + T4]/dy²] = T3⁰/dt
                    # NOTE: use the RHOcp obtained at the T3 node!!
                    L[g,g-(Ny+1)] = -k/RHOcp[i,j] / dx^2               # T1
                    L[g,g-1]      = -k/RHOcp[i,j] / dy^2               # T2
                    L[g,g]        = 1. /dt + 2. * k /RHOcp[i,j] / dx^2 + 2. * k /RHOcp[i,j] / dy^2    # T3
                    L[g,g+1]      = -k/RHOcp[i,j] / dy^2           # T4
                    L[g,g+(Ny+1)] = -k/RHOcp[i,j] / dx^2           # T5

                    # RHS
                    R[g] = T0[i,j] / dt
                end


            end
        end  # end of matrix composition

        L_sparse = sparse(L)
        S        = L_sparse \ R

        # S = L \ R


        for j in 1:1:Nx+1
            for i in 1:1:Ny+1
                # global index for current node
                g = (j-1) * (Ny+1) + i
                Tdt[i,j] = S[g]
            end
        end

        # T0 = copy(Tdt)
        T0 = Tdt


        if DO_VIZ
            p1 = heatmap(xpr, ypr,  RHOcp, aspect_ratio=1, xlims=(xpr[1],xpr[end]), ylims=(ypr[1],ypr[end]), c=:turbo, title="Volumetric isobaric heat capacity")
            p2 = heatmap(xpr, ypr,  Tdt  , aspect_ratio=1, xlims=(xpr[1],xpr[end]), ylims=(ypr[1],ypr[end]), c=:turbo, title="Temperature")

            display(plot(p1, p2, layout=(2,1))); frame(anim)
        end



    end # end of the timestepping

    value = Tdt[17,15] # [1704.92596509904], K=3, 2D 100x100km, 35x45  implicit
    println(value)


    gif(anim, "Heat_conservation.gif", fps = 2)


end


heat_conservation()