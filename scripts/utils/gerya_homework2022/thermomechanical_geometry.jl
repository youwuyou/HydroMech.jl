using Plots



function main()

    # 2). define numerical model
    # Eulerian 2D grid
    xsize = 100000;             # Horizontal size, [m]
    ysize = 100000;             # Vertical size,   [m]

    # NOTE: (Nx - 1) * (Ny - 1) == 1496
    Nx = 35;
    Ny = 45;
    dx = xsize/(Nx-1);     # Horizontal grid step, [m]
    dy = ysize/(Ny-1);     # Vertical grid step,   [m]


    # i-b). basic nodes
    x   = 0:dx:xsize+dx;        # Horizontal   [m]
    y   = 0:dy:ysize+dy;        # Vertical     [m]

    # ii). Vx-nodes
    xvx = 0:dx:xsize+dx;         # Horizontal   [m]
    yvx = -dy/2:dy:ysize+dy/2;   # Vertical     [m]

    # iii). Vy-nodes
    xvy = -dx/2:dx:xsize+dx/2;   # Horizontal  [m]
    yvy = 0:dy:ysize+dy;         # Vertical    [m]

    # iv). P-nodes
    xpr = -dx/2:dx:xsize+dx/2;    # Horizontal  [m]
    ypr = -dy/2:dy:ysize+dy/2;    # Vertical    [m]

    # Lagrangian 2D markers
    # only one array not 2 for 2D
    # 25 markers per 2D cell -> check by 37400 / 1496 == 25
    Nxm = (Nx-1) * 5;            # no. of markers in x-direction at initial state  -> 170
    Nym = (Ny-1) * 5;            # no. of markers in y-direction at initial state  -> 220
    Nm  = Nxm * Nym;             # no. of markers over the 2D grid                 -> 37400


    # array initializations
    # i-a). markers
    xm   = zeros(1, Nm);
    ym   = zeros(1, Nm);
    dxm  = xsize/Nxm;            # distance between the markers, not Nxm-1 for more distance to the boundary
    dym  = ysize/Nym;

    # values on markers
    RHOm   = zeros(1, Nm);    # Density   ρₘ
    ETAm   = zeros(1, Nm);    # Viscosity ηₘ
    Km     = zeros(1, Nm);    # Conductivity kₘ NEW:
    RHOcpm = zeros(1, Nm);    # Volumetric heat capacity ρCpₘ
    Tm     = zeros(1, Nm);    # Temperature Tₘ


    # Interpolated properties
    # - mechanical properties
    RHOvy = zeros(Ny+1,Nx+1);
    ETAb  = zeros(Ny+1,Nx+1);
    ETAp  = zeros(Ny+1,Nx+1);

    # - thermal properties NEW:
    T0     = zeros(Ny+1, Nx+1);
    Tdt    = zeros(Ny+1, Nx+1);
    Kvx    = zeros(Ny+1, Nx+1);
    Kvy    = zeros(Ny+1, Nx+1);
    RHOcpp = zeros(Ny+1, Nx+1);
    T0p    = zeros(Ny+1, Nx+1); 

    # Flux and temperature
    qx     = zeros(Ny+1, Nx+1);
    qy     = zeros(Ny+1, Nx+1);


    # Arrays for the solutions Vx, Vy, Pr of size Ny+1, Nx+1
    # calculated from solving the stokes => define later for efficiency


    # physics
    R         = 20000.0;           # [m] = 20 [km]
    gy        = 10.0;              # [m/s^2]  Vertical gravity 
    eta_int   = 1e18;            # [Pa · s]
    eta_ext   = 1e19;

    rho_int   = 3200.0;            # [kg/m^3]
    rho_ext   = 3300.0;

    # NEW: assign values to the new markers for K, RHOcp, T
    k_int     = 2.0;             # [W/m/K] 
    k_ext     = 3.0;    
    rhocp_int = 3.52e6;          # [J/(K·m³)]
    rhocp_ext = 3.3e6;        
    T_int     = 1773.0;            # [K]
    T_ext     = 1573.0;                


    # Sticky air <-> adding a fluid layer at the top (low density and low viscosity)
    # with sufficiently large thickness
    # FOR RHO
    # water <-> 1 kg/m^3
    # air   <-> 1000 kg/m^3
    # FOR ETA
    # and fluid viscosity much larger than normally => ensures low stress
    eta_air     = 1e+17;
    # eta_air   = 1e+15;
    rho_air     = 1.0;

    # NEW: assign values to the new markers for K, RHOcp, T
    k_air       = 3000.0;
    rhocp_air   = 3.3e+6;
    T_air       = 273.0;

    # Lithosphere (NEW!)
    rho_litho   = 3350;
    eta_litho   = 1e+21;
    k_litho     = 4.0;
    rhocp_litho = 3.015e+5;   # CP = 900

    # T_litho = @(y) (2/13 * 100)*y+(273.0 - 4/13e+6);
    T_litho(y) = -((T_air - T_ext)/(0.8*ysize))*(y-20000)+273;

    # T_litho = @(y) -((T_air - T_ext)/(0.2 * ysize))*(y-20000) + 273;
    # T_litho   = 500;

    m = 1;    # start of the index m ∈ {1,2,...,Nm} 
    for jm in 1:1:Nxm
        for im in 1:1:Nym
            
            # NON-RANDOM
            xm[m] = dxm * 0.5 + (jm-1) * dxm;
            ym[m] = dym * 0.5 + (im-1) * dym;
            

            # RANDOM
            # NOTE: (rand-0.5) * dxm adds some small displacement
            # xm[m] = dxm * 0.5 + (jm-1) * dxm + (rand - 0.5) * dxm;
            # ym[m] = dym * 0.5 + [im-1) * dym + (rand - 0.5) * dym;
            
            # prescribe initial properties to RHOm, ETAm
            # rmark=((xm[m]-xsize/2)^2+(ym[m]-ysize/2)^2)^0.5;
            rmark=((xm[m]-xsize/2)^2+(ym[m]-ysize*0.3)^2)^0.5;  # (NEW!) move the plume downwards
            
            if rmark < R
                # plume
                RHOm[m]   = rho_int;
                ETAm[m]   = eta_int;
                
                # Heat conservation
                Km[m]     = k_int;
                RHOcpm[m] = rhocp_int;
                Tm[m]     = T_int;
                
            else
                # mantle
                RHOm[m] = rho_ext;
                ETAm[m] = eta_ext;
                
                # Heat conservation
                Km[m]     = k_ext;
                RHOcpm[m] = rhocp_ext;
                Tm[m]     = T_ext;
            end
            
            # lithosphere
            if 0.6 * ysize < ym[m] < 0.8 * ysize
                RHOm[m]   = rho_litho;
                ETAm[m]   = eta_litho;

                # Heat conservation
                Km[m]     = k_litho;
                RHOcpm[m] = rhocp_litho;
                Tm[m]     = T_litho(ym[m]);   # obtain the T_litho depending on y-coordinate
                # Tm[m]     = T_litho;   # obtain the T_litho depending on y-coordinate
            end
            
            # sticky air
            if ym[m] > 0.8 * ysize
                RHOm[m] = rho_air;
                ETAm[m] = eta_air;   # sticky values, real value would be 1e-6

                # Heat conservation NEW:
                Km[m]     = k_air;
                RHOcpm[m] = rhocp_air;
                Tm[m]     = T_air;
            end

            
            m = m+1;
        end
    end

    RHOtopo = (3300 + 1) / 2;


    # basic nodes
    WTSUMb        = zeros(Ny+1, Nx+1); # => used to store the denominator
    ETAWTSUMb     = zeros(Ny+1, Nx+1); # => used to store the numerator for ETA

    # vx nodes
    WTSUMvx       = zeros(Ny+1, Nx+1);
    KVXWTSUMvx    = zeros(Ny+1, Nx+1); # Kvx

    # vy nodes
    WTSUMvy      = zeros(Ny+1, Nx+1); 
    RHOWTSUMvy   = zeros(Ny+1, Nx+1);  # => used to store the numerator for RHO
    KVYWTSUMvy    = zeros(Ny+1, Nx+1); # Kvy

    # pressure nodes
    WTSUMp       = zeros(Ny+1, Nx+1);
    WTSUMp_t     = zeros(Ny+1, Nx+1);  # Special weight sum for T
    ETAWTSUMp    = zeros(Ny+1, Nx+1);  # ETAp
    RHOCPWTSUMp  = zeros(Ny+1, Nx+1);  # RHOcpp
    TWTSUMp      = zeros(Ny+1, Nx+1);  # T0p

    

    # Go through all the markers on the 2D grid
    # => compute contribution of the markers to 4 surrounding nodal points
    for m in 1:1:Nm
        
        # edge case check: only interpolate markers within the grid
        if xm[m] >= 0 && xm[m] <= xsize && ym[m] >= 0 && ym[m] <= ysize
            

            # 1). Basic nodes:  ETAb
            #   ηₘ => ETAb     this is basic-node
            #                     /
            #   ETAb[i, j]       o--------o ETAb[i, j+1]
            #                  | * m    |
            #                 |        |
            #   ETAb[i+1, j]  o--------o ETAb[i+1, j+1]       

            # Indices
            i = round(Int,(ym[m] - y[1])/dy) + 1;
            j = round(Int,(xm[m] - x[1])/dx) + 1;

            # obtain relative distance ΔXm(j] from the marker to node
            q_x = abs(xm[m]-x[j])/dx;   # distance quotient in x-direction
            q_y = abs(ym[m]-y[i])/dy;   # distance quotient in y-direction
            
            # adding up weights
            Wtmij   = (1. - q_x) * (1. - q_y);
            Wtmi1j  = (1. - q_x) * q_y;
            Wtmij1  =  q_x * (1. - q_y);
            Wtmi1j1 =  q_x * q_y;

            WTSUMb[i,j]        = WTSUMb[i,j] + Wtmij;        # Upper left  ↔ Wtmij
            WTSUMb[i, j+1]     = WTSUMb[i, j+1] + Wtmij1;    # Upper right ↔ Wtmi1j
            WTSUMb[i+1,j]      = WTSUMb[i+1,j] + Wtmi1j;     # Lower left  ↔ Wtmij1
            WTSUMb[i+1,j+1]    = WTSUMb[i+1,j+1] + Wtmi1j1;  # Lower right ↔ Wtmi1j1

            # ETAb
            ETAWTSUMb[i,j]     = ETAWTSUMb[i,j] + ETAm[m] * Wtmij;          # Upper left ↔ Wtmij
            ETAWTSUMb[i,j+1]   = ETAWTSUMb[i,j+1] + ETAm[m] * Wtmij1;       # Upper right ↔ Wtmi1j
            ETAWTSUMb[i+1,j]   = ETAWTSUMb[i+1,j] + ETAm[m] * Wtmi1j;       # Lower left ↔ Wtmij1
            ETAWTSUMb[i+1,j+1] = ETAWTSUMb[i+1,j+1] + ETAm[m] * Wtmi1j1;    # Lower right ↔ Wtmi1j1


            # 2). Vx-node:  Kvx
            #     ρₘ => RHO       this is vx-node
            #                     /
            #    Kvx[i, j]       o--------o Kvx[i, j+1]
            #                   | * m    |
            #                  |        |
            #    Kvx[i+1, j]  o--------o Kvx[i+1, j+1]       
            i = round(Int,(ym[m] - yvx[1])/dy) + 1;
            j = round(Int,(xm[m] - xvx[1])/dx) + 1;

            q_x = abs(xm[m]-xvx[j])/dx;
            q_y = abs(ym[m]-yvx[i])/dy;
            
            Wtmij   = (1. - q_x) * (1. - q_y);
            Wtmi1j  = (1. - q_x) * q_y;
            Wtmij1  =  q_x * (1. - q_y);
            Wtmi1j1 =  q_x * q_y;

            WTSUMvx[i,j]        = WTSUMvx[i,j] + Wtmij;   
            WTSUMvx[i, j+1]     = WTSUMvx[i, j+1] + Wtmij1;
            WTSUMvx[i+1,j]      = WTSUMvx[i+1,j] + Wtmi1j;
            WTSUMvx[i+1,j+1]    = WTSUMvx[i+1,j+1] + Wtmi1j1;

            # Kvx NEW:
            KVXWTSUMvx[i,j]     = KVXWTSUMvx[i,j] + Km[m] * Wtmij;           # Upper left ↔ Wtmij     
            KVXWTSUMvx[i,j+1]   = KVXWTSUMvx[i,j+1] + Km[m] * Wtmij1;        # Upper right ↔ Wtmi1j
            KVXWTSUMvx[i+1,j]   = KVXWTSUMvx[i+1,j] +   Km[m] * Wtmi1j;      # Lower left ↔ Wtmij1
            KVXWTSUMvx[i+1,j+1] = KVXWTSUMvx[i+1,j+1] +  Km[m] * Wtmi1j1;    # Lower right ↔ Wtmi1j1
    


            # 3). Vy-node:  RHOvy, Kvy
            #     ρₘ => RHO       this is vy-node
            #                     /
            #   RHO[i, j]       o--------o RHO[i, j+1]
            #                  | * m    |
            #                 |        |
            #   RHO[i+1, j]  o--------o RHO[i+1, j+1]       
            i = round(Int,(ym[m] - yvy[1])/dy) + 1;
            j = round(Int,(xm[m] - xvy[1])/dx) + 1;

            q_x = abs(xm[m]-xvy[j])/dx;
            q_y = abs(ym[m]-yvy[i])/dy;
            
            Wtmij   = (1. - q_x) * (1. - q_y);
            Wtmi1j  = (1. - q_x) * q_y;
            Wtmij1  =  q_x * (1. - q_y);
            Wtmi1j1 =  q_x * q_y;

            WTSUMvy[i,j]        = WTSUMvy[i,j] + Wtmij;   
            WTSUMvy[i, j+1]     = WTSUMvy[i, j+1] + Wtmij1;
            WTSUMvy[i+1,j]      = WTSUMvy[i+1,j] + Wtmi1j;
            WTSUMvy[i+1,j+1]    = WTSUMvy[i+1,j+1] + Wtmi1j1;

            # RHOvy
            RHOWTSUMvy[i,j]     = RHOWTSUMvy[i,j] + RHOm[m] * Wtmij;           # Upper left ↔ Wtmij     
            RHOWTSUMvy[i,j+1]   = RHOWTSUMvy[i,j+1] + RHOm[m] * Wtmij1;        # Upper right ↔ Wtmi1j
            RHOWTSUMvy[i+1,j]   = RHOWTSUMvy[i+1,j] +   RHOm[m] * Wtmi1j;      # Lower left ↔ Wtmij1
            RHOWTSUMvy[i+1,j+1] = RHOWTSUMvy[i+1,j+1] +  RHOm[m] * Wtmi1j1;    # Lower right ↔ Wtmi1j1

            # Kvy NEW:
            KVYWTSUMvy[i,j]     = KVYWTSUMvy[i,j] + Km[m] * Wtmij;           # Upper left ↔ Wtmij     
            KVYWTSUMvy[i,j+1]   = KVYWTSUMvy[i,j+1] + Km[m] * Wtmij1;        # Upper right ↔ Wtmi1j
            KVYWTSUMvy[i+1,j]   = KVYWTSUMvy[i+1,j] +   Km[m] * Wtmi1j;      # Lower left ↔ Wtmij1
            KVYWTSUMvy[i+1,j+1] = KVYWTSUMvy[i+1,j+1] +  Km[m] * Wtmi1j1;    # Lower right ↔ Wtmi1j1
    

            # 4). Pressure nodes:  ETAp, RHOcpp, T0p
            #   ηₘ => ETAp     this is pr-node
            #                     /
            #   ETAp[i, j]       o--------o ETAp[i, j+1]
            #                  | * m    |
            #                 |        |
            #   ETAp[i+1, j]  o--------o ETAp[i+1, j+1]       
            i = round(Int,(ym[m] - ypr[1])/dy) + 1;
            j = round(Int,(xm[m] - xpr[1])/dx) + 1;
    
            q_x = abs(xm[m]-xpr[j])/dx;
            q_y = abs(ym[m]-ypr[i])/dy;
            
            Wtmij   = (1. - q_x) * (1. - q_y);
            Wtmi1j  = (1. - q_x) * q_y;
            Wtmij1  =  q_x * (1. - q_y);
            Wtmi1j1 =  q_x * q_y;

            WTSUMp[i,j]        = WTSUMp[i,j] + Wtmij;   
            WTSUMp[i, j+1]     = WTSUMp[i, j+1] + Wtmij1;
            WTSUMp[i+1,j]      = WTSUMp[i+1,j] + Wtmi1j;
            WTSUMp[i+1,j+1]    = WTSUMp[i+1,j+1] + Wtmi1j1;

            # ETAp
            ETAWTSUMp[i,j]     = ETAWTSUMp[i,j] + ETAm[m] * Wtmij;
            ETAWTSUMp[i,j+1]   = ETAWTSUMp[i,j+1] + ETAm[m] * Wtmij1;
            ETAWTSUMp[i+1,j]   = ETAWTSUMp[i+1,j] + ETAm[m] * Wtmi1j;
            ETAWTSUMp[i+1,j+1] = ETAWTSUMp[i+1,j+1] + ETAm[m] * Wtmi1j1;

            # RHOcpp NEW:
            RHOCPWTSUMp[i,j]     = RHOCPWTSUMp[i,j] + RHOcpm[m] * Wtmij;
            RHOCPWTSUMp[i,j+1]   = RHOCPWTSUMp[i,j+1] + RHOcpm[m] * Wtmij1;
            RHOCPWTSUMp[i+1,j]   = RHOCPWTSUMp[i+1,j] + RHOcpm[m] * Wtmi1j;
            RHOCPWTSUMp[i+1,j+1] = RHOCPWTSUMp[i+1,j+1] + RHOcpm[m] * Wtmi1j1;

            # T0p NEW: Special for T!
            TWTSUMp[i,j]        = TWTSUMp[i,j] + Tm[m] * RHOcpm[m] * Wtmij;
            TWTSUMp[i,j+1]      = TWTSUMp[i,j+1] + Tm[m] * RHOcpm[m] * Wtmij1;
            TWTSUMp[i+1,j]      = TWTSUMp[i+1,j] + Tm[m] * RHOcpm[m] * Wtmi1j;
            TWTSUMp[i+1,j+1]    = TWTSUMp[i+1,j+1] + Tm[m] * RHOcpm[m] * Wtmi1j1;

        end


    end


    # COMPUTE ETA, RHO
    # Mechanical
    # 1). RHOvy ↔ y-stokes  => ρₘ   on vy nodes
    # 2). ETAb ↔ x,y-stokes => ηₘ   on basic nodes
    # 3). ETAp ↔ x,y-stokes => ηₘ   on pressure nodes

    # Thermo
    # 4). Kvx ↔ x-heat flux => Kₘ   on vx nodes
    # 5). Kvy ↔ y-heat flux => Kₘ   on vy nodes
    # 6). RHOcpp ↔ heat conservation => ρCₚₘ on pressure nodes
    # 7). T0p ↔ heat conservation => Tₘ on pressure nodes

    for j in 1:1:Nx+1
        for i in 1:1:Ny+1

            # b-nodes
            if (WTSUMb[i,j] > 0)
                ETAb[i,j]  = ETAWTSUMb[i,j] / WTSUMb[i,j];
            end

            # vx-nodes
            if WTSUMvx[i,j] > 0
                Kvx[i,j] = KVXWTSUMvx[i,j] / WTSUMvx[i,j];
            end

            # vy-nodes
            if WTSUMvy[i,j] > 0
                RHOvy[i,j] = RHOWTSUMvy[i,j] / WTSUMvy[i,j];
                Kvy[i,j]   = KVYWTSUMvy[i,j] / WTSUMvy[i,j];  # NEW: new!
            end

            # P-nodes
            if WTSUMp[i,j] > 0
                ETAp[i,j]   = ETAWTSUMp[i,j] / WTSUMp[i,j];
                RHOcpp[i,j] = RHOCPWTSUMp[i,j] / WTSUMp[i,j];
                T0p[i,j]    = TWTSUMp[i,j] / RHOCPWTSUMp[i,j];  # NEW: special for T!
            end

            
        end
    end    



    p1 = heatmap(xpr, ypr,  RHOcpp, aspect_ratio=1, xlims=(xpr[1],xpr[end]), ylims=(ypr[1],ypr[end]), c=:turbo, title="Volumetric isobaric heat capacity")
    p2 = heatmap(xpr, ypr,  T0p  , aspect_ratio=1, xlims=(xpr[1],xpr[end]), ylims=(ypr[1],ypr[end]), c=:turbo, title="Temperature")

    display(plot(p1, p2, layout=(2,1)))

    savefig("geometry.png")

end


main()