% Week08   - You Wu
% Content:   Solution of 2D heat conservation equation
%            ρCp· DT/Dt = k· (∂²/∂x² T + ∂²/∂y² T) 
%            
% Method:  - implicit with time stepping constraints
%          - properties on the pressure nodes


% 1). clear figures
clear all
clf


% 2). define numerical model
% Eulerian 2D grid
xsize = 100000;             % Horizontal size, [m] = 100[km]
ysize = 100000;             % Vertical size,   [m] = 100[km]
Nx = 35;
Ny = 45;

dx = xsize/(Nx-1);          % Horizontal grid step, [m]
dy = ysize/(Ny-1);          % Vertical grid step,   [m]
min_dx_dy = min(dx,dy);     % minimum is fixed


% basic nodes
% x = 0:dx:xsize;
% y = 0:dy:ysize;

% p nodes
xpr = -dx/2:dx:xsize+dx/2;  % Horizontal  [m]
ypr = -dy/2:dy:ysize+dy/2;  % Vertical    [m]


% array initializations
T0  = zeros(Ny+1, Nx+1);
Tdt = zeros(Ny+1, Nx+1);
RHOcp = zeros(Ny+1, Nx+1);


% physics
% constant thermal conductivity
k = 3.;                  % [W/m/K]
T_int = 1773;            % [K]
T_ext = 1573;            % [K]
rhocp_int = 1.1 * 3.2e6; % [J/(K·m³)]
rhocp_ext = 3.3e6;       % [J/(K·m³)]
R = 20000;               % [m] = 20 [km]

% properties defined on the pressure nodes
for j = 1:1:Nx+1
    for i = 1:1:Ny+1
        r = ((xpr(j)-xsize/2)^2 + (ypr(i)-ysize/2)^2)^0.5;

        if(r < R)
            T0(i,j) = T_int;
            RHOcp(i,j) = rhocp_int;
        else
            T0(i,j) = T_ext;
            RHOcp(i,j) = rhocp_ext;
        end

    end
end


%%%%%%%%%%%%%%%% TIME STEPPING START %%%%%%%%%%%%%%%%%%%%%%%%%%

% index of the matrices
n = (Nx+1) * (Ny+1);
% ERROR !!!: better to define matrixes outside the timestepping
L = sparse(n,n);
R =  zeros(n,1);

% timestepping
ntimesteps = 10;

min_val = min(min(RHOcp))

dt  = min_dx_dy^2 / (4 * k/min(min(RHOcp)));

for timestep=1:1:ntimesteps

    %%%%%%%%%%%%%%%%%% UNKNOWN SOLVING START %%%%%%%%%%%%%%%%%%%%%%%%
    % ERROR !!!: better to define matrixes outside the timestepping
%     L = sparse(n,n);
%     R =  zeros(n,1);
    
    % COMPOSITION OF THE MATRIX
    for j = 1:1:Nx+1
        for i =1:1:Ny+1
            % Define global index g
            g = (j-1) * (Ny+1) + i;

            if(i==1 || i==Ny+1 || j==1 || j==Nx+1)
                if (j == 1 && i > 1 && i < Ny+1)
                    % external nodes at the left boundary (Neumann)
                    L(g,g)        = 1;
                    L(g,g+(Ny+1)) = -1;
                    R(g,1)        = 0;
                end
                
                if(j==Nx+1 && i > 1 && i < Ny+1)
                    % external nodes at the right boundary (Neumann)
                    L(g,g)        = 1;
                    L(g,g-(Ny+1)) = -1;
                    R(g,1)        = 0;
                end

                if(i == 1)
                    % external nodes at the upper boundary (Dirichlet)
                    % ERROR !!!: wrong BC
%                     L(g,g)       = 1;
                    L(g,g)       = 1/2;
                    L(g,g+1)       = 1/2;
                    R(g,1)       = T_ext;
                end

                if(i == Ny+1)
                    % external nodes at the lower boundary (Dirichlet)
                    % ERROR !!!: wrong BC
%                     L(g,g)       = 1;
                    L(g,g)       = 1/2;
                    L(g,g-1)       = 1/2;
                    R(g,1)       = T_ext;
                end
        
            else
                % internal: heat conservation equation
                %  ρCp· DT/Dt = k· (∂²/∂x² T + ∂²/∂y² T)
                           
                %             (i-1,j)
                %             T2 
                %             |
                %             |
                %  (i,j-1)   (i,j)     (i,j+1) 
                %  T1--------T3--------T5
                %            |
                %            |
                %            (i+1,j)
                %           T4  

                % Timestepping formulation
                %  T3/dt - k/ρCp ( (T1 - 2T3 + T5)/dx²  +  (T2 - 2T3 + T4)/dy²) = T3⁰/dt
                % NOTE: use the RHOcp obtained at the T3 node!!
                L(g,g-(Ny+1)) = -k/RHOcp(i,j) / dx^2;               % T1
                L(g,g-1)      = -k/RHOcp(i,j) / dy^2;               % T2
                L(g,g)        = 1./dt + 2. * k /RHOcp(i,j) / dx^2 ...
                                      + 2. * k /RHOcp(i,j) / dy^2;    % T3
                L(g,g+1)      = -k/RHOcp(i,j) / dy^2;           % T4
                L(g,g+(Ny+1)) = -k/RHOcp(i,j) / dx^2;           % T5

                % RHS
                R(g,1) = T0(i,j) / dt;
            end
            
            
        end
    end

    % solve LSE
    S = L \ R;

    % reload the solution
    for j=1:1:Nx+1
        for i=1:1:Ny+1
            % global index for current node
            g = (j-1) * (Ny+1) + i;
            Tdt(i,j) = S(g);
        end
    end

    %%%%%%%%%%%%%%%%%% UNKNOWN SOLVING END %%%%%%%%%%%%%%%%%%%%%%%%
    % reset T0 for next step
    T0 = Tdt;

    % VISUALIZATION
    figure(1); clf; 
    colormap('Jet')

    % a). Volumetric isobaric heat capacity
    % RHOCP
    subplot(1,2,1)
    pcolor(xpr,ypr,RHOcp)
    axis ij image;
    colorbar
    title('RHOCP, J/m^3/K')


    % c). Temperature
    subplot(1,2,2)
    pcolor(xpr,ypr,Tdt)
    shading interp;
    axis ij image;
    colorbar
    title('TDT, K')

end
%%%%%%%%%%%%%%%% TIME STEPPING END %%%%%%%%%%%%%%%%%%%%%%%%%%


% REFERENCE
% for 10 time steps with dt=min(dx,dy)^2/(4*K/min(min(RHOCP)));
aaa(1,1)=Tdt(17,15); %[1704.92596509904], K=3, 2D 100x100km, 35x45  implicit

