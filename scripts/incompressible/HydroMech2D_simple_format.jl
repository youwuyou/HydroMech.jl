# using the same structure as in the miniapp example
# does not use the MetaHydroMech.jl! Remove the use of it in the source if this is more desired.

const USE_GPU   = false  # Use GPU? If this is set false, then no GPU needs to be available
const DO_VIZ    = false
const SAVE_TEST = false

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using Statistics, Printf, LinearAlgebra


# Packages used by the application for output
## visualization
@static if DO_VIZ
    using Plots
end

# testing
@static if SAVE_TEST
    using JLD
end




# Compute kernel of the incompressible two-phase flow solver
@inbounds @parallel function update_old!(Phi_o::Data.Array, ∇V_o::Data.Array, Phi::Data.Array, ∇V::Data.Array)
    @all(Phi_o) = @all(Phi)
    @all(∇V_o)  = @all(∇V)
    return
end



@inbounds @parallel function compute_params_∇!(EtaC::Data.Array, K_muf::Data.Array, Rog::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Phi::Data.Array, Pf::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, μs::Data.Number, η2μs::Data.Number, R::Data.Number, λPe::Data.Number, k_μf0::Data.Number, _ϕ0::Data.Number, nperm::Data.Number, θ_e::Data.Number, θ_k::Data.Number, ρfg::Data.Number, ρsg::Data.Number, ρgBG::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @all(EtaC)  = (1.0-θ_e)*@all(EtaC)  + θ_e*( μs/@all(Phi)*η2μs*(1.0+0.5*(1.0/R-1.0)*(1.0+tanh((@all(Pf)-@all(Pt))/λPe))) )
    @all(K_muf) = (1.0-θ_k)*@all(K_muf) + θ_k*( k_μf0 * (@all(Phi)* _ϕ0)^nperm )
    @all(Rog)   = ρfg*@all(Phi) + ρsg*(1.0-@all(Phi)) - ρgBG
    @all(∇V)    = @d_xa(Vx)* _dx  + @d_ya(Vy)* _dy
    @all(∇qD)   = @d_xa(qDx)* _dx + @d_ya(qDy)* _dy

    return
end


@inbounds @parallel function compute_RP!(dτPf::Data.Array, RPt::Data.Array, RPf::Data.Array, K_muf::Data.Array, ∇V::Data.Array, ∇qD::Data.Array, Pt::Data.Array, Pf::Data.Array, EtaC::Data.Array, Phi::Data.Array, Pfsc::Data.Number, Pfdmp::Data.Number, min_dxy2::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @inn(dτPf) = min_dxy2/@maxloc(K_muf)/4.1/Pfsc
    @all(RPt)  =                 - @all(∇V)  - (@all(Pt) - @all(Pf))/(@all(EtaC)*(1.0-@all(Phi)))
    @all(RPf)  = @all(RPf)*Pfdmp - @all(∇qD) + (@all(Pt) - @all(Pf))/(@all(EtaC)*(1.0-@all(Phi)))

    return
end



@inbounds @parallel function compute_P_τ!(Pt::Data.Array, Pf::Data.Array, τxx::Data.Array, τyy::Data.Array, σxy::Data.Array, RPt::Data.Array, RPf::Data.Array, dτPf::Data.Array, Vx::Data.Array, Vy::Data.Array, ∇V::Data.Array, dτPt::Data.Number, μs::Data.Number, β_n::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @all(Pt)  = @all(Pt) +      dτPt *@all(RPt)
    @all(Pf)  = @all(Pf) + @all(dτPf)*@all(RPf)
    @all(τxx) = 2.0*μs*( @d_xa(Vx)* _dx - 1.0/3.0*@all(∇V) - β_n*@all(RPt) )
    @all(τyy) = 2.0*μs*( @d_ya(Vy)* _dy - 1.0/3.0*@all(∇V) - β_n*@all(RPt) )
    @all(σxy) = 2.0*μs*(0.5*( @d_yi(Vx)* _dy + @d_xi(Vy)* _dx ))
    return
end

@inbounds @parallel function compute_res!(Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, τxx::Data.Array, τyy::Data.Array, σxy::Data.Array, Pt::Data.Array, Rog::Data.Array, dampX::Data.Number, dampY::Data.Number, _dx::Data.Number, _dy::Data.Number)
    # @all(Rx)    = @d_xi(τxx)* _dx + @d_ya(σxy)* _dy - @d_xi(Pt)* _dx
    # @all(Rx)    = (@d_xi(τxx)- @d_xi(Pt))* _dx + @d_ya(σxy)* _dy 
    # @all(Ry)    = @d_yi(τyy)* _dy + @d_xa(σxy)* _dx - @d_yi(Pt)* _dy - @av_yi(Rog)
    @all(Ry)    = (@d_yi(τyy)- @d_yi(Pt))* _dy + @d_xa(σxy)* _dx - @av_yi(Rog)

    # @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVxdτ) = dampX*@all(dVxdτ) + (@d_xi(τxx)- @d_xi(Pt))* _dx + @d_ya(σxy)* _dy
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end


@inbounds @parallel function compute_update!(Vx::Data.Array, Vy::Data.Array, qDx::Data.Array, qDy::Data.Array, Phi::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, K_muf::Data.Array, Pf::Data.Array, Phi_o::Data.Array, ∇V::Data.Array, ∇V_o::Data.Array, dτV::Data.Number, ρfg::Data.Number, ρgBG::Data.Number, CN::Data.Number, dt::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @inn(Vx)  =  @inn(Vx) + dτV*@all(dVxdτ)
    @inn(Vy)  =  @inn(Vy) + dτV*@all(dVydτ)
    @inn(qDx) = -@av_xi(K_muf)*(@d_xi(Pf)* _dx)                     # fluid momentum x
    @inn(qDy) = -@av_yi(K_muf)*(@d_yi(Pf)* _dy + (ρfg - ρgBG))      # fluid momentum y
    @all(Phi) =  @all(Phi_o) + (1.0-@all(Phi))*(CN*@all(∇V_o) + (1.0-CN)*@all(∇V))*dt

    return
end



# 2D kernel
@inline @inbounds @parallel_indices (iy) function free_slip_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@inline @inbounds @parallel_indices (ix) function free_slip_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end


@inline @inbounds function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx, Vy, size_Vx_x, size_Vy_y) where {T}
    freeslip_x, freeslip_y = freeslip

    # free slip boundary conditions
    freeslip_x && (@parallel (1:size_Vy_y) free_slip_x!(Vy))
    freeslip_y && (@parallel (1:size_Vx_x) free_slip_y!(Vx))

    return nothing
end


@inline @inbounds function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, C, size_C_x, size_C_y) where {T}
    freeslip_x, freeslip_y = freeslip

    # free slip boundary conditions
    freeslip_x && (@parallel (1:size_C_y) free_slip_x!(C))
    freeslip_y && (@parallel (1:size_C_x) free_slip_y!(C))

    return nothing
end






##################################################
@doc raw"""
    IncompressibleHydroMechEquations2D
The incompressible two-phase flow equations without the inertial term. 

i). Total momentum (solid matrix and fluid)
```math
\nabla \cdot \underline{\underline{\sigma}} + g \rho^{[t]} = 0
```


ii). Fluid momentum
```math
v^{[D]} = -\frac{k^{[\phi]}}{\mu^{[f]}}(\nabla p^{[f]}-\rho^{[f]}g)
```


iii). Incompressible solid mass
```math
\nabla \cdot v^{[s]} = - \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}
```


iv). Incompressible fluid mass
```math
\nabla \cdot v^{[D]} = \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}
```

"""
@views function HydroMech2D_incompressible(;t_tot_)
    # Physics - scales
    ρfg      = 1.0             # fluid rho*g
    k_μf0    = 1.0             # reference permeability
    ηC0      = 1.0             # reference bulk viscosity
  
    # Physics - non-dimensional parameters
    η2μs     = 10.0            # bulk/shear viscosity ration
    R        = 500.0           # Compaction/decompaction strength ratio for bulk rheology
    # R        = 1000.0           # trial2
    # R        = 0.5
    nperm    = 3.0             # Carman-Kozeny exponent
    ϕ0       = 0.01            # reference porosity
    ra       = 2               # radius of initil porosity perturbation
    λ0       = 1.0             # standard deviation of initial porosity perturbation
    t_tot    = t_tot_          # total time
  
    # Physics - dependent scales
    ρsg      = 2.0*ρfg         # solid rho*g
    lx       = 20.0            # domain size x
    ly       = ra*lx           # domain size y
    ϕA       = 2*ϕ0            # amplitude of initial porosity perturbation
    λPe      = 0.01            # effective pressure transition zone
    dt       = 1e-5            # physical time-step
  
    # Numerics
    CN       = 0.5             # Crank-Nicolson CN=0.5, Backward Euler CN=0.0
    res      = 128
    nx, ny   = res-1, ra*res-1 # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf
    ε        = 1e-5            # non-linear tolerance
    iterMax  = 5e3             # max nonlinear iterations
    nout     = 200             # error checking frequency
    β_n      = 1.0             # numerical compressibility
    Vdmp     = 5.0             # velocity damping for momentum equations
    Pfdmp    = 0.8             # fluid pressure damping for momentum equations
    Vsc      = 2.0             # reduction of PT steps for velocity
    Ptsc     = 2.0             # reduction of PT steps for total pressure
    Pfsc     = 4.0             # reduction of PT steps for fluid pressure
    θ_e      = 9e-1            # relaxation factor for non-linear viscosity
    θ_k      = 1e-1            # relaxation factor for non-linear permeability
    dt_red   = 1e-3            # reduction of physical timestep
  
    # Derived physics
    μs       = ηC0*ϕ0/η2μs                       # solid shear viscosity
    λ        = λ0*sqrt(k_μf0*ηC0)                # initial perturbation width
    ρgBG     = ρfg*ϕ0 + ρsg*(1.0-ϕ0)             # Background density
  
    # Derived numerics
    dx, dy   = lx/(nx-1), ly/(ny-1)              # grid step in x, y
    min_dxy2 = min(dx,dy)^2
    dτV      = min_dxy2/μs/(1.0+β_n)/4.1/Vsc     # PT time step for velocity
    dτPt     = 4.1*μs*(1.0+β_n)/max(nx,ny)/Ptsc
    dampX    = 1.0-Vdmp/nx
    dampY    = 1.0-Vdmp/ny

    
    # Array allocations
    Phi_o    = @zeros(nx  ,ny  )
    Pt       = @zeros(nx  ,ny  )
    Pf       = @zeros(nx  ,ny  )
    Rog      = @zeros(nx  ,ny  )
    ∇V       = @zeros(nx  ,ny  )
    ∇V_o     = @zeros(nx  ,ny  )
    ∇qD      = @zeros(nx  ,ny  )
    dτPf     = @zeros(nx  ,ny  )
    RPt      = @zeros(nx  ,ny  )
    RPf      = @zeros(nx  ,ny  )
    τxx      = @zeros(nx  ,ny  )
    τyy      = @zeros(nx  ,ny  )
    σxy      = @zeros(nx-1,ny-1)
    dVxdτ    = @zeros(nx-1,ny-2)
    dVydτ    = @zeros(nx-2,ny-1)
    Rx       = @zeros(nx-1,ny-2)
    Ry       = @zeros(nx-2,ny-1)
    Vx       = @zeros(nx+1,ny  )
    Vy       = @zeros(nx  ,ny+1)
    qDx      = @zeros(nx+1,ny  )


    # Initial conditions
    qDy      =   zeros(nx  ,ny+1)
    Phi      = ϕ0*ones(nx  ,ny  )
    Radc     =   zeros(nx  ,ny  )
    Radc    .= [(((ix-1)*dx-0.5*lx)/λ/4.0)^2 + (((iy-1)*dy-0.25*ly)/λ)^2 for ix=1:size(Radc,1), iy=1:size(Radc,2)]
    Phi[Radc.<1.0] .= Phi[Radc.<1.0] .+ ϕA
    EtaC     = μs./Phi.*η2μs
    K_muf    = k_μf0.*(Phi./ϕ0)
    ϕ0bc     = mean.(Phi[:,end])
    qDy[:,[1,end]] .= (ρsg.-ρfg).*(1.0.-ϕ0bc).*k_μf0.*(ϕ0bc./ϕ0).^nperm
    
    Phi      = Data.Array(Phi)
    EtaC     = Data.Array(EtaC)
    K_muf    = Data.Array(K_muf)
    qDy      = Data.Array(qDy)
    t        = 0.0
    it       = 1

    # boundary condition
    freeslip = (freeslip_x=true, freeslip_y=true)

    # HPC precomputation
    _dx, _dy    = 1.0/dx, 1.0/dy
    _ϕ0         = 1.0/ϕ0
    length_Ry   = length(Ry)
    length_RPf  = length(RPf)
    data_size   = sizeof(Data.Number)

 
    # Preparation of visualisation
    if DO_VIZ
        ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)
    end
    
  
    # Time loop
    while t<t_tot
        @parallel update_old!(Phi_o, ∇V_o, Phi, ∇V)
        err=2*ε; iter=1; niter=0
       
        while err > ε && iter <= iterMax
            if (iter==11)  global wtime0 = Base.time()  end


            @parallel compute_params_∇!(EtaC, K_muf, Rog, ∇V, ∇qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, μs, η2μs, R, λPe, k_μf0, _ϕ0, nperm, θ_e, θ_k, ρfg, ρsg, ρgBG, _dx, _dy)
            @parallel compute_RP!(dτPf, RPt, RPf, K_muf, ∇V, ∇qD, Pt, Pf, EtaC, Phi, Pfsc, Pfdmp, min_dxy2, _dx, _dy)

            apply_free_slip!(freeslip, dτPf, nx, ny)
 
            @parallel compute_P_τ!(Pt, Pf, τxx, τyy, σxy, RPt, RPf, dτPf, Vx, Vy, ∇V, dτPt, μs, β_n, _dx, _dy)
            @parallel compute_res!(Ry, dVxdτ, dVydτ, τxx, τyy, σxy, Pt, Rog, dampX, dampY, _dx, _dy)

            @parallel compute_update!(Vx, Vy, qDx, qDy, Phi, dVxdτ, dVydτ, K_muf, Pf, Phi_o, ∇V, ∇V_o, dτV, ρfg, ρgBG, CN, dt, _dx, _dy)
          
            apply_free_slip!(freeslip, Vx, Vy, nx+1, ny+1)
            apply_free_slip!(freeslip, qDx, qDy, nx+1, ny+1)


            if mod(iter,nout)==0
                global norm_Ry, norm_RPf
                # norm_Ry = norm(Ry)/length(Ry); norm_RPf = norm(RPf)/length(RPf); err = max(norm_Ry, norm_RPf)
                norm_Ry = norm(Ry)/length_Ry; norm_RPf = norm(RPf)/length_RPf; err = max(norm_Ry, norm_RPf)
                # @printf("iter = %d, err = %1.3e [norm_Ry=%1.3e, norm_RPf=%1.3e] \n", iter, err, norm_Ry, norm_RPf)
            end
            iter+=1; niter+=1
        end
   
        # Performance
        wtime    = Base.time() - wtime0
        A_eff    = (8*2)/1e9*nx*ny*data_size  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(niter-10)                     # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
        @printf("it = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", it, wtime, round(T_eff, sigdigits=2))


        # Visualisation
        if DO_VIZ
            default(size=(500,700))
            if mod(it,5)==0
                p1 = heatmap(X, Y,  Array(Phi)'  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="porosity")
                p2 = heatmap(X, Y,  Array(Pt-Pf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="effective pressure")
                p3 = heatmap(X, Yv, Array(qDy)'  , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical Darcy flux")
                p4 = heatmap(X, Yv, Array(Vy)'   , aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="vertical velocity")
                display(plot(p1, p2, p3, p4)); frame(anim)
            end
        end

        # Time
        dt = dt_red/(1e-10+maximum(abs.(∇V)))
        t  = t + dt
        it+=1
    end
    
    # Visualization
    if DO_VIZ
        gif(anim, "HydroMech2D_incompressible.gif", fps = 15)
    end

    # Testing
    ## store data in case further testings needed
    if SAVE_TEST
        save("../../test/2D/incom_Peff.jld", "data", Array(Pt-Pf)')  # store case for reference testing
    end

    # return effective pressure at final time
    return Array(Pt-Pf)'
end


# if isinteractive()
#     HydroMech2D_incompressible(;t_tot_=0.02) # for reproducing porosity wave benchmark
#     # HydroMech2D_incompressible(;t_tot_=0.1) # for reproducing porosity wave benchmark
#     # HydroMech2D_incompressible(;t_tot_=0.0005) # for reproducing the test result
# end