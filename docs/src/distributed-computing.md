# Distributed Computing





## Tutorial



### Simple Performance Estimation

We can estimate the performance using the following metrics 

$$T_\text{eff} = \frac{A_\text{eff}}{t_\text{it}} = \frac{2 D_u + D_k}{\Delta t / \text{niter}}$$

TODO: add the example of the effective memory






### Parallelizing a serial code


- STEP 1: Precompute scalars, remove divisions

```julia
# instead of division, we precompute the fractions to be multipled on
_β_dτ_D     = 1. /β_dτ_D
```

- STEP 2: Remove element-wise operators and use loops instead for updating the elements of the arrays, where we introduce the indices like `ix`, `iy`

```julia
# the pressure update using the element-wise arithmetic operations
Pf     .-= ((qDx[2:end, :] - qDx[1:end-1, :]).* _dx .+ (qDy[:, 2:end] - qDy[:, 1:end-1]).* _dy).* _β_dτ_D
```


- STEP 3: Remove the julia functions like `diff(A, dims=1)` and use the indices `ix`, `iy` instead to "manually" compute the differences. Another possibility is to use the macros of the `ParallelStencil` package by `@d_xa`, `@d_ya` etc

```julia
# we manually implemented the macros
macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end

# and use them for the loop version of differences calculation
Pf[ix,iy]     -= (@d_xa(qDx) * _dx + @d_ya(qDy)* _dy) * _β_dτ_D

```


- STEP 4: After verifying the correctness of the bounds to be iterated on, add the macro `@inbounds` at the needed places

- STEP 5: Move the loops into a compute kernel in the following forms

```julia
function compute_Pf!(Pf,...)
    nx, ny = size(Pf)
    ...
    return nothing
end
```


### Parallelizing using `ParallelStencil.jl`


For the macros that can be used, check the [FiniteDifferences.jl](https://github.com/omlins/ParallelStencil.jl/blob/83c607b2d4fdcc38dceb130b3458ff736ebe9a18/src/FiniteDifferences.jl)

```julia

using Printf, LazyArrays, Plots, BenchmarkTools
using JLD  # for storing testing data


@views av1(A) = 0.5.*(A[1:end-1].+A[2:end])
@views avx(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])
@views avy(A) = 0.5.*(A[:,1:end-1].+A[:,2:end])


macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end



# Darcy's flux update
function compute_flux_darcy!(Pf, T, qDx, qDy, _dx, _dy, k_ηf, αρgx, αρgy, _1_θ_dτ_D)
    nx, ny = size(Pf)

    for iy = 1:ny
        for ix = 1:nx-1
            # qDx[2:end-1,:] .-= (qDx[2:end-1,:] .+ k_ηf.*((Pf[2:end,:] .- Pf[1:end-1, :]) .* _dx .- αρgx.*avx(T))).* _1_θ_dτ_D
            qDx[ix+1,iy] -= (qDx[ix+1,iy] + k_ηf * (@d_xa(Pf) * _dx - αρgx *  0.5 * (T[ix,iy] + T[ix+1,iy]))) * _1_θ_dτ_D
            
        end
    end
    
    for iy = 1:ny-1
        for ix = 1:nx
            # qDy[:,2:end-1] .-= (qDy[:,2:end-1] .+ k_ηf.*((Pf[:, 2:end] .- Pf[:, 1:end-1]) .* _dy .- αρgy.*avy(T))).* _1_θ_dτ_D
            qDy[ix,iy+1] -= (qDy[ix,iy+1] + k_ηf * (@d_ya(Pf) * _dy - αρgy * 0.5 * (T[ix, iy] + T[ix, iy+1]))) * _1_θ_dτ_D
        end
    end

end


# pressure update
function compute_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)
    nx, ny = size(Pf)

    for iy = 1:ny
        for ix = 1:nx
            # Pf     .-= ((qDx[2:end, :] - qDx[1:end-1, :]).* _dx .+ (qDy[:, 2:end] - qDy[:, 1:end-1]).* _dy).* _β_dτ_D
            @inbounds Pf[ix,iy]     -= (@d_xa(qDx) * _dx + @d_ya(qDy)* _dy) * _β_dτ_D
        end
    end

    return nothing
end


function compute_flux_temp!(Pf, T, qTx, qTy, _dx, _dy, λ_ρCp, _1_θ_dτ_T)
    nx, ny = size(Pf)

    for iy = 1:ny-2
        for ix = 1:nx-1
            # qTx            .-= (qTx .+ λ_ρCp.*(Diff(T[:,2:end-1],dims=1)./dx))./(1.0 + θ_dτ_T)
            qTx[ix,iy]  -= (qTx[ix,iy] + λ_ρCp*(@d_xa(T[:,2:end-1])* _dx)) * _1_θ_dτ_T                    
        end
    end
    
    for iy = 1:ny-1
        for ix = 1:nx-2
            # qTy            .-= (qTy .+ λ_ρCp.*(Diff(T[2:end-1,:],dims=2)./dy))./(1.0 + θ_dτ_T)
            qTy[ix,iy]  -= (qTy[ix,iy] + λ_ρCp*(@d_ya(T[2:end-1,:])* _dy)) * _1_θ_dτ_T
        end
    end

end



function compute_T!(T, dTdt, qTx, qTy, _dx, _dy, _dt_β_dτ_T)
    nx, ny = size(T)

    for iy = 1:ny-2
        for ix = 1:nx-2
            # T[2:end-1,2:end-1] .-= (dTdt .+ @d_xa(qTx).* _dx .+ @d_ya(qTy).* _dy).* _dt_β_dτ_T
            T[ix+1,iy+1] -= (dTdt[ix,iy] + @d_xa(qTx)* _dx + @d_ya(qTy)* _dy)* _dt_β_dτ_T                    
        end
    end
end



@views function porous_convection_2D_xpu(ny_, nt_; do_visu=false, do_check=true, test=true)
    # physics
    lx,ly       = 40., 20.
    k_ηf        = 1.0
    αρgx,αρgy   = 0.0,1.0
    αρg         = sqrt(αρgx^2+αρgy^2)
    ΔT          = 200.0
    ϕ           = 0.1
    Ra          = 1000                    # changed from 100
    λ_ρCp       = 1/Ra*(αρg*k_ηf*ΔT*ly/ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
  
    # numerics
    ny          = ny_                     # ceil(Int,nx*ly/lx)
    nx          = 2 * (ny+1) - 1          # 127
    nt          = nt_                     # 500
    re_D        = 4π
    cfl         = 1.0/sqrt(2.1)
    maxiter     = 10max(nx,ny)
    ϵtol        = 1e-6
    nvis        = 20
    ncheck      = ceil(max(nx,ny)) # ceil(0.25max(nx,ny))
  
    # preprocessing
    dx,dy       = lx/nx,ly/ny
    xn,yn       = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly,0,ny+1)
    xc,yc       = av1(xn),av1(yn)
    θ_dτ_D      = max(lx,ly)/re_D/cfl/min(dx,dy)
    β_dτ_D      = (re_D*k_ηf)/(cfl*min(dx,dy)*max(lx,ly))
   
    # hpc value precomputation
    _dx, _dy    = 1. /dx, 1. /dy
    _ϕ          = 1. / ϕ
    _1_θ_dτ_D   = 1 ./(1.0 + θ_dτ_D)
    _β_dτ_D     = 1. /β_dτ_D


    # array initialization
    Pf          = zeros(nx,ny)
    r_Pf        = zeros(nx,ny)
    qDx,qDy     = zeros(nx+1,ny),zeros(nx,ny+1)
    qDx_c,qDy_c = zeros(nx,ny),zeros(nx,ny)
    qDmag       = zeros(nx,ny)     
    T           = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2); T[:,1] .= ΔT/2; T[:,end] .= -ΔT/2
    T_old       = copy(T)
    dTdt        = zeros(nx-2,ny-2)
    r_T         = zeros(nx-2,ny-2)
    qTx         = zeros(nx-1,ny-2)
    qTy         = zeros(nx-2,ny-1)
   
    st          = ceil(Int,nx/25)
    Xc, Yc      = [x for x=xc, y=yc], [y for x=xc,y=yc]
    Xp, Yp      = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]

    # visu
    if do_visu
        # needed parameters for plotting

        # plotting environment
        ENV["GKSwstype"]="nul"
        if isdir("viz_out")==false mkdir("viz_out") end
        loadpath = "viz_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
    end



    # action
    t_tic = 0.0; niter = 0
    for it = 1:nt
        T_old .= T

        # time step
        dt = if it == 1 
            0.1*min(dx,dy)/(αρg*ΔT*k_ηf)
        else
            min(5.0*min(dx,dy)/(αρg*ΔT*k_ηf),ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1)
        end

        _dt = 1. /dt   # precomputation
        
        
        re_T    = π + sqrt(π^2 + ly^2/λ_ρCp * _dt)
        θ_dτ_T  = max(lx,ly)/re_T/cfl/min(dx,dy)
        β_dτ_T  = (re_T*λ_ρCp)/(cfl*min(dx,dy)*max(lx,ly))
        
        _1_θ_dτ_T   = 1 ./ (1.0 + θ_dτ_T)
        _dt_β_dτ_T  = 1 ./(_dt + β_dτ_T) # precomputation

        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
        while max(err_D,err_T) >= ϵtol && iter <= maxiter
            if (it==1 && iter == 11) t_tic = Base.time(); niter=0 end

            # hydro            
            compute_flux_darcy!(Pf, T, qDx, qDy, _dx, _dy, k_ηf, αρgx, αρgy, _1_θ_dτ_D)
            compute_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)
            
            # thermo
            compute_flux_temp!(Pf, T, qTx, qTy, _dx, _dy, λ_ρCp, _1_θ_dτ_T)
            #     dTdt        = zeros(nx-2,ny-2)
            

            dTdt           .= (T[2:end-1,2:end-1] .- T_old[2:end-1,2:end-1]).* _dt .+
                                (max.(qDx[2:end-2,2:end-1],0.0).*Diff(T[1:end-1,2:end-1],dims=1).* _dx .+
                                 min.(qDx[3:end-1,2:end-1],0.0).*Diff(T[2:end  ,2:end-1],dims=1).* _dx .+
                                 max.(qDy[2:end-1,2:end-2],0.0).*Diff(T[2:end-1,1:end-1],dims=2).* _dy .+
                                 min.(qDy[2:end-1,3:end-1],0.0).*Diff(T[2:end-1,2:end  ],dims=2).* _dy).* _ϕ

            
            # for iy = 1:ny-2
            #     for ix = 1:nx-2
            #         dTdt[ix,iy]           = (T[ix+1,iy+1] - T_old[ix+1,iy+1]) * _dt +
            #         (max(qDx[2:end-2,2:end-1],0.0) * @d_xa(T[1:end-1,2:end-1]) * _dx  +
            #          min(qDx[3:end-1,2:end-1],0.0) * @d_xa(T[2:end  ,2:end-1]) * _dx  +
            #          max(qDy[2:end-1,2:end-2],0.0) * @d_ya(T[2:end-1,1:end-1]) * _dy  +
            #          min(qDy[2:end-1,3:end-1],0.0) * @d_ya(T[2:end-1,2:end  ]) * _dy) * _ϕ

            #     end
            # end
            
            compute_T!(T, dTdt, qTx, qTy, _dx, _dy, _dt_β_dτ_T)


            # TODO: add the boundary condition kernel afterwards
            # Boundary condition
            T[[1,end],:]        .= T[[2,end-1],:]


            if do_check && iter % ncheck == 0
                r_Pf  .= Diff(qDx,dims=1).* _dx .+ Diff(qDy,dims=2).* _dy
                r_T   .= dTdt .+ Diff(qTx,dims=1).* _dx .+ Diff(qTy,dims=2).* _dy
                err_D  = maximum(abs.(r_Pf))
                err_T  = maximum(abs.(r_T))
                # @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",iter/nx,err_D,err_T)
            end
            iter += 1; niter += 1
        end
        # @printf("it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",it,iter/nx,err_D,err_T)

        if it % nvis == 0
            qDx_c .= avx(qDx)
            qDy_c .= avy(qDy)
            qDmag .= sqrt.(qDx_c.^2 .+ qDy_c.^2)
            qDx_c ./= qDmag
            qDy_c ./= qDmag
            qDx_p = qDx_c[1:st:end,1:st:end]
            qDy_p = qDy_c[1:st:end,1:st:end]
            

            # visualisation
            if do_visu
                heatmap(xc,yc,T';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),aspect_ratio=1,c=:turbo)
                png(quiver!(Xp[:], Yp[:], quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black),
                    @sprintf("viz_out/porous2D_%04d.png",iframe+=1))
            end
        end

    end


    t_toc = Base.time() - t_tic
    # FIXME: change the expression to compute the effective memory throughput!
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                              # Execution time per iteration [s]
    T_eff = A_eff / t_it                               # Effective memory throughput [GB/s]
    
    @printf("Time = %1.3f sec, T_eff = %1.3f GB/s \n", t_toc, T_eff)


    if test == true
        save("../test/qDx_p_ref_30_2D.jld", "data", qDx_c[1:st:end,1:st:end])  # store case for reference testing
        save("../test/qDy_p_ref_30_2D.jld", "data", qDy_c[1:st:end,1:st:end])
    end
    
    # Return qDx_p and qDy_p at final time
    return [qDx_c[1:st:end,1:st:end], qDy_c[1:st:end,1:st:end]]   
end



if isinteractive()
    porous_convection_2D_xpu(63, 500; do_visu=false, do_check=true,test=false)  # ny = 63
end



```