# Tutorial

This page serves as a tutorial for the use of the `HydroMech.jl` package. The structure overview of the package is briefly described under the "overview" section.


# Workflow

The general workflow when using the `HydroMech.jl` package consists of:

i). package environment setup

ii). problem setup for the specific PDEs to be solved

iii). selection of the PT solver. All PT solvers are involved by the same name `solve!()` but with different parameters. The correct one will get selected based on the parameters passed, this is decided by multiple dispatch, a Julia language feature. The `solve!()` shall be placed within the physical time loop and itself embeds a pseudo-time loop implicitly defined in its source module.

iv). Post-processing of the data, including the visualization or returning of certain data values which can be utilized for testing purposes


## Example: Basic Workflow

Followingly we are going to illustrate the usage of the solvers through a step-by-step example. Let us create a julia script called `Example.jl` and follow the instructions to understand the workflow of the `HydroMech.jl` package. Note that for illustration purpose the following example is NOT a working example, it solely serves as an example which illustrates the workflow.

- STEP 1: include the package and set up the environment needed for the `ParallelStencil.jl` package. This includes the selection of the device, number type to be used and the dimension of the problem.

We have to set up the environment accordingly in each script due to the usage of the `ParallelStencil.jl` package, which requires specification of the modules (`ParallelStencil.FiniteDifferences1D`, `ParallelStencil.FiniteDifferences2D` ...). One thing to be noticed is that when one wants to use the Github Action to automatize the workflow, the commited code shall not have `:gpu` specified for the environment setup. Since currently we cannot perform testing using Github GPU resources and this will result it Github Action failure.


```julia
# within the Example.jl script
using HydroMech

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# you can optionally use some other packages such as Plots.jl etc depending on your needs

```


- STEP 2: Define a function where we solve a desired problem by calling the solver

Unfortunately the call of the PT solver via `solve!()` is still quite stupid yet, as it requires a lot of parameters to be passed and is yet unstructured. Further improvements are needed by the use of `GeoParams.jl` package, which can significantly enhance the readability of the code. This is still under construction!

Similarly the precomputations of certain values are required in the preparation phase, we will possibly adjust this for sake of the convinience when using the solver routines.

```julia

@views function example_problem()
   
   #==================== PROBLEM SETUP =======================#
   # this depends on the model problem one wants to solve, we recommend to distinguish between the physical and numerical properties

   # Physics:  here we assign values to physical properties such as reference density ??0, reference porosity ??0
   # Numerics:  here we assign values to numerical properties such as nx,ny 

   # Array allocation:
   # the empty arrays shall be initialized using macros @zeros
    Pt       = @zeros(nx  ,ny  )
    Pf       = @zeros(nx  ,ny  )
    
   # Initial conditions:
   # the arrays with initial values shall be firstly defined as normal julia arrays, and then wrapped by the PTArray
   # PTArray is a compromise for the both CPU and GPU array usage in the ParallelStencil.jl package, more see MetaHydroMech.jl for details
   Phi_cpu   = ??0*ones(nx  ,ny  )
   Phi       = PTArray(Phi_cpu)



   # Boundary conditions:
   # here we define a named tuple, this will be passed to the solver in order to specify the directions along which
   # the boundary conditions shall apply
    freeslip = (freeslip_x=true, freeslip_y=true)


    # HPC precomputation:
    # these values are needed for the solver for efficiency since division is more computationally heavy than performing multiplication
    # and we don't want to measure the norm of the initial values redundantly which is used in the error comparison
    _dx, _dy    = 1.0/dx, 1.0/dy
    _??0         = 1.0/??0
    length_Ry   = length(Ry)
    length_RPf  = length(RPf)


  
   #==================== PHYSICAL TIMELOOP =======================#
   # define parameters needed to perform your physical time loop
    t_tot               = 0.02    # total time
    t                   = 0.0     # current time
    it                  = 1       # no. iterations

    while t<t_tot

        # Pseudo-time loop solving
        solve!(EtaC, K_muf, Rhog, ???V, ???qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, ??s, ??2??s, R, ??Pe, k_??f0, _??0, nperm, ??_e, ??_k, ??fg, ??sg, ??gBG, _dx, _dy,
                  d??Pf, RPt, RPf, Pfsc, Pfdmp, min_dxy2,
                  freeslip, nx, ny, ??xx, ??yy, ??xy,d??Pt, ??_n,
                  Rx, Ry, dVxd??, dVyd??, dampX, dampY,
                  Phi_o, ???V_o, d??V, CN, dt,
                  ??, iterMax, nout, length_Ry, length_RPf, it
              )


        # Optional
        # one can also save the partial results or define each frame of a gif-animation here

        # Time
        dt = dt_red/(1e-10+maximum(abs.(???V)))
        t  = t + dt
        it+=1
    end
    

    # Optional
    # possible post-processing here such as the call of some plotting routines


    # return desired values after the solving
    return Array(Pt-Pf)'
end

```


- STEP 3: Last step is to call the function we just defined and see if we get the results as expected!

```julia
if isinteractive()
    example_problem()
end
```


This concludes the main idea of the package usage. For a working example please refer to the [`PorosityWave2D.jl` benchmark](https://github.com/youwuyou/HydroMech.jl/blob/main/benchmark/incompressible/PorosityWave2D.jl) which followed the above structure.


## PT Solvers

The core of the HydroMech.jl are the [solvers](https://github.com/youwuyou/HydroMech.jl/tree/main/src/solvers). We saw how one can call different `solve!()` routines by passing different parameters (again, parameters will be bundled in further development, yet it is still quite ugly unfortunately...). 


```julia
# an example call to the TPF incompressible solver
solve!(EtaC, K_muf, Rhog, ???V, ???qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, ??s, ??2??s, R, ??Pe, k_??f0, _??0, nperm, ??_e, ??_k, ??fg, ??sg, ??gBG, _dx, _dy,
    d??Pf, RPt, RPf, Pfsc, Pfdmp, min_dxy2,
    freeslip, nx, ny, ??xx, ??yy, ??xy,d??Pt, ??_n,
    Rx, Ry, dVxd??, dVyd??, dampX, dampY,
    Phi_o, ???V_o, d??V, CN, dt,
    Kd, Kphi, _Ks, ??, ??, ??d, ??s, ??f, B, Pt_o, Pf_o,
    ??, iterMax, nout, length_Ry, length_RPf, it
    )
```

The PT solvers essentially call different update routines for the residuals, physical properties (under `src/equations`) and as well as for the boundary updates (under `src/boundaryconditions`) in each pseudo-time loop. The selection of the routines to be called is based on the problem we aim to solve. 

Let's take a peek at the `solve!()` routine of the TPF incompressible solver, we focus on the pseudo-time loop within the solver routine. Now one can see why some many parameters get passed to the solver: due to the massive amount of parameters needed in the original equations.

```julia
    while err > ?? && iter <= iterMax
        # timing
        if (iter==11)  global wtime0 = Base.time()  end

        # involve the incompressible TPF solver
        @parallel compute_params_???!(EtaC, K_muf, Rhog, ???V, ???qD, Phi, Pf, Pt, Vx, Vy, qDx, qDy, ??s, ??2??s, R, ??Pe, k_??f0, _??0, nperm, ??_e, ??_k, ??fg, ??sg, ??gBG, _dx, _dy)

        # pressure update from the conservation of mass equations
        @parallel compute_residual_mass_law!(d??Pt, d??Pf, RPt, RPf, K_muf, ???V, ???qD, Pt, Pf, EtaC, Phi, Pfsc, Pfdmp, min_dxy2, _dx, _dy)
        apply_free_slip!(freeslip, d??Pf, nx, ny)
        @parallel compute_pressure!(Pt, Pf, RPt, RPf, d??Pf, d??Pt)
        @parallel compute_tensor!(??xx, ??yy, ??xy, Vx, Vy, ???V, RPt, ??s, ??_n, _dx, _dy)
        
    
        # velocity update from the conservation of momentum equations
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(Rx, Ry, dVxd??, dVyd??, ??xx, ??yy, ??xy, Pt, Rhog, dampX, dampY, _dx, _dy)
        @parallel compute_velocity!(Vx, Vy, qDx, qDy, dVxd??, dVyd??, K_muf, Pf, d??V, ??fg, ??gBG, _dx, _dy)
        apply_free_slip!(freeslip, Vx, Vy, nx+1, ny+1)
        apply_free_slip!(freeslip, qDx, qDy, nx+1, ny+1)
    
        # update the porosity
        @parallel compute_porosity!(Phi, Phi_o, ???V, ???V_o, CN, dt)


        # ... error updates
    end
```


## Equations

One can see a `solve!()` routine defined in `src/solvers` as wrapper for many some update kernels which are defined within separate scripts of `src/equations`. The idea underlying this design is due to the fact that the core of different problems consisting of PDEs still centers around very few fundamental conservation laws. We thus organize equations (kernel updates) into different scripts and named them as `MassConservation.jl`, `MomentumConservation.jl` and `EnergyConservation.jl` etc.

You may have noticed the naming of certain kernel update routines as `compute_residual_mass_law!()` and `compute_residual_momentum_law!()`, these methods reflect exactly the governing equation one aims to solve. For more information about the governing equations for each associated solver please refer to the API of HydroMech.

Besides the residual updates the computation kernels for various physical properties associated with each conservation law also are defined here.