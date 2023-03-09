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


```julia

@views function example_problem()
   
   #==================== PROBLEM SETUP =======================#
   # this depends on the model problem one wants to solve, we recommend to distinguish between the physical and numerical properties

   # MESH
    nx, ny   = 255, 511
    mesh     = PTGrid((nx,ny), (lx,ly),(dx,dy))



    # RHEOLOGY
    # define concrete values for parameters such as μˢ needed in the wanted rheology
    # types avaliable to be used can be seen under `src/types/Rheology.jl`
    ɸ0 = 0.01  
    #... and define some more variables needed

    rheology = ViscousRheology(μˢ,µᶠ,C,R,λp,k0,ɸ0,nₖ,θ_e,θ_k)

    # TWO PHASE FLOW
    # define the forces
    ρfg      = 1.0                     # fluid rho*g
    ρsg      = 2.0*ρfg                 # solid rho*g
    ρgBG     = ρfg*ɸ0 + ρsg*(1.0-ɸ0)   #Background density
    

   # Initial conditions:
   # the arrays with initial values shall be firstly defined as normal julia arrays
    𝝫              = ɸ0*ones(nx  ,ny  )
    𝞰ɸ              = μˢ./𝝫./C
    𝐤ɸ_µᶠ           = k0.*(𝝫./ɸ0)
   
   # we need to then further change the values in the object "flow" of accordingly, where we need to wrap the CPU arrays to be capable to be used on both CPU and GPU using the PTArray wrapper
   # PTArray is a compromise for the both CPU and GPU array usage in the ParallelStencil.jl package, more see MetaHydroMech.jl for details\

    flow              = TwoPhaseFlow2D(mesh, (ρfg, ρsg, ρgBG))
    flow.𝝫            = PTArray(𝝫)
    flow.𝞰ɸ           = PTArray(𝞰ɸ)
    flow.𝐤ɸ_µᶠ        = PTArray(𝐤ɸ_µᶠ)


    # PHYSICS FOR COMPRESSIBILITY
    µ   = 25.0
    # ...

    compressibility = Compressibility(mesh, µ, Ks, βs, βf)



    # PT COEFFICIENT  
    βₚₜ      = 1.0             # numerical compressibility
    # ...
    
    pt = PTCoeff(OriginalDamping,mesh,μˢ,Vsc,βₚₜ,dampX,dampY,Pfdmp,Pfsc,Ptsc)



   # Boundary conditions:
   # here we define a named tuple, this will be passed to the solver in order to specify the directions along which
   # the boundary conditions shall apply
    freeslip = (freeslip_x=true, freeslip_y=true)


   #==================== PHYSICAL TIMELOOP =======================#
   # define parameters needed to perform your physical time loop
    t_tot               = 0.02    # total time
    t                   = 0.0     # current time
    it                  = 1       # no. iterations

    while t<t_tot

        # Pseudo-time loop solving
        solve!(flow, compressibility, rheology, mesh, freeslip, pt,Δt,it)


        # Optional
        # one can also save the partial results or define each frame of a gif-animation here

        # Time
        dt = dt_red/(1e-10+maximum(abs.(∇V)))
        t  = t + dt
        it+=1
    end
    

    # Optional
    # possible post-processing here such as the call of some plotting routines


    # return desired values from the flow variable after the solving
    return Array(flow.Pt - flow.Pf)'
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

The core of the HydroMech.jl are the [solvers](https://github.com/youwuyou/HydroMech.jl/tree/main/src/solvers). We saw how one can call different `solve!()` routines by passing different parameters, currently we can choose between the incompressible and compressible two-phase flow solvers using the following commands.

```julia
# an example call to the TPF incompressible solver
solve!(flow, rheology, mesh, freeslip, pt, Δt, it)
```


```julia
# an example call to the TPF incompressible solver
solve!(flow, compressibility, rheology, mesh, freeslip, pt, Δt, it)
```

In the above examples, the variables such as `flow::TwoPhaseFlow2D`, `rheology::ViscousRheology` etc. are of types that are defined under the `src/types` scripts. The PT solvers essentially call different update routines for the residuals, physical properties (under `src/equations`) and as well as for the boundary updates (under `src/boundaryconditions`) in each pseudo-time loop. The selection of the routines to be called is based on the problem we aim to solve. 

Let's take a peek at the `solve!()` routine of the TPF incompressible solver, we focus on the pseudo-time loop within the solver routine. Now one can see the advantages why we added one more layer of the abstraction: due to the massive amount of parameters needed in the original equations. In the current `solve!()`, we need not to explicitly unpack the variables from the struct, we used the `Adapt.jl` package in order to allow the use of struct members on GPUs.

```julia
    while err > ε && iter <= iterMax
        # timing
        if (iter==11)  global wtime0 = Base.time()  end

        # involve the incompressible TPF solver
        @parallel compute_params_∇!(flow.𝞰ɸ, flow.𝐤ɸ_µᶠ, flow.𝞀g, flow.∇V, flow.∇qD, flow.𝝫, flow.Pf, flow.Pt, flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, rheology.μˢ, _C, rheology.R, rheology.λp, rheology.k0, _ɸ0, rheology.nₖ, rheology.θ_e, rheology.θ_k, flow.ρfg, flow.ρsg, flow.ρgBG, _dx, _dy)

        # pressure update from the conservation of mass flow
        @parallel compute_residual_mass_law!(pt.dτPt, pt.dτPf, flow.R.Pt, flow.R.Pf, flow.𝐤ɸ_µᶠ, flow.∇V, flow.∇qD, flow.Pt, flow.Pf, flow.𝞰ɸ, flow.𝝫, pt.Pfsc, pt.Pfdmp, min_dxy2, _dx, _dy)
        apply_free_slip!(freeslip, pt.dτPf, nx, ny)
        @parallel compute_pressure!(flow.Pt, flow.Pf, flow.R.Pt, flow.R.Pf, pt.dτPf, pt.dτPt)
        @parallel compute_tensor!(flow.𝞃.xx, flow.𝞃.yy, flow.𝞃.xy, flow.V.x, flow.V.y,  flow.∇V, flow.R.Pt, rheology.μˢ, pt.βₚₜ, _dx, _dy)

    
        # velocity update from the conservation of momentum flow
        # for both fluid and solid
        @parallel compute_residual_momentum_law!(flow.R.Vx, flow.R.Vy, pt.dVxdτ, pt.dVydτ, flow.𝞃.xx, flow.𝞃.yy, flow.𝞃.xy, flow.Pt, flow.𝞀g, pt.dampX, pt.dampY, _dx, _dy)
        @parallel compute_velocity!(flow.V.x, flow.V.y, flow.qD.x, flow.qD.y, pt.dVxdτ, pt.dVydτ, flow.𝐤ɸ_µᶠ, flow.Pf, pt.dτV, flow.ρfg, flow.ρgBG, _dx, _dy)
        apply_free_slip!(freeslip, flow.V.x, flow.V.y, nx+1, ny+1)
        apply_free_slip!(freeslip, flow.qD.x, flow.qD.y, nx+1, ny+1)
    
        # update the porosity
        @parallel compute_porosity!(flow.𝝫, flow.𝝫_o, flow.∇V, flow.∇V_o, CN, Δt)



        # ... error updates
    end
```


## Evolution Operators

One can see a `solve!()` routine defined in `src/solvers` as wrapper for many some update kernels which are defined within separate scripts of `src/evolution_operators`. The idea underlying this design is due to the fact that the core of different problems consisting of PDEs still centers around very few fundamental conservation laws. We thus organize each single-step update (kernel updates) into different scripts and named them as `MassConservation.jl`, `MomentumConservation.jl` and `EnergyConservation.jl` etc.

You may have noticed the naming of certain kernel update routines as `compute_residual_mass_law!()` and `compute_residual_momentum_law!()`, these methods reflect exactly the governing equation one aims to solve. For more information about the governing equations for each associated solver please refer to the API of HydroMech.

Besides the residual updates the computation kernels for various physical properties associated with each conservation law also are defined here.