# [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl)

The [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) package exploits the two powerful emerging paradigms in HPC, *Massively parallel relaxation-cased solvers* and *HPC with Julia*.

The package is aimed to provide a reusable, extensible and high-performance framework, so that they may be applied within existing application codes and used to develop new ones. 

JustRelax contains solvers which are based on the accelerated pseudo-transient (PT) iterative method. 






# Features

- [x] 2D viscous stokes
- [x] 2D visco-elsatic stokes
- [ ] 2D non-Newtonian rheology
- [ ] 2D visco-elasto-plasticity
- [ ] 2D 2-phase flow
- [x] 3D Visco-elastic stokes
- [ ] Add the Zaremba-Jaumann rate of the Cauchy stress
- [ ] Refactor thermal diffusion (2D and 3D)
- [x] Paraview interface for 3D viz with [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl)
- [ ] Advection: Particles-in-cell
- [ ] Scalability tests
- [ ] Support for [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) (**ongoing**)
- [ ] I/O






# Workflow


The [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) package contains the solvers for different geophysical problems modelled using PDEs.

The package is self-contained and is equipped with the [boundary conditions](https://github.com/PTsolvers/JustRelax.jl/blob/main/src/boundaryconditions/BoundaryConditions.jl) and [global computational kernels]()