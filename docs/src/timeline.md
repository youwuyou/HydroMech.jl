# Timeline

This is a small timeline of the project development. 


## Course Project

The first part consists of a project for the [course 101-0250-00L](https://pde-on-gpu.vaw.ethz.ch/final_proj/).


| Datum | Content | Current goals / Todos |
| -------- | -------- | -------- |
| Oct. 25    | overview of the JustRelax.jl package, setting goals for the julia course | roadmap is provided via [hackmd](https://hackmd.io/@albert-de-montserrat/rkqpTQS4i)  |
| Nov. 28    | Meeting    | settle the repository structure, current goal focuses on the reproducing the 2D porosity wave benchmark, the implementation of the compressible term and the fluid injection benchmark. Using `const PTArray` in `MetaHydroMech.jl` to avoid type instabilities |
| Nov. 29    | Julia course project week 1    | separated the boundary condition update routines into the `src/boundaryconditions`, debugging for the CUDA-aware MPI on RACKlette cluster at CSCS (yet failed). |
| Dec. 6    | Julia course project week 2    | Final aim of the course is to migrate the code to 3D |
| Dec. 8    | Meeting     | CUDA-aware MPI works now by switching mpi implementation used from mpich to openmpi |
| Dec. 11    | Update     | Compressibility terms added for mass conservation, reliability needs to be verified |
| Dec. 13    | Julia course project week 3    | Compressibility for the mass conservation equation seems visually correct. Next step is to use implicit time stepping scheme and use the newest damping terms such that the augmented system is accelerated as in the damped wave equation case |
| Dec. 16    | Meeting     | Same content as in the course of this week, next step after the course would be to add the inertia term in the momentum equation, which recovers the NS equation  |
| Dec. 17    | Update     | Huge changes to the repository structure, now the fundamental conservation laws are included under `src/equations`, current aim is both the theory study and some setup for further benchmarks  |
| Dec. 18 - 28    | Liteature study for the PT method (MOL approach, general study for the iterative methods, timestepping schemes)   | working on a final report for the course project |
| Dec. 30    | project report deadline for the [course 101-0250-00L](https://pde-on-gpu.vaw.ethz.ch/final_proj/)   |  |


## Bachelor's Thesis

The second part consists of the bachelor's thesis, extending the using the code previously developed for earthquake cycle simulations.


| Datum | Content | Current goals / Todos |
| -------- | -------- | -------- |
| Feb. 22    | Meeting with Luca   | Official start of the bachelor's thesis, first step is to test the fully compressible mass conservation equations using the fluid injection benchmark (BP1) |
| Mar. 9    |  Meeting with Luca  | Clear-up of the benchmark setup, I was not aware of the boundary conditions described in [Dal Zilio et al. 2022](https://www.sciencedirect.com/science/article/pii/S0040195122003109) do NOT apply for the fluid injection benchmark. BP1 only consists of simple fluid diffusion so mechanical effects shall be negligible, by setting homogenous boundary conditions for solid part |
| Mar. 16   | Meeting with Luca  | First results of fluid injection benchmark, deviations from analytical are clearly visible and a lot of pt-timesteps required for convergence. We forward to implementation of the rheology (adding elasticity and plasticity) as in [Dal Zilio et al. 2022](https://www.sciencedirect.com/science/article/pii/S0040195122003109). |
| Mar. 18 - 20   | Literature Study  | Going through the sympy dispersion analysis scripts for PT damping parameters, in order to better understand the convergence behavior of the solver |
| Mar. 23 - 26 | Update | Reproduced the shear banding benchmark with visco-plastic predictor formulation as in [Duretz et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2021GC009675), modifications of the code towards formulation as in Dal Zilio, further understanding of results needed |
| Apr. 4 - 6  | Meetings | On Tuesday, Ludovic & Ivan kindly cleared up some of my confusions regarding the physical nature of fluid flow in porous medium, the expected behavior of adding elasticity and plasticity, theory of dilatancy and compaction with exampls of sand dilation, and some confusion I had for Duretz's formulation and the dispersion analysis script. On Wednesday, Albert provided useful suggestions for the code and we did some modifications to the shear banding experiment's setup. On Thursday, Luca clarified the desired plasiticity implementation shall be rate- and state plasticity instead of rate- stengthening as in the H-MEC paper, next step is to implement the plasticity correctly and move on to the earthquake cycles. |
| Apr. 9  | Update | Uploaded the highly accurate solution for the fluid injection benchmark (BP1). Changes for improvement involved to use the latest update scheme for solid part as in [Räss et al. 2022](https://gmd.copernicus.org/articles/15/5757/2022/), apply different pt time step reduction factors on fault zone and the domain |
| Jul. 20  | Update | Uploaded the first version of the seismo-mechanical earthquake cycle, PT damping parameters still need to be tuned for convergence |


