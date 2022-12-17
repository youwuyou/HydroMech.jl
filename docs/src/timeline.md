# Timeline

This is a small timeline of the project development


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
| Dec. 22    | project report deadline for the [course 101-0250-00L](https://pde-on-gpu.vaw.ethz.ch/final_proj/)   |  |