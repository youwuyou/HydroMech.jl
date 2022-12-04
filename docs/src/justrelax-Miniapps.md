# Miniapps


## 7 Stages

> In the following section we introduce the structure of the main function of an "Application" code

- 1. Quantities needed to describe "where the problem lives", in terms of (parallel) topology

- 2. Initialize tools which can represent this domain concretely in parallel (IGG here, could be PETSc/DM)

- 3. Concrete representations of data and population of values
   - Includes information on embedding/coordinates

- 4. Tools, dependent on the data representation, to actually solve a particular physical problem (here JustRelax.jl, but could be PETSc's SNES)
    - Note that here, the physical timestepping scheme is baked into this "physical problem"

- 5. Analysis and output which depends on the details of the solver

- 6. "Application" Analysis and output which does not depend on the details of the solver

- 7. Finalization/Cleanup


In a real application, steps 4., 5., and 6. will likely be repeated multiple times and be interspersed with other logic (e.g. a particle advection step).


**Note: CompGrids.jl combines 1. 2., and part of 3. (coordinates, and the identity of the fields)**
