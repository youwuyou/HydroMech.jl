# HydroMech.jl

[![CI action](https://github.com/youwuyou/HydroMech.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/youwuyou/HydroMech.jl/actions/workflows/ci.yml)


## Description

HydroMech.jl was developed during the attendance of the course 101-0250-00 at ETH Zürich, which is deisgned to be a module of the PTSolver/JustRelax.jl package. It utilizes the pseudo-transient method for efficiency of the PDEs solving. It is designed to be able to run on both CPUs and GPUs.


### Part 1: multi-XPUs diffusion solver

- [documentation](/docs/part1.md)

- solving a given PDE as stage one of the course project


### Project part 2: Personal project

- Hydro-mechanical solvers (2D, 3D) as an integrated part of the [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) package

- [website documentation](http://justrelax-framework.org/dev/)

NOTE: we use the [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) package to generate the static website which contains the detailed documentation of the project containing the run instructions, code structure, further optimizations and the benchmarking results using the learnt techniques of the Julia distributed computing in the [course 101-0250-00 Solving PDEs in parallel on GPUs](https://github.com/eth-vaw-glaciology/course-101-0250-00) at ETH Zürich.




## Structure

```bash
HydroMech.jl
├── CNAME
├── LICENSE
├── docs             # documentation and final report
├── Project.toml
├── README.md
├── scripts-part1    # scripts for course project part 1
├── scripts-part2    # scripts for course project part 2
├── src
└── test             # test/part*.jl ↔ testing scripts
```


## Testing

- unit and reference testing for both part 1 and part 2 projects are included within the `test` folder


