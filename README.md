# HydroMech.jl

[![CI action](https://github.com/youwuyou/HydroMech.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/youwuyou/HydroMech.jl/actions/workflows/ci.yml) 

[![][docs-dev-img]][docs-dev-url]


## Description

HydroMech.jl was developed during the attendance of the course 101-0250-00 at ETH Zürich, which is designed to be a module of the PTSolver/JustRelax.jl package. It utilizes the pseudo-transient method for efficiency of the PDEs solving. It is designed to be able to run on both CPUs and GPUs.


## Course project hand-out

### HydroMech.jl

- Hydro-mechanical solvers (2D, 3D) as an integrated part of the [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) package

- [website documentation](http://justrelax-framework.org/dev/)

NOTE: we use the [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) package to generate the static website which contains the detailed documentation of the project containing the run instructions, code structure, further optimizations and the benchmarking results using the learnt techniques of the Julia distributed computing in the [course 101-0250-00 Solving PDEs in parallel on GPUs](https://github.com/eth-vaw-glaciology/course-101-0250-00) at ETH Zürich.




## Structure

```bash
HydroMech.jl
├── benchmark        # run scripts for reproducing benchmarks in the paper
├── CNAME
├── docs             # documentation and final report
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── scripts          # contains the most current version of the scripts in development
├── src              # should contain the developed code after verification of the correctness
└── test             # test/part*.jl ↔ testing scripts
```


## Testing

- unit and reference testing are included within the `test` folder


