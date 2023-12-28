# HydroMech.jl

Modeling earthquake source processes with an accelerated pseudo-transient solver.

## Description

The physics of the PT solver package HydroMech.jl is formulated based on the numerical framework [H-MEC](https://www.sciencedirect.com/science/article/pii/S0040195122003109), which is a continuum-based modeling approach that is developed to investigate physical systems that comprise both fluid (hydro) and solid (mechanical) phases, thereby referred to as "hydro-mechanical". The goal is to simulate how crustal stress and fluid pressure evolve during the earthquake cycle.


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


