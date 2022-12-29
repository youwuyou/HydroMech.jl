# Overview

HydroMech.jl in still in its development and is expected to release its first version by the end of Dec. 2022.


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

- currently only the reference test for incompressible TPF solver is availble using the porosity wave benchmark
