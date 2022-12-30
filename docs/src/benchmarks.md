# Benchmarks

Followingly are some selected benchmark runs performed using the hydro-mechanical solver.

## Porosity wave benchmark

Using the decompaction weakening approach as in [Raess et al.](https://academic.oup.com/gji/article/218/3/1591/5497299?login=true). We set up the benchmark for the porosity wave regime as limit of two-phase flow. The goal of this benchmark is to verify the reproducibility of the methodology as described [here](methodology.md)



### Results



### Simple Run

#### Incompressible

*Resulting animation of the code provided in the [`HydroMech2D.jl`](https://github.com/omlins/ParallelStencil.jl/blob/main/miniapps/HydroMech2D.jl), which is the starting point of our implementation of the solver.*

`R=500`, `t=0.02`

![2D wave](./assets/images/incompressible_R500.gif)


`R=1.0`, `t=0.2`

![2D wave](./assets/images/incompressible_R1.gif)


#### Compressible

*Followingly are the (truncated) results using the same model setup but with additional physical variables added for the compressible terms of the mass conservation equation*

`R=1000`

![2D wave](./assets/images/compressible_R1000.gif)


`R=500`

![2D wave](./assets/images/compressible_R500.gif)


`R=1.0`, `t=0.03`

![2D wave](./assets/images/compressible_R1.gif)



```julia
# dimensionalized parameters used in (Dal Zilio et al. 2022)
µ = 25G*Pa            # shear modulus
Ks = 50G*Pa           # bulk modulus
βs = 2.5 10^-11 1/Pa  # solid compressibility
βf = 4.0 10^-10 1/Pa  # fluid compressibility
```



### Sensitivity Analysis

We keep all the parameters identical as in Räss et al. 2019 and perform 5 experiments with increasing grid resolution



``



``


- Experiment 3: `nx = 512`, `ny=1024`


```julia
# final output
it = 2055, time = 2.922e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2056, time = 2.919e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2057, time = 2.917e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2058, time = 2.898e+00 sec (@ T_eff = 120.00 GB/s) 
it = 2059, time = 2.914e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2060, time = 2.915e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2061, time = 2.915e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2062, time = 2.919e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2063, time = 2.919e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2064, time = 2.918e+00 sec (@ T_eff = 110.00 GB/s) 
it = 2065, time = 2.896e+00 sec (@ T_eff = 120.00 GB/s) 
```



## Fluid injection 2D benchmark


Reproduce fluid injection 2D benchmark in [Luca et al. 2022](https://www.sciencedirect.com/science/article/pii/S0040195122003109)