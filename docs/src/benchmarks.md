# Benchmarks



## Porosity wave benchmark

Using the decompaction weakening approach as in [Raess et al.](https://academic.oup.com/gji/article/218/3/1591/5497299?login=true). We set up the benchmark for the porosity wave regime as limit of two-phase flow. The goal of this benchmark is to verify the reproducibility of the methodology as described [here](methodology.md)


[  ] DI versus PT (2D)

[  ] numerical vs. exact analytical solutions of solitary waves (1D)

[  ] convergence study with varying spatial and temporal resolutions (2D)



### Results

*Resulting animation of the code provided in the [`HydroMech2D.jl`](https://github.com/omlins/ParallelStencil.jl/blob/main/miniapps/HydroMech2D.jl), which is the starting point of our implementation of the solver.*

`R=500`, `t=0.02`

![2D wave](./assets/images/HydroMech2D_R500.gif)


`R=1.0`, `t=0.2`

![2D wave](./assets/images/HydroMech2D_R1.gif)




## Fluid injection 2D benchmark


Reproduce fluid injection 2D benchmark in [Luca et al. 2022](https://www.sciencedirect.com/science/article/pii/S0040195122003109)