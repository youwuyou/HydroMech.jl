# Pseudo-Transient Method

The Pseudo-Transient Method (PT method), is an iterative method which is:

- matrix-free

- builds on a fixed-point iteration

    - update of each grid point is local, does not require global reductions

- enables easy-to-develop multi-physics coupling due to its conciseness

- similarity between mathematical and discretised code notation



## History

```@raw html
<details><summary>1911 - Pioneering work by Richardson </summary>
```
> Richardson proposed an iterative solution approach to PDEs related to dam-engineering calculations. Early developed iterative algorithms are well-suited for early low-memory computers but lack in efficient convergence rates.
```@raw html
</details>
```


```@raw html
<details><summary>1950 - First present of the PT method in the literature (Frankel)</summary>
```
> The idea of accelerating the convergence by increasing the order of PDE dates back to the work by [Frankel (1950)](https://www.jstor.org/stable/2002770?origin=crossref#metadata_info_tab_contents). Frankel noted the analogy between the iteration process and transient physics. And the accelarated method was called the *second-order Richardson method*.

> Introduced as an extension of the Richardson and Liebmann methods, with dependency on the previous iterations added.

```@raw html
</details>
```

```@raw html
<details><summary>1965 - The PT method originated as a dynamic-relaxation method (Oter) </summary>
```
> The PT method was applied for calculating the stresses and displacements in concrete pressure vessels.
```@raw html
</details>
```

```@raw html
<details><summary>1972 - Enhanced convergence rates of the PT methods showed (Young) </summary>
```
> The PT method was firstly performed on par.
```@raw html
</details>
```


```@raw html
<details><summary>1976 - First introduction in geosciences (Cundall) </summary>
```
> The PT method was introduced by Cundall as the *Fast Lagranngian Analysis of Continua (FLAC) algorithm*
```@raw html
</details>
```

```@raw html
<details><summary>1993, 1994 - Applications of the FLAC method (Poliakov et al.) </summary>
```
> The FLAC method was successfully applied to simulate the Rayleigh–Taylor instability in visco-elastic flow (Poliakov et al. 1993), and the formation of shear bands in rocks (Poliakov et al. 1994).
```@raw html
</details>
```


```@raw html
<details><summary>1993 - Application in buckling (Ramesh and Krishnamoorthy) </summary>
```
> 
```@raw html
</details>
```


```@raw html
<details><summary>1999 - Application in form-finding (Barnes) </summary>
```
> 
```@raw html
</details>
```

```@raw html
<details><summary>2009 - Application in failure (Kilic and Madenci) </summary>
```
> 
```@raw html
</details>
```

```@raw html
<details><summary>2011 - FEM community still referenced it as the DR-method (Rezauee-Pajand) </summary>
```
> 
```@raw html
</details>
```

```@raw html
<details><summary>2020 - Review on the accurate estimate of extremal eigenvalues for the Chebyshev's semi-iterative methods (Saad) </summary>
```
> NOTE: second-order or extrapolated methods are also termed semi-iterative.
```@raw html
</details>
```




```@raw html
<details><summary>2022 - Accelerated pseudo-transient method (Räss et al.) </summary>
```
> Assessing the robustness and scalability of the accelerated pseudo-transient method [Räss et al. (2022)](https://gmd.copernicus.org/articles/15/5757/2022/)

```@raw html
</details>
```

# Accelerated Pseudo-Transient Method

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6553699.svg)](https://doi.org/10.5281/zenodo.6553699)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6553714.svg)](https://doi.org/10.5281/zenodo.6553714)


Followingly we abstract some important aspects reported in the paper "Assessing the robustness and scalability of the accelerated pseudo-transient method [Räss et al. (2022)](https://gmd.copernicus.org/articles/15/5757/2022/), in which the accelerated PT method was introduced.

It has the following advantages:

- Ensures the iteration count to scale linearly with numerical resolution increase


## Application

The method is applicable to:

- Strongly nonlinear problems
    - shear-banding in a visco-elasto-plastic medium


- Finding solution of stationary problems

- Finding solution of problems with transient terms
    - involve both physical time $t$ and pseudo-time $\tau$
    - also called "dual-time method" or "dual time stepping" [(Mandal et al 2011)](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0207(19980330)41:6%3C1153::AID-NME334%3E3.0.CO;2-9) 




## Derivation

A physically motivated derivation is well-presented . To understand how powerful the PT method is, we cited here a small paragraph from the paper:


> "The PT methods build on a physical description of a process. It therefore becomes possible to model strongly nonlinear processes and achieve convergence starting from nearly arbitrary initial conditions."

The accelerated PT method for elliptic equations is mathematically equivalent to the second-order Richardson rule.


## Numerics

### Convergence

The convergence rate of the *accelerated PT method* is very sensitive to the iteration parameters' choice.


### Iteration parameters

The choice of the iteration parameters are essential for the accelerated PT method as the method is highly sensitive to it.

By analysing the equations of the basic physical processes in their continuous form, we can select the optimal iteration parameters. For more information regarding how to choose the optimal iterations parameters see [here](iteration-parameters.md).



### Boundary conditions (B.C.)

> "The choice of the type of boundary conditions affects only the values of the optimal iteration parameters and does not limit the generality of the method"


### Robustness



## Performance

#### Choice of physical processes

The choice of transient physical processes influences the performance of iterative methods.


#### CPU-based


#### GPU-based

- Weak scaling benchmarks with more than 96% parallel efficiency on 2197 Nvidia Tesla P100 GPUs on the *Piz Daint* supercomputer [[Räss et al. (2022)](https://gmd.copernicus.org/articles/15/5757/2022/)]






