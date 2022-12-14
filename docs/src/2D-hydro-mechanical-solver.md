# 2D Hydro-mechanical Solver

The solvers in our package aims to solve the two-phase flow equations.


## Two-phase flow (TPF) solver 

Comparing to the Stokes flow, equations for three more unknowns are to be solved which are related to the Darcy flux. Here we assume the constant porosity of the solid.



### Case 1: Incompressible

- no mass transfer between the solid and fluid and vice versa

- used in the current developed code of JustRelax.jl

*i). Total momentum (solid matrix and fluid)*

$$\nabla \cdot \underline{\underline{\sigma}} + g \rho^{[t]} = \nabla p^{[t]}$$


*ii). Fluid momentum*

$$v^{[D]} = -\frac{k^{[\phi]}}{\eta^{[f]}}(\nabla p^{[f]}-\rho^{[f]}g)$$


*iii). incompressible solid mass*

$$\nabla \cdot v^{[s]} = - \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}$$

*iv). incompressible fluid mass*

$$\nabla \cdot v^{[D]} = \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}$$



NOTE 1: Porosity-dependent permeability is given by $k^{[\phi]} = k^{[\phi]}_r (\frac{\phi}{\phi}_r)^m (\frac{1- \phi}{1-\phi}_r)^n$


NOTE 2: Porosity-dependent viscosity is given by $\eta^{[\phi]} = K_p \frac{\eta^{[t]}}{\phi}$



### Case 2: Compressible

- used in the H-MEC model

- to add the compressible terms for the solid & fluid mass, following changes are made in comparison to the incompressible solver

    - precomputation of parameters: drained bulk modulus `Kd`, Biot-Willis coefficient `ɑ`, Skempton coefficient `B`

    - the residual calculation of the incompressible solver `RPt`, `RPf` has additional terms



*i). Total momentum (solid matrix and fluid)*

$$\nabla \cdot \underline{\underline{\sigma}} + g \rho^{[t]} = \rho^{[t]}\frac{D^{[s]}v^{[s]}}{Dt}$$

*ii). Fluid momentum*

$$v^{[D]} = -\frac{k^{[\phi]}}{\eta^{[f]}}(\nabla p^{[f]}-\rho^{[f]}(g-\frac{D^{[f]}v^{[f]}}{D t}))$$


*iii). Fully compressible solid mass*

$$\nabla \cdot v^{[s]} = -\frac{1}{K^{[d]}}(\frac{D^{[s]} p^{[t]}}{D t} - \alpha \frac{D^{[f]} p^{[f]}}{Dt}) - \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}$$

*iv). Fully compressible fluid mass*

$$\nabla \cdot v^{[D]} = \frac{\alpha}{K^{[d]}}(\frac{D^{[s]} p^{[t]}}{D t} - \frac{1}{B} \frac{D^{[f]} p^{[f]}}{Dt}) + \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}$$



NOTE:  porosity-dependent permeability

$$k^{[\phi]} = k^* (\frac{\phi^*}{\phi})^n$$

- with reference values abstracted from the Table 1
    - reference permeability     ↔    $k^* = 10^{-16} m^2$
    - reference porosity         ↔    $\phi^* = 1 \%$

NOTE:  effective visco-plastic compaction viscosity

$$\eta^{[\phi]} = \frac{2m}{1+m} \frac{\eta_{s(vp)}}{\phi} =^{m=1} \frac{\eta_{s(vp)}}{\phi}$$

- geometrical factor m = 1 for cylindrical pores
- effective visco-plastic shear viscosity of the solid matrix ↔ $\eta_{s(vp)}$



### Others
- [McKenzie (1984)](https://doi.org/10.1093/petrology/25.3.713)