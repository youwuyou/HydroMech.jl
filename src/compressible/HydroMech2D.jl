@doc raw"""
    CompressibleHydroMechEquations2D
The compressible two-phase flow equations without the inertial term. 

i). Total momentum (solid matrix and fluid)
```math
\nabla \cdot \underline{\underline{\sigma}} + g \rho^{[t]} = 0
```


ii). Fluid momentum
```math
v^{[D]} = -\frac{k^{[\phi]}}{\eta^{[f]}}(\nabla p^{[f]}-\rho^{[f]}g)
```


iii). Compressible solid mass
```math
\nabla \cdot v^{[s]} = - \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}
```


iv). Compressible fluid mass
```math
\nabla \cdot v^{[D]} = \frac{p^{[t]}-p^{[f]}}{\eta^{[\phi]}(1-\phi)}
```

"""
@views function HydroMech2D_compressible()


end



# TODO: this code will be completed after the verification of the incompressible 2D code
#   using the 2D porosity wave benchmark