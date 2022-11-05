# Stiffness of PDEs



## Approaches


### Operator splitting

- separate the system of PDEs into two parts

    - i). Non-stiff part: first system which contains only hyperbolic operators

    - ii). Stiff part: second system which contains parabolic operators ('stiff'), solved independently from the first system at each time step.

- different orders for various approaches

    - a). 1st order (Godunov-type)

    - b). 2nd order (Strang splitting)

    - c). other higher order splitting...

- convergence requirements
    
    - smoothness of the solution of the system of PDEs