# Von Neumann Stability Analysis

- also called as Fourier stability analysis

- a procesure used to check the stability of FD schemes when applied to linear PDEs

- based on the Fourier decomposition of numerical error

## Example 1: Upper-bound for the $\Delta \tau$ in explicit time stepping scheme

For the damped wave equation when we want to solve it using the PT method, we need to hold to restrictions for parameters if the explicit time stepping scheme is used.

The upper-bound is given as 

$$\Delta \tau \leq \frac{C}{V_p} \Delta x$$

Where the non-dimensional number $C \approx \frac{1}{\sqrt{n_d}}$ can be determined using Von Neumann stability analysis, where $n_d$ is the number of spatial dimensions. [(Alkhimenkov et al., 2021)](https://academic.oup.com/gji/article/225/1/354/6027602)


NOTE: for implicit time stepping scheme of the wave equations, the restriction on the time step vanishes, the time step will be determined by the scheme for Darcy's flux. In case also the implicit scheme is used for the Darcy's flux, the whole scheme will be unconditionally syable.
