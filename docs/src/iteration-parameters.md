# Iteration Parameters

The choice of the iteration parameters are essential for the accelerated PT method as the method is highly sensitive to it.


## Choice of iteration parameters

Generally, the values of the iteration parameters are associated with the maximum eigenvalue of the stiffness matrix.

The optimal iteration parameters for a variety of **basic physical processes** can be determined analytically but for most of the practical problems the eigenvalue problem needs to be solved.

For the eigenvalue problem see more about its numerical aspects [here](eigenvalue-problem.md).

### Factor 1:  Choice of the B.C.



### Factor 2:  Numerical stability restriction

- for explicit time integration, the size of the timesteps is upper-bounded




# Case studies

## Physical processes

### 1). Diffusion


$$\rho \frac{\partial H}{\partial t} = -\nabla_k q_k$$

$$q_i = -D \nabla_k H, i = 1... n_d$$

Or by plugging the second equation into the first one we obtained a single equation for describing the diffusive process.


$$\rho \frac{\partial H}{\partial t} = \nabla_k (D \nabla_k H)$$

- H ↔ some physical quantity

- D ↔ diffusion coefficient 

- ρ ↔ proportionality coefficient

- t ↔ physical time


The stationary diffuion process is given by the above equation when $\frac{\partial H}{\partial t}\rightarrow 0$

$$0 = \nabla_k (D \nabla_k H)$$


#### Applying PT method

For the accelerated PT method we do the following:

- STEP 1: add the inertia term $\theta_r \frac{\partial q_i}{\partial \tau}$ to the LHS of the first equation

- STEP 2: plug the obtained equation from step 1 into the equation 2 to obtain the damped wave equation.

    - PDE type switch from parabolic to hyperbolic

    - describes also wave propagation

$$\tilde{\rho} \theta_r \frac{\partial^2 H}{\partial t^2} + \tilde{\rho} \frac{\partial H}{\partial \tau} = \nabla_k (D \nabla_k H)$$


- STEP 3: find the optimal [Reynolds number](https://www.wikiwand.com/en/Reynolds_number)
    
    - $Re = \frac{\tilde{\rho}V_p L}{D}$, where $V_p = \sqrt{\frac{D}{\tilde{\rho} \theta_r}}$ is the finite speed of the information signal of the wave propagation.

    - This can be done by the [dispersion analysis](dispersion-analysis.md), the optimal value of $Re$ in this case is $Re_\text{opt} = 2 \pi$




- STEP 4: obtain the optimal parameters of $\tilde{\rho}, \theta_r$ using the optimal Reynolds number 

    - Generally: "Low Re ⇒ flows tend to be laminar" and "High Re ⇒ flows tend to be turbulent"

$$\tilde{\rho} = Re \frac{D}{\tilde{V}L}$$

$$\theta_r = \frac{D}{\tilde{\rho} \tilde{V}^2} = \frac{L}{Re \tilde{V}}$$

> Here we need to solve for $V_p = \tilde{V}$, where $\tilde{V} := \frac{\tilde{C}\Delta x}{\Delta \tau}$ is the numerical velocity we just introduced. Note that $\tilde{C} \approx 0.95 C$ is used here, which is an emperically determined parameter deduced from numerical experiments.

> In case $D := D(x_k)$ is not constant, we need to determine its values locally to each grid point. For particularly discontinuous distribution of $D$, taking a local maximum of $D$ between two neighbouring grid cells for determining the iteration parameters shall be sufficient. [Räss et al](https://gmd.copernicus.org/articles/15/5757/2022/) 

- STEP 5: perform explicit time stepping

> Restriction for the size of the pseudo timestep damped wave equation: $$\Delta \tau \leq \frac{C}{V_p} \Delta x$$, where $$\Delta x = \frac{L}{n_x}$$, value of the non-dimensional number $$C$$ is determined for the linearised problem (von Neumann stability analysis)


#### Results

The number of iterations required for the method to converge is linearly dependent on the numrical grid resolution $n_x$








