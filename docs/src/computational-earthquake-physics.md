# Computational Earthquake Physics


## Geodynamics


> "The crucial point that was finally understood by the geological community is that both
> viscous (i.e., fluid-like) and elastic (i.e., solid-like) behaviour is a characteristic of the Earth
> depending on the time scale of deformation. The Earth’s mantle, which is elastic on a
> human time scale, is viscous on geological time scales (>10 000 years) and can be strongly
> internally deformed due to solid-state creep. -- (Gerya 2019)"

The dual viscous-elastic behaviour of the Earth can be demonstrated using the "silly putty", which jumps up like a rubber ball (acts like solid) when we drop it on the floor for a very short timescale, but demonstrates more fluid-like behaviors in a longer time period (few days/weeks).

See demonstration of the visco-elasticity [here](https://www.youtube.com/watch?v=UsE6x2NYec4)

## Rock rheology

> Rheology:  the composite physical property characterizing *deformation behavior of a material*.

> Rock rheology:  includes several different deformation mechanisms, and is in general visco-elasto-plastic.
>  Elastic properties are important to be taken into account on a relatively short time scale ($<10^4$ years) for
>  fast processes like magma intrusion. On the other hand, at low temperature rocks can be subjected to localized
>  brittle and plastic deformation.


Followingly is a small summary of different types of rheology. The major difference between them is the composition of the bulk deviatoric strain rate $\dot{\epsilon}_{ij}'$. For example, it is decomposed into 2 respective components for visco-plastic rheology as $\dot{\epsilon}_{ij}' = \dot{\epsilon}_{ij\text{(viscous)}}' + \dot{\epsilon}_{ij\text{(plastic)}}'$


- **visco-plastic:** a strain rate based formation is more suitable. Simplification by assuming that elastic effects are negligible and can be ignored on the long time scales.


- **visco-elastic:** a stress-based formulation is more suitable. Viscous and elastic rheological relations are combined under certain physical assumptions. Maxwell visco-elastic rheology is the most commenly used type. Definitions of shear and bulk moduli are modified in the equations. More see section 12.4 in (Gerya 2019)


- **visco-elasto-plastic:** a stress-based formulation is more suitable. It characterises the non-linear instantaneous response at higher stress levels or temperature


- **poro-elastic:** Biot's model and its validity at low stress level and negligible viscous relaxation


- **poro-elasto-plastic:** 


- **thermo-hydro-chemico-mechanical (THCM):** thermal and chemical couplings to deformation (mechanics)


NOTE: in general, coupling among processes triggers non-linear interactions that may result in significant and spontaneous localization of flow, heat and deformation.


### Plastic rheology

> Assumption: Existence of an absolute shear stress limit $\sigma_\text{yield}$ for a body and after reaching this limit *plastic yielding* occurs.


![Plastic yielding](https://static.cambridge.org/binary/version/id/urn:cambridge.org:id:binary:20190507095603410-0152:9781316534243:14314fig12_5.png?pub-status=live)


Formulations based on the simplified Griffith theory:


**Case 1: Dry rocks**

$$\sigma_\text{yield} = \begin{cases} \sigma_c + \gamma_\text{int}P, & P > \frac{\sigma_c - \sigma_t}{1-\gamma_\text{int}} \text{ (confined fractures)} \\ \sigma_t + P, & P < \frac{\sigma_c - \sigma_t}{1-\gamma_\text{int}} \text{ (tensile fractures)} \end{cases}$$

Or more compactly

$$\sigma_\text{yield} = \max [0, \min(\sigma_c + \gamma_\text{int} P^{[t]}, \sigma_t + \gamma_\text{int} P^{[t]})]$$

**Case 2: Fluid-bearing rocks**

$$\sigma_\text{yield} = \begin{cases} \sigma_c + \gamma_\text{int}(P^{[t]}-P^{[f]}), & (P^{[t]}-P^{[f]}) > \frac{\sigma_c - \sigma_t}{1-\gamma_\text{int}} \text{ (confined fractures)} \\ \sigma_t + (P^{[t]}-P^{[f]}), & (P^{[t]}-P^{[f]}) < \frac{\sigma_c - \sigma_t}{1-\gamma_\text{int}} \text{ (tensile fractures)} \end{cases}$$


where $\sigma_c$ and $\sigma_t$ are respectively *compressive length* and *tensile length* such that $2 \leq \frac{\sigma_c}{\sigma_t} \leq 8$


Or using effective pressure for the formulation

$$\sigma_\text{yield} = \max [0, \min(\sigma_c + \gamma_\text{int} P_\text{eff}, \sigma_t + \gamma_\text{int} P_\text{eff})]$$



> NOTE: Yield point - the point on a stress-strain curve that indicates the limit of elastic behavior and the beginning of plastic behavior [wiki](https://www.wikiwand.com/en/Yield_(engineering))


## Two-phase flow problems

> Focus: on the evolution of coupled fluid-solid systems, where the fluid is a low viscosity phase, which could be gas, liquid, melt and their mixtures.

> Special analytical and numerical treatment are needed due to intrinsic coupling between fluid percolation and deformation of the solid matrix (Gerya)

