# Lane-Emden Solver

This project numerically solves the Lane-Emden equation for selected polytropic indices using a fourth-order Runge-Kutta (RK4) method.

It computes dimensionless stellar structure quantities, validates numerical results against analytical solutions, produces diagnostic plots, and estimates the Chandrasekhar mass for a relativistic white dwarf modeled as an $n = 3$ polytrope.

---

## Features

- Numerical integration of the Lane-Emden equation (RK4)
- Support for multiple polytropic indices: $n = 0, 1, \tfrac{3}{2}, 3, 5$
- Detection of the first zero $\xi_1$ (stellar surface)
- Validation against analytical solutions (where available)
- Computation of:
  - Dimensionless radius $\xi_1$
  - Dimensionless mass $\mu = -\xi_1^2 \theta'(\xi_1)$
- Chandrasekhar mass estimate for $n = 3$
- Automatic generation of plots

---

## Output

The script produces:

- **Dimensionless summary table**, including:
  - Polytropic index $n$
  - First zero $\xi_1$
  - Dimensionless mass $\mu$
  - Numerical errors (when analytical solutions exist)

- **Plots:**
  - Lane-Emden solutions → `Lane_Emden_Solutions.pdf`
  - Dimensionless density profiles → `Dimensionless_density_profiles.pdf`

- **Chandrasekhar mass estimate**:
  - Expressed in SI units and solar masses

---

## Physics Background

The Lane-Emden equation describes self-gravitating polytropic spheres:

$$
\frac{1}{\xi^2} \frac{d}{d\xi} \left( \xi^2 \frac{d\theta}{d\xi} \right) = -\theta^n
$$

with boundary conditions:

$$
\theta(0) = 1, \qquad \theta'(0) = 0.
$$

The polytropic equation of state is:

$$
P = K \rho^{1 + \frac{1}{n}}.
$$

---

## Chandrasekhar Mass and Choice of $K$

For $n = 3$, the polytropic model describes a **relativistic degenerate electron gas**, appropriate for high-density white dwarfs.

In this regime, the polytropic constant $K$ is not arbitrary, but is derived from the physics of a relativistic Fermi gas. In this project, we use:

$$
K = \frac{3^{1/3} \pi^{2/3}}{2^{4/3} \cdot 4} \frac{\hbar c}{m_p^{4/3}}
$$

where:

- $\hbar$ is the reduced Planck constant  
- $c$ is the speed of light  
- $m_p$ is the proton mass  

This choice corresponds to the **ultra-relativistic limit** of electron degeneracy pressure and ensures that the model reproduces the correct scaling for white dwarf structure.

With this $K$, the stellar mass becomes:

$$
M_{\mathrm{Ch}} =
4\pi \mu
\left( \frac{(n+1)K}{4\pi G} \right)^{3/2}
$$

For $n = 3$, this mass is independent of central density, yielding the **Chandrasekhar mass limit**:

$$
M_{\mathrm{Ch}} \approx 1.44\, M_\odot.
$$
