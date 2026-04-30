# ==========================================================
# Lane-Emden Equation Solver for Polytropic Stellar Models
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math
import os

# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np


# --- Physical constants ---
G       = 6.67430e-11          # Gravitational constant [m^3 kg^-1 s^-2]
M_SUN   = 1.98847e30           # Solar mass [kg]
M_P     = 1.67262192369e-27    # Proton mass [kg]
HBAR    = 1.054571817e-34      # Reduced Planck constant [J s]
C_LIGHT = 2.99792458e8         # Speed of light in vacuum [m s^-1]


# --- Lane-Emden solver ---
def lane_emden_rhs(xi, theta, phi, n):
    """
    Return the right-hand side of the Lane-Emden system.

    The Lane-Emden equation is rewritten as a first-order system:

        theta' = phi
        phi'   = -2 phi / xi - theta^n
    """

    # Fractional powers of negative theta are not real-valued.
    # This only matters when plotting the mathematical continuation
    # beyond the physical surface theta = 0.
    if theta < 0.0 and abs(n - round(n)) > 1.0e-12:
        theta_power = 0.0
    else:
        theta_power = theta

    if xi == 0.0:
        dphi = 0.0
    else:
        dphi = -2.0 * phi / xi - theta_power**n

    return {
        "dtheta": phi,
        "dphi": dphi,
    }


def rk4_step(xi, theta, phi, step_size, n):
    """
    Advance the Lane-Emden system by one fourth-order Runge-Kutta step.
    """

    h = step_size

    k1 = lane_emden_rhs(xi, theta, phi, n)

    k2 = lane_emden_rhs(
        xi + 0.5 * h,
        theta + 0.5 * h * k1["dtheta"],
        phi + 0.5 * h * k1["dphi"],
        n,
    )

    k3 = lane_emden_rhs(
        xi + 0.5 * h,
        theta + 0.5 * h * k2["dtheta"],
        phi + 0.5 * h * k2["dphi"],
        n,
    )

    k4 = lane_emden_rhs(
        xi + h,
        theta + h * k3["dtheta"],
        phi + h * k3["dphi"],
        n,
    )

    theta_next = theta + h / 6.0 * (
        k1["dtheta"]
        + 2.0 * k2["dtheta"]
        + 2.0 * k3["dtheta"]
        + k4["dtheta"]
    )

    phi_next = phi + h / 6.0 * (
        k1["dphi"]
        + 2.0 * k2["dphi"]
        + 2.0 * k3["dphi"]
        + k4["dphi"]
    )

    return {
        "xi": xi + h,
        "theta": theta_next,
        "phi": phi_next,
    }


def find_zero_linear(xi_old, theta_old, xi_new, theta_new):
    """
    Estimate the first zero of theta using linear interpolation.
    """

    return xi_old - theta_old * (xi_new - xi_old) / (theta_new - theta_old)


def solve_lane_emden(n, xi_max=10.0, step_size=1.0e-4, stop_at_zero=True):
    """
    Solve the Lane-Emden equation for a given polytropic index.

    The physical boundary conditions are

        theta(0) = 1
        theta'(0) = 0.

    Since the equation is singular at xi = 0, the integration starts at
    xi = step_size using the Taylor expansion near the origin:

        theta(xi)  = 1 - xi^2 / 6 + n xi^4 / 120 + ...
        theta'(xi) = -xi / 3 + n xi^3 / 30 + ...
    """

    xi = step_size
    theta = 1.0 - xi**2 / 6.0 + n * xi**4 / 120.0
    phi = -xi / 3.0 + n * xi**3 / 30.0

    xi_values = [xi]
    theta_values = [theta]
    phi_values = [phi]

    xi_zero = None

    while xi < xi_max:
        xi_old = xi
        theta_old = theta
        phi_old = phi

        step = rk4_step(xi, theta, phi, step_size, n)

        xi = step["xi"]
        theta = step["theta"]
        phi = step["phi"]

        # The first zero of theta defines the physical surface
        # of the finite-radius polytrope.
        if theta <= 0.0 and xi_zero is None:
            xi_zero = find_zero_linear(xi_old, theta_old, xi, theta)
            phi_zero = phi_old + (phi - phi_old) * (xi_zero - xi_old) / (xi - xi_old)

            if stop_at_zero:
                xi_values.append(xi_zero)
                theta_values.append(0.0)
                phi_values.append(phi_zero)
                break

        xi_values.append(xi)
        theta_values.append(theta)
        phi_values.append(phi)

    return {
        "n": n,
        "xi": np.array(xi_values),
        "theta": np.array(theta_values),
        "phi": np.array(phi_values),
        "xi_zero": xi_zero,
    }


# --- Analytical solutions and validation ---
def theta_exact(xi, n):
    """
    Return the analytical Lane-Emden solution when available.

    Closed-form solutions are available for n = 0, n = 1, and n = 5.
    """

    if n == 0.0:
        return 1.0 - xi**2 / 6.0

    if n == 1.0:
        return np.sin(xi) / xi

    if n == 5.0:
        return 1.0 / np.sqrt(1.0 + xi**2 / 3.0)

    raise ValueError(f"No analytical solution is implemented for n = {n}.")


def exact_first_zero(n):
    """
    Return the exact first zero of theta for finite-radius analytical cases.

    Only the n = 0 and n = 1 analytical solutions have a finite first zero.
    The n = 5 solution remains positive and approaches zero only at infinity.
    """

    if n == 0.0:
        return math.sqrt(6.0)

    if n == 1.0:
        return math.pi

    return None


def validation_error(solution):
    """
    Compute the maximum absolute error against the analytical solution.

    If the numerical solution stops at the first zero of theta, the final
    interpolated point is excluded from the comparison because it is not an
    RK4 integration point.
    """

    n = solution["n"]
    xi = solution["xi"]
    theta_numeric = solution["theta"]

    if solution["xi_zero"] is not None:
        xi = xi[:-1]
        theta_numeric = theta_numeric[:-1]

    theta_reference = theta_exact(xi, n)

    return float(np.max(np.abs(theta_numeric - theta_reference)))


# --- Dimensionless quantities ---
def dimensionless_mass(solution):
    """
    Compute the dimensionless mass:

    mu = -xi_1^2 theta'(xi_1)
    """

    if solution["xi_zero"] is None:
        return None

    xi_1 = solution["xi_zero"]
    phi_1 = solution["phi"][-1]

    return float(np.real(-xi_1**2 * phi_1))


# --- White dwarf / Chandrasekhar mass ---
def white_dwarf_K():
    """
    Polytropic constant for a high-density relativistic white dwarf,
    using the expression given in the project statement.
    """

    return (
        (3.0**(1.0 / 3.0) * math.pi**(2.0 / 3.0))
        / (2.0**(4.0 / 3.0) * 4.0)
        * HBAR
        * C_LIGHT
        / M_P**(4.0 / 3.0)
    )


def chandrasekhar_mass(solution, G=G):
    """
    Compute the Chandrasekhar mass from the n = 3 Lane-Emden solution.

    For n = 3, the mass is independent of central density.
    """

    n = solution["n"]

    if n != 3.0:
        raise ValueError("Chandrasekhar mass calculation requires n = 3.")

    mu = dimensionless_mass(solution)
    K = white_dwarf_K()

    M_ch = (
        4.0
        * math.pi
        * mu
        * (((n + 1.0) * K) / (4.0 * math.pi * G)) ** 1.5
    )

    return {
        "K": K,
        "M_ch": M_ch,
        "M_ch_solar": M_ch / M_SUN,
    }


def print_chandrasekhar_summary(solution_n3):
    """
    Print the high-density white dwarf mass.
    """

    result = chandrasekhar_mass(solution_n3)

    print("\nHigh-density white dwarf model")
    print("--------------------------------")
    print(f"K = {result['K']:.6e} SI")
    print(f"M_Ch = {result['M_ch']:.6e} kg")
    print(f"M_Ch = {result['M_ch_solar']:.6f} M_sun")


# --- Output utilities ---
def print_summary(solutions):
    """
    Print a summary table of the dimensionless Lane-Emden solutions.

    The table includes:
        - n: polytropic index
        - xi_1: first zero of theta (dimensionless radius)
        - mu: dimensionless mass
        - max_error: max absolute error vs analytical solution (if available)
        - zero_error: error on xi_1 vs exact value (if available)
    """

    header = (
        "n        xi_1             mu              "
        "max_error        zero_error"
    )
    separator = "-" * len(header)

    print("Dimensionless summary")
    print(header)
    print(separator)

    for solution in solutions:
        n = solution["n"]
        xi_1 = solution["xi_zero"]
        mu = dimensionless_mass(solution)

        # --- xi_1 and mu ---
        if xi_1 is None or mu is None:
            xi_text = "None"
            mu_text = "None"
            zero_error_text = "None"
        else:
            xi_text = f"{xi_1:.10f}"
            mu_text = f"{mu:.10f}"

            xi_exact = exact_first_zero(n)
            if xi_exact is not None:
                zero_error = abs(xi_1 - xi_exact)
                zero_error_text = f"{zero_error:.3e}"
            else:
                zero_error_text = "None"

        # --- validation error ---
        try:
            max_error = validation_error(solution)
            max_error_text = f"{max_error:.3e}"
        except ValueError:
            max_error_text = "None"

        print(
            f"{n:<8.1f} "
            f"{xi_text:<16} "
            f"{mu_text:<15} "
            f"{max_error_text:<15} "
            f"{zero_error_text}"
        )


# --- Plot ---
os.makedirs("Images", exist_ok=True)

def plot_solution(solutions):

    plt.figure()

    for solution in solutions:
        xi = solution["xi"]
        theta = solution["theta"]
        n = solution["n"]

        plt.plot(xi, theta, label=f"n = {n}")

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta$")
    plt.xlim(0, 10)
    plt.ylim(-0.5, 1.0)

    plt.axhline(0.0, color="black", linewidth=0.8)

    n_labels = ", ".join(str(solution["n"]) for solution in solutions)
    plt.title(f"Lane-Emden solutions (n = {n_labels})")

    plt.grid(True)
    plt.legend()

    filename = "Images/lane_emden.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved as: {filename}")

def plot_dimensionless_density_profiles(solutions):
    """
    Plot rho/rho_c = theta^n for the required polytropes.
    """

    plt.figure()

    for solution in solutions:
        n = solution["n"]

        if n not in [0.0, 1.0, 3./2., 3.0]:
            continue

        if solution["xi_zero"] is None:
            continue

        xi = solution["xi"]
        theta = np.maximum(solution["theta"], 0.0)
        xi_1 = solution["xi_zero"]

        if n == 0.0:
            density = np.ones_like(theta)
            density[-1] = 0.0
        else:
            density = theta**n

        plt.plot(xi / xi_1, density, label=f"n = {n}")

    plt.xlabel(r"$\frac{r}{R} = \frac{\xi}{\xi_1}$")
    plt.ylabel(r"$\frac{\rho}{\rho_c}$")
    plt.title("Dimensionless density profiles")
    plt.grid(True)
    plt.legend()

    filename = "Images/dimensionless_density_profiles.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Density plot saved as: {filename}")


# --- Main program ---
def main():
    """
    Run the Lane-Emden solver for selected polytropic indices.
    """

    n_values = [0.0, 1.0, 3./2., 3.0, 5.0]
    xi_max = 10.0
    h = 1.e-4

    solutions = []
    plot_solutions = []

    print("=== Lane-Emden solver ===\n")

    for n in n_values:
        solution = solve_lane_emden(n, xi_max=xi_max, step_size=h, stop_at_zero=True)
        solutions.append(solution)

        plot_solution_extended = solve_lane_emden(
            n, xi_max=xi_max, step_size=h, stop_at_zero=False
        )
        plot_solutions.append(plot_solution_extended)

    print_summary(solutions)

    plot_solution(plot_solutions)

    plot_dimensionless_density_profiles(solutions)

    solution_n3 = next(solution for solution in solutions if solution["n"] == 3.0)
    print_chandrasekhar_summary(solution_n3)


if __name__ == "__main__":
    main()
