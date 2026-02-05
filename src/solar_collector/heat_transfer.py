"""
Heat transfer correlations for solar collector models.

This module provides empirical correlations for calculating convective heat
transfer coefficients in pipe flow, particularly the Dittus-Boelter correlation
for internal forced convection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solar_collector.fluid_properties import SYLTHERM800
from solar_collector.solar_collector_dae_pyo import PIPE_DIAMETER

# Default fluid properties for thermal oil
DEFAULT_FLUID_DENSITY = 800.0  # kg/m³
DEFAULT_FLUID_SPECIFIC_HEAT = 2000.0  # J/kg·K
DEFAULT_FLUID_THERMAL_CONDUCTIVITY = 0.12  # W/m·K (typical thermal oil)
DEFAULT_FLUID_DYNAMIC_VISCOSITY = 0.01  # Pa·s (typical thermal oil)
NUSSELT_NUMBER_LAMINAR = 4.36

# Flow regime thresholds (Reynolds number)
RE_LAMINAR_MAX = 2300  # Upper limit for laminar flow
RE_TURBULENT_MIN = 4000  # Lower limit for fully turbulent flow

# Default colormap for temperature-dependent plots
DEFAULT_COLORMAP = "plasma"


def calculate_heat_transfer_coefficient_turbulent(
    velocity,
    pipe_diameter,
    fluid_density,
    fluid_viscosity,
    fluid_thermal_conductivity,
    fluid_specific_heat,
):
    """
    Calculate internal heat transfer coefficient in a pipe using Dittus-Boelter
    correlation.

    The Dittus-Boelter correlation is widely used for turbulent flow in smooth
    pipes with moderate temperature differences. It provides good accuracy
    (±25%) for most engineering applications.

    Parameters:
    -----------
    velocity : float
        Fluid velocity [m/s]
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_density : float
        Fluid density [kg/m³]
    fluid_viscosity : float
        Dynamic viscosity [Pa·s]
    fluid_thermal_conductivity : float
        Thermal conductivity [W/m·K]
    fluid_specific_heat : float
        Specific heat capacity [J/kg·K]

    Returns:
    --------
    h : float
        Heat transfer coefficient [W/m²·K]
    Re : float
        Reynolds number [-]
    Pr : float
        Prandtl number [-]
    Nu : float
        Nusselt number [-]

    Notes:
    ------
    - Reynolds number: Re = ρ*v*D/μ (inertial forces / viscous forces)
    - Prandtl number: Pr = μ*cp/k (momentum diffusivity / thermal diffusivity)
    - Nusselt number: Nu = h*D/k (convective / conductive heat transfer)

    Correlations used:
    - Turbulent flow (Re > 4000): Nu = 0.023 * Re^0.8 * Pr^0.4

    For Laminar flow (Re ≤ 4000) use
    calculate_heat_transfer_coefficient_nusselt with Nu = 4.36.

    Valid range:
    - 0.7 ≤ Pr ≤ 160
    - Re > 10,000 (but works reasonably well down to Re ≈ 4000)
    - L/D > 10 (fully developed flow)
    """
    # Reynolds number
    Re = calculate_reynolds_number(
        velocity, pipe_diameter, fluid_density, fluid_viscosity
    )

    # Prandtl number
    Pr = calculate_prandtl_number(
        fluid_viscosity, fluid_specific_heat, fluid_thermal_conductivity
    )

    # Nusselt number (Dittus-Boelter correlation)
    Nu = 0.023 * Re**0.8 * Pr**0.4

    # Heat transfer coefficient
    h = calculate_heat_transfer_coefficient_nusselt(
        pipe_diameter, fluid_thermal_conductivity, Nu
    )

    return h, Re, Pr, Nu


def calculate_heat_transfer_coefficient_nusselt(
    pipe_diameter, fluid_thermal_conductivity, Nu=4.36
):
    """
    Calculate internal heat transfer coefficient for laminar flow in a
    pipe.

    Parameters:
    -----------
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_thermal_conductivity : float
        Thermal conductivity [W/m·K]
    Nu : float, default=4.36
        Nusselt number [-]

    Returns:
    --------
    h : float
        Heat transfer coefficient [W/m²·K]

    Notes:
    ------
    - Nusselt number: Nu = h*D/k (convective / conductive heat transfer)

    Correlations used:
    - Laminar flow (Re ≤ 4000): Nu = 4.36 (constant for uniform heat flux)

    Valid range:
    - 0.7 ≤ Pr ≤ 160
    - Re > 10,000 (but works reasonably well down to Re ≈ 4000)
    - L/D > 10 (fully developed flow)
    """

    # Heat transfer coefficient
    h = Nu * fluid_thermal_conductivity / pipe_diameter

    return h


def calculate_reynolds_number(
    velocity, pipe_diameter, fluid_density, fluid_viscosity
):
    """
    Calculate Reynolds number for pipe flow

    Parameters:
    -----------
    velocity : float
        Fluid velocity [m/s]
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_density : float
        Fluid density [kg/m³]
    fluid_viscosity : float
        Dynamic viscosity [Pa·s]

    Returns:
    --------
    Re : float
        Reynolds number [-]
    """
    return fluid_density * velocity * pipe_diameter / fluid_viscosity


def calculate_prandtl_number(
    fluid_viscosity, fluid_specific_heat, fluid_thermal_conductivity
):
    """
    Calculate Prandtl number for fluid

    Parameters:
    -----------
    fluid_viscosity : float
        Dynamic viscosity [Pa·s]
    fluid_specific_heat : float
        Specific heat capacity [J/kg·K]
    fluid_thermal_conductivity : float
        Thermal conductivity [W/m·K]

    Returns:
    --------
    Pr : float
        Prandtl number [-]
    """
    return fluid_viscosity * fluid_specific_heat / fluid_thermal_conductivity


def get_flow_regime(reynolds_number):
    """
    Determine flow regime based on Reynolds number

    Parameters
    ----------
    reynolds_number : float
        Reynolds number [-]

    Returns
    -------
    regime : str
        Flow regime: 'laminar', 'transitional', or 'turbulent'
    """
    if reynolds_number < RE_LAMINAR_MAX:
        return "laminar"
    elif reynolds_number < RE_TURBULENT_MIN:
        return "transitional"
    else:
        return "turbulent"


def estimate_pressure_drop_laminar(
    velocity, pipe_length, pipe_diameter, fluid_density, fluid_viscosity,
    check_regime=True
):
    """
    Estimate pressure drop for laminar flow using Darcy-Weisbach equation.

    Uses Hagen-Poiseuille friction factor: f = 64/Re.
    Compatible with Pyomo/CasADi symbolic types when check_regime=False.

    Parameters
    ----------
    velocity : float or symbolic
        Fluid velocity [m/s]
    pipe_length : float or symbolic
        Pipe length [m]
    pipe_diameter : float or symbolic
        Pipe inner diameter [m]
    fluid_density : float or symbolic
        Fluid density [kg/m³]
    fluid_viscosity : float or symbolic
        Dynamic viscosity [Pa·s]
    check_regime : bool, optional
        If True, warn when Re > RE_LAMINAR_MAX (default: True).
        Set to False for symbolic (Pyomo/CasADi) use.

    Returns
    -------
    pressure_drop : float or symbolic
        Pressure drop [Pa]
    friction_factor : float or symbolic
        Darcy friction factor [-]
    """
    Re = calculate_reynolds_number(
        velocity, pipe_diameter, fluid_density, fluid_viscosity
    )

    if check_regime:
        try:
            Re_val = np.asarray(Re)
            if np.any(Re_val > RE_LAMINAR_MAX):
                print(f"Warning: Re > {RE_LAMINAR_MAX}, flow may not be laminar")
        except (TypeError, ValueError):
            pass  # Skip check for symbolic types

    # Friction factor for laminar flow (Hagen-Poiseuille)
    f = 64 / Re

    # Darcy-Weisbach equation
    pressure_drop = (
        f * (pipe_length / pipe_diameter) * (fluid_density * velocity**2 / 2)
    )

    return pressure_drop, f


def estimate_pressure_drop_turbulent(
    velocity, pipe_length, pipe_diameter, fluid_density, fluid_viscosity,
    check_regime=True
):
    """
    Estimate pressure drop for turbulent flow using Darcy-Weisbach equation.

    Uses Blasius correlation for smooth pipes: f = 0.316/Re^0.25.
    Compatible with Pyomo/CasADi symbolic types when check_regime=False.

    Parameters
    ----------
    velocity : float or symbolic
        Fluid velocity [m/s]
    pipe_length : float or symbolic
        Pipe length [m]
    pipe_diameter : float or symbolic
        Pipe inner diameter [m]
    fluid_density : float or symbolic
        Fluid density [kg/m³]
    fluid_viscosity : float or symbolic
        Dynamic viscosity [Pa·s]
    check_regime : bool, optional
        If True, warn when Re < RE_TURBULENT_MIN (default: True).
        Set to False for symbolic (Pyomo/CasADi) use.

    Returns
    -------
    pressure_drop : float or symbolic
        Pressure drop [Pa]
    friction_factor : float or symbolic
        Darcy friction factor [-]
    """
    Re = calculate_reynolds_number(
        velocity, pipe_diameter, fluid_density, fluid_viscosity
    )

    if check_regime:
        try:
            Re_val = np.asarray(Re)
            if np.any(Re_val < RE_TURBULENT_MIN):
                print(f"Warning: Re < {RE_TURBULENT_MIN}, flow may not be fully turbulent")
        except (TypeError, ValueError):
            pass  # Skip check for symbolic types

    # Friction factor for turbulent flow (Blasius correlation)
    f = 0.316 / Re**0.25

    # Darcy-Weisbach equation
    pressure_drop = (
        f * (pipe_length / pipe_diameter) * (fluid_density * velocity**2 / 2)
    )

    return pressure_drop, f


def estimate_pressure_drop(
    velocity, pipe_length, pipe_diameter, fluid_density, fluid_viscosity
):
    """
    Estimate pressure drop using Darcy-Weisbach equation.

    Automatically selects laminar or turbulent correlation based on Re.
    Note: Not compatible with symbolic types (Pyomo/CasADi) due to conditional.
    Use estimate_pressure_drop_laminar or estimate_pressure_drop_turbulent instead.

    Parameters
    ----------
    velocity : float
        Fluid velocity [m/s]
    pipe_length : float
        Pipe length [m]
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_density : float
        Fluid density [kg/m³]
    fluid_viscosity : float
        Dynamic viscosity [Pa·s]

    Returns
    -------
    pressure_drop : float
        Pressure drop [Pa]
    friction_factor : float
        Darcy friction factor [-]
    """
    Re = calculate_reynolds_number(
        velocity, pipe_diameter, fluid_density, fluid_viscosity
    )

    # Friction factor correlation
    if Re < RE_LAMINAR_MAX:  # Laminar flow
        f = 64 / Re
    else:  # Turbulent flow (Blasius correlation for smooth pipes)
        f = 0.316 / Re**0.25

    # Darcy-Weisbach equation
    pressure_drop = (
        f * (pipe_length / pipe_diameter) * (fluid_density * velocity**2 / 2)
    )

    return pressure_drop, f


def make_reynolds_number_plot(
    fluid,
    pipe_diameter,
    fluid_temperatures,
    velocity_range=(0.02, 1.0),
    n_points=101,
    figsize=(7, 3.5),
    title=None,
):
    """
    Create a plot of Reynolds number vs flow velocity with flow regime regions.

    Parameters
    ----------
    fluid : FluidProperties
        Fluid properties object with temperature-dependent properties
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_temperatures : list of float
        Temperatures [K] at which to plot Reynolds number curves
    velocity_range : tuple
        (min, max) velocity range for plot [m/s]
    n_points : int
        Number of points for velocity array
    figsize : tuple
        Figure size
    title : str, optional
        Plot title. If None, auto-generates with fluid name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    velocity = np.logspace(
        np.log10(velocity_range[0]), np.log10(velocity_range[1]), n_points
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot Reynolds number for each temperature
    cmap = plt.get_cmap(DEFAULT_COLORMAP)
    colors = cmap(np.linspace(0.15, 0.85, len(fluid_temperatures)))

    for T, color in zip(fluid_temperatures, colors):
        T_C = T - 273.15
        rho = fluid.density(T)
        mu = fluid.viscosity(T)
        re = calculate_reynolds_number(velocity, pipe_diameter, rho, mu)
        ax.plot(velocity, re, color=color, label=f"{T_C:.0f}°C")

    # Set axis scales and limits (log-log plot)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Shade flow regime regions (horizontal bands based on Re thresholds)
    ax.axhspan(ylim[0], 2300, color="tab:green", alpha=0.1)
    ax.axhspan(4000, ylim[1], color="tab:red", alpha=0.1)

    # Horizontal lines at critical Reynolds numbers
    ax.axhline(2300, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(4000, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add text annotations for flow regimes (geometric means for log scale)
    x_text = np.sqrt(xlim[0] * xlim[1])
    ax.text(
        x_text,
        np.sqrt(ylim[0] * 2300),
        "Laminar",
        ha="center",
        va="center",
        fontsize=9,
        color="darkgreen",
        alpha=0.8,
    )
    ax.text(
        x_text,
        np.sqrt(2300 * 4000),
        "Transitional",
        ha="center",
        va="center",
        fontsize=9,
        color="gray",
        alpha=0.8,
    )
    ax.text(
        x_text,
        np.sqrt(4000 * ylim[1]),
        "Turbulent",
        ha="center",
        va="center",
        fontsize=9,
        color="darkred",
        alpha=0.8,
    )

    x_ticks = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    ax.set_xlabel("Flow velocity [m/s]")
    ax.set_ylabel("Re")
    if title is None:
        title = f"Reynolds Number for Absorber Pipe Flow with {fluid.name}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Temperature", loc="lower right")

    return fig, ax


def make_prandtl_number_plot(
    fluid,
    T_range=(373.15, 673.15),
    n_points=101,
    figsize=(7, 3.5),
    title=None,
):
    """
    Create a plot of Prandtl number vs temperature.

    Parameters
    ----------
    fluid : FluidProperties
        Fluid properties object with temperature-dependent properties
    T_range : tuple
        (min, max) temperature range [K] for plot
    n_points : int
        Number of points for temperature array
    figsize : tuple
        Figure size
    title : str, optional
        Plot title. If None, auto-generates with fluid name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    T = np.linspace(T_range[0], T_range[1], n_points)
    T_C = T - 273.15

    # Calculate Prandtl number
    Pr = fluid.prandtl_number(T)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(T_C, Pr, color="tab:blue", linewidth=2)

    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Pr")
    if title is None:
        title = f"Prandtl Number for {fluid.name}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])

    return fig, ax


def make_heat_transfer_coeff_plot(
    fluid,
    pipe_diameter,
    fluid_temperatures,
    velocity_range=(0.02, 1.0),
    n_points=101,
    figsize=(7, 3.5),
    title=None,
):
    """
    Create a plot of heat transfer coefficient vs flow velocity.

    Shows turbulent (Dittus-Boelter) and laminar heat transfer coefficients
    for multiple fluid temperatures.

    Parameters
    ----------
    fluid : FluidProperties
        Fluid properties object with temperature-dependent properties
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_temperatures : list of float
        Temperatures [K] at which to plot heat transfer coefficient curves
    velocity_range : tuple
        (min, max) velocity range for plot [m/s]
    n_points : int
        Number of points for velocity array
    figsize : tuple
        Figure size
    title : str, optional
        Plot title. If None, auto-generates with fluid name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    velocity = np.logspace(
        np.log10(velocity_range[0]), np.log10(velocity_range[1]), n_points
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heat transfer coefficient for each temperature
    cmap = plt.get_cmap(DEFAULT_COLORMAP)
    colors = cmap(np.linspace(0.15, 0.85, len(fluid_temperatures)))

    for T, color in zip(fluid_temperatures, colors):
        T_C = T - 273.15
        rho = fluid.density(T)
        mu = fluid.viscosity(T)
        k = fluid.thermal_conductivity(T)
        cp = fluid.heat_capacity(T)

        # Turbulent heat transfer coefficient (Dittus-Boelter)
        h_turb, _, _, _ = calculate_heat_transfer_coefficient_turbulent(
            velocity, pipe_diameter, rho, mu, k, cp
        )

        ax.semilogx(velocity, h_turb, color=color, label=f"{T_C:.0f}°C")

    x_ticks = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_ylim([0, None])

    ax.set_xlabel("Flow velocity [m/s]")
    ax.set_ylabel(r"$h$ [W/m²·K]")
    if title is None:
        title = f"Heat Transfer Coefficient for {fluid.name}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Temperature", loc="upper left")

    return fig, ax


if __name__ == "__main__":
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fluid = SYLTHERM800()
    d_pipe = PIPE_DIAMETER
    temperatures_C = [100, 200, 300, 400]
    temperatures_K = [T + 273.15 for T in temperatures_C]

    # Reynolds number plot
    fig1, ax1 = make_reynolds_number_plot(fluid, d_pipe, temperatures_K)
    plt.tight_layout()
    plt.savefig(plot_dir / "reynolds_number_vs_velocity.png", dpi=300)
    plt.show()

    # Prandtl number plot
    T_range_K = (100 + 273.15, 400 + 273.15)
    fig2, ax2 = make_prandtl_number_plot(fluid, T_range=T_range_K)
    plt.tight_layout()
    plt.savefig(plot_dir / "prandtl_number_vs_temperature.png", dpi=300)
    plt.show()

    # Heat transfer coefficient plot
    fig3, ax3 = make_heat_transfer_coeff_plot(fluid, d_pipe, temperatures_K)
    plt.tight_layout()
    plt.savefig(plot_dir / "heat_transfer_coeff_vs_velocity.png", dpi=300)
    plt.show()
