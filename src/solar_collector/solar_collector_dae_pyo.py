"""Solar collector thermal dynamics simulation using Pyomo DAE.

This module implements a 1D partial differential equation (PDE) model for heat
transfer in a solar collector pipe using Pyomo's differential-algebraic
equation (DAE) framework. The model solves the advection-diffusion equation
with heat input and convective losses.

Functions
---------
create_collector_model(...) -> ConcreteModel
    Creates Pyomo model with temperature variable T, derivative variables,
    physical parameters, and time-varying input parameters (v, I, T_inlet).

add_pde_constraints(model) -> ConcreteModel
    Adds PDE constraint, initial conditions, and boundary conditions.

solve_model(model, ...) -> Results
    Applies finite difference discretization and solves with IPOPT.

plot_results(model, ...) -> (Figure, Figure)
    Plots time series (velocity, irradiance, temperatures) and contour plot.

print_temp_profiles(model, ...)
    Prints temperature profiles and numerical diagnostics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    SolverFactory,
    TransformationFactory,
    Var,
)

from solar_collector.config import PLOT_COLORS, VAR_INFO

# Constants
ZERO_C = 273.15  # K
THERMAL_DIFFUSIVITY = 0.25  # m²/s
FLUID_DENSITY = 800  # kg/m³
SPECIFIC_HEAT = 2000.0  # J/kg·K
HEAT_TRANSFER_COEFF_EXT = 10.0  # W/m²·K
PIPE_DIAMETER = 0.07  # m
COLLECTOR_LENGTH = 96.0  # m

# Solar collector parameters (based on Yebra & Rhinehart model)
MIRROR_WIDTH = 5.76  # m (width of parabolic mirrors)
CONCENTRATION_FACTOR = 26.0  # Solar concentration ratio
OPTICAL_EFFICIENCY = 0.8  # Efficiency factor for mirror/alignment losses


def create_collector_model(
    L=COLLECTOR_LENGTH,
    t_final=5.0,
    n_x=50,
    n_t=50,
    velocity_func=None,
    irradiance_func=None,
    inlet_func=None,
    thermal_diffusivity=THERMAL_DIFFUSIVITY,
    fluid_density=FLUID_DENSITY,
    specific_heat=SPECIFIC_HEAT,
    T_ambient=ZERO_C + 20.0,
    pipe_diameter=PIPE_DIAMETER,
    heat_transfer_coeff_ext=HEAT_TRANSFER_COEFF_EXT,
    concentration_factor=CONCENTRATION_FACTOR,
    optical_efficiency=OPTICAL_EFFICIENCY,
):
    """
    Create Pyomo model for pipe flow heat transport PDE

    Parameters:
    -----------
    L : float, default=10.0
        Length of solar collector section of pipe [m] (domain with heat input)
    t_final : float, default=5.0
        Final simulation time [s]
    n_x : int, default=50
        Number of spatial discretization points
    n_t : int, default=50
        Number of temporal discretization points
    velocity_func : callable, optional
        Function for time-varying fluid velocity v(t) [m/s]
        If None, uses default constant velocity
    irradiance_func : callable, optional
        Function for time-varying solar irradiance I(t) [W/m²]
        (natural/direct normal irradiance before concentration)
        If None, uses default zero irradiance
    inlet_func : callable, optional
        Function for time-varying inlet temperature T_inlet(t) [K]
        If None, uses default constant inlet temperature
    thermal_diffusivity : float, default=THERMAL_DIFFUSIVITY
        Thermal diffusivity α [m²/s]
    fluid_density : float, default=FLUID_DENSITY
        Fluid density ρ [kg/m³]
    specific_heat : float, default=SPECIFIC_HEAT
        Specific heat capacity cp [J/kg·K]
    T_ambient : float, default=ZERO_C + 20.0
        Ambient temperature [K] for convective heat loss
    pipe_diameter : float, default=0.05
        Inner pipe diameter [m] for heat loss calculations
    heat_transfer_coeff_ext : float, default=10.0
        Convective heat transfer coefficient h [W/m²·K]
    concentration_factor : float, default=26.0
        Solar concentration ratio c (mirror width / effective absorber width)
    optical_efficiency : float, default=0.8
        Optical efficiency ε accounting for mirror/alignment losses

    Returns:
    --------
    model : pyomo.ConcreteModel
        Pyomo model with variables, parameters, and derivative variables
        defined.

    Notes:
    ------
    - Creates extended domain: 0 to L_extended = L * 1.1
    - Solar collector section: 0 < x <= L (with heat input/loss)
    - Buffer extension: L < x <= L_extended (no heat input/loss)
    """

    model = ConcreteModel()

    # Extend domain by 10% beyond nominal length
    L_extended = L * 1.1
    model.x = ContinuousSet(bounds=(0, L_extended))
    model.t = ContinuousSet(bounds=(0, t_final))

    # Store both nominal and extended pipe lengths
    model.L = L  # Nominal length with heat input
    model.L_extended = L_extended  # Full domain length

    # Temperature variable with proper bounds
    model.T = Var(model.t, model.x, bounds=(0.0, None))

    # Derivative variables
    model.dTdt = DerivativeVar(model.T, wrt=model.t)
    model.dTdx = DerivativeVar(model.T, wrt=model.x)
    model.d2Tdx2 = DerivativeVar(model.T, wrt=(model.x, model.x))

    # Physical parameters
    model.alpha = Param(initialize=thermal_diffusivity)  # [m²/s]
    model.rho = Param(initialize=fluid_density)  # [kg/m³]
    model.cp = Param(initialize=specific_heat)  # [J/kg·K]

    # Heat loss parameters
    model.T_ambient = Param(initialize=T_ambient)  # [K]
    model.D = Param(initialize=pipe_diameter)  # [m]

    # Default parameter functions if none provided
    if velocity_func is None:

        def velocity_func(t):
            return 0.2  # velocity [m/s]
            # if t > 3:
            #     return 0.4
            # return 0.2

    if irradiance_func is None:

        def irradiance_func(t):
            # Natural solar irradiance [W/m²] (before concentration)
            # Typical DNI values: 800-1000 W/m² on a clear day
            if t > 60.0 and t <= 240.0:
                return 800.0
            return 0.0

    if inlet_func is None:

        def inlet_func(t):
            return ZERO_C + 270.0

    # Store functions for later use
    model.velocity_func = velocity_func
    model.irradiance_func = irradiance_func
    model.inlet_func = inlet_func

    # Add h parameter to model
    model.h_ext = Param(initialize=heat_transfer_coeff_ext)  # [W/m²·K]

    # Solar collector parameters (Yebra & Rhinehart model)
    model.c = Param(initialize=concentration_factor)  # Concentration ratio
    model.epsilon = Param(initialize=optical_efficiency)  # Optical efficiency

    # Store correlation settings
    model.pipe_diameter = pipe_diameter
    model.fluid_density = fluid_density
    model.specific_heat = specific_heat

    # Time-varying parameters (will be initialized after discretization)
    model.v = Param(model.t, mutable=True)
    model.I = Param(model.t, mutable=True)  # Solar irradiance [W/m²]
    model.T_inlet = Param(model.t, mutable=True)

    return model


def add_pde_constraints(model, T_initial):
    """
    Add PDE constraint and boundary/initial conditions
    """

    # Main PDE constraint - different for physical and extended sections
    @model.Constraint(model.t, model.x)
    def pde_constraint(m, t, x):
        # Skip initial time
        if t == 0:
            return Constraint.Skip
        if x == 0:  # Only skip inlet, NOT outlet
            return Constraint.Skip

        # Density * specific heat for fluid
        rho_cp = m.rho * m.cp

        # For nominal physical pipe (0 < x <= L): heat input/loss
        if x <= m.L:
            # Solar heat input based on Yebra & Rhinehart model:
            # q_eff = I * c * ε / 2 [W/m²]
            # where:
            #   I = natural solar irradiance [W/m²]
            #   c = concentration factor (mirror width / absorber width)
            #   ε = optical efficiency (mirror/alignment losses)
            #   /2 = accounts for 180° illumination of pipe surface
            q_eff = m.I[t] * m.c * m.epsilon / 2.0

            # Convert effective heat flux [W/m²] to volumetric [W/m³]
            # Heat input per unit length: q_eff * π * D [W/m]
            # Fluid volume per unit length: π * D² / 4 [m³/m]
            # Simplified: q_eff * 4 / D [W/m³]
            heat_input_volumetric = q_eff * 4.0 / m.D

            # Heat loss per unit length:
            #   h * π * D * (T - T_ambient) [W/m]
            # Heat loss per unit volume:
            #   h * π * D * (T - T_ambient) / (π * D² / 4) [W/m³]
            # Simplified:
            #   h * 4 * (T - T_ambient) / D [W/m³]
            heat_loss_volumetric = (
                m.h_ext * 4.0 * (m.T[t, x] - m.T_ambient) / m.D
            )

            # PDE: ρcp∂T/∂t + ρcpv(t) ∂T/∂x = ρcpα∂²T/∂x² + q_input - q_loss
            return rho_cp * m.dTdt[t, x] + rho_cp * m.v[t] * m.dTdx[t, x] == (
                rho_cp * m.alpha * m.d2Tdx2[t, x]
                + heat_input_volumetric
                - heat_loss_volumetric
            )
        else:
            # For extended section (L < x < L_extended): No heat input and no
            # heat loss.
            # PDE: ∂T/∂t + v(t)∂T/∂x = α∂²T/∂x² (no q(t) term, no heat loss)
            return (
                m.dTdt[t, x] + m.v[t] * m.dTdx[t, x]
                == m.alpha * m.d2Tdx2[t, x]
            )

    # Initial condition: T(x, 0) = T₀(x) - excluding x=0 (inlet)
    @model.Constraint(model.x)
    def initial_condition(m, x):
        if x == 0:
            return Constraint.Skip
        return m.T[0, x] == T_initial  # initial temperature

    # Inlet boundary condition: T(0, t) = T_inlet(t) at all times
    @model.Constraint(model.t)
    def inlet_bc(m, t):
        # Set inlet temperature at x=0
        return m.T[t, 0] == m.T_inlet[t]

    # Note: Outlet boundary condition: ∂T/∂x = 0 at x = L_extended is
    # implemented after discretization in the main script

    # Dummy objective (required by Pyomo)
    model.obj = Objective(expr=1)

    return model


def solve_model(
    model, n_x=50, n_t=50, max_iter=1000, tol=1e-6, print_level=5, tee=True
):
    """
    Discretize and solve the PDE model using finite differences

    Parameters:
    -----------
    model : pyomo.ConcreteModel
        The Pyomo model created by create_pipe_flow_model()
    n_x : int, default=50
        Number of finite elements for spatial discretization
    n_t : int, default=50
        Number of finite elements for temporal discretization
    max_iter : int, default=1000
        Maximum number of IPOPT solver iterations
    tol : float, default=1e-6
        Solver tolerance for convergence
    print_level : int, default=5
        IPOPT print level (0=no output, 5=detailed output)
    tee : bool, default=True
        Whether to display solver output to console

    Returns:
    --------
    results : pyomo solver results object
        Contains solver status, termination condition, and solution
        statistics.

    Notes:
    ------
    - Uses CENTRAL finite difference for spatial discretization
      (2nd order accuracy)
    - Uses BACKWARD Euler for temporal discretization (stability)
    - Initializes time-varying parameters after discretization
    - Provides uniform initial temperature guess for all variables
    """

    # Apply finite difference discretization
    # Use CENTRAL difference for spatial discretization (better accuracy)
    TransformationFactory("dae.finite_difference").apply_to(
        model, nfe=n_x, scheme="CENTRAL", wrt=model.x
    )

    # Temporal discretization (backward Euler for stability)
    TransformationFactory("dae.finite_difference").apply_to(
        model, nfe=n_t, scheme="BACKWARD", wrt=model.t
    )

    # Note: this outlet constraint can only be added after above
    # discretization is defined and discrete points exist
    @model.Constraint(model.t)
    def outlet_bc(m, t):
        if t == 0:  # Skip initial time
            return Constraint.Skip
        x_vals = sorted(m.x)
        x_outlet = x_vals[-1]
        x_before = x_vals[-2]
        return m.T[t, x_outlet] == m.T[t, x_before]

    print(
        f"Discretized with {len(model.x)} x points and {len(model.t)} t points"
    )

    # Initialize time-varying parameters after discretization
    for t in model.t:
        velocity_t = model.velocity_func(t)
        model.v[t].set_value(velocity_t)
        model.I[t].set_value(model.irradiance_func(t))
        model.T_inlet[t].set_value(model.inlet_func(t))

    # Provide good initial guess for temperature field
    for t in model.t:
        for x in model.x:
            if t == 0:
                T_guess = ZERO_C + 50.0
            else:
                T_guess = ZERO_C + 50.0
            model.T[t, x].set_value(T_guess)

    # Configure solver with simpler options
    solver = SolverFactory("ipopt")
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["print_level"] = print_level

    print("Solving with IPOPT...")
    results = solver.solve(model, tee=tee)

    return results


def plot_results(
    model,
    t_eval,
    x_eval,
    name="Oil Temp Model",
    var_info=VAR_INFO,
    colors=PLOT_COLORS,
    figsize=(8, 7.5),
):
    """
    Plot the temperature field solution with input functions visualization

    Parameters:
    -----------
    model : pyomo.ConcreteModel
        Solved Pyomo model containing temperature solution
    t_eval : array-like
        Times to evaluate (used for compatibility, not actively used)
    x_eval : array-like
        Positions to evaluate (used for compatibility, not actively used)
    figsize : tuple, default=(12, 6)
        Figure size as (width, height) in inches (not used in current layout)

    Returns:
    --------
    fig1 : matplotlib.figure.Figure
        Figure with 4 time series plots (velocity, heat input, inlet temp,
        outlet temp).
    fig2 : matplotlib.figure.Figure
        Figure with contour plot of temperature field (time vs position)

    Notes:
    ------
    Figure 1 (4x1 layout):
    - Row 1: Fluid velocity over time
    - Row 2: Solar heat input over time
    - Row 3: Inlet temperature over time
    - Row 4: Outlet temperature (at collector end) over time

    Figure 2:
    - Contour plot with time on x-axis and position on y-axis
    - Red dashed line marks end of solar collector section (x = L)
    """
    t_eval = np.array(t_eval).reshape(-1)
    x_eval = np.array(x_eval).reshape(-1)

    # Extract solution data
    t_vals = pd.Index(model.t)
    x_vals = pd.Index(model.x)

    # Create meshgrid
    T_grid, X_grid = np.meshgrid(t_vals, x_vals, indexing="ij")

    # Extract temperature values
    temp_vals = np.array(
        [[pyo.environ.value(model.T[t, x]) for x in x_vals] for t in t_vals]
    )

    # Extract input parameter values
    v_vals = [pyo.environ.value(model.v[t]) for t in t_vals]
    I_vals = [pyo.environ.value(model.I[t]) for t in t_vals]
    T_inlet_vals = [pyo.environ.value(model.T_inlet[t]) for t in t_vals]

    # Find index closest to collector end (x = L) for outlet temperature
    end_idx = np.argmin(np.abs(np.array(x_vals) - model.L))
    outlet_temps = temp_vals[:, end_idx] - ZERO_C

    # Define consistent temperature range for colorbars (0-400°C)
    temp_levels = np.linspace(0, 400, 21)

    # FIGURE 1: Time series plots (4 rows, 1 column)
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)

    # 1. Velocity time series
    ax1.plot(t_vals, v_vals, color=colors["v"], linewidth=2)
    ax1.set_ylabel("Velocity [m/s]")
    ax1.set_title(f"{name} - {var_info['v']['long_name']}")
    ax1.grid(True, alpha=0.3)

    # 2. Solar irradiance time series
    ax2.plot(
        t_vals,
        np.array(I_vals),
        color=colors["q_solar_conc"],
        linewidth=2,
    )
    ax2.set_ylabel("Irradiance [W/m²]")
    ax2.set_title(f"{name} - Solar Irradiance (DNI)")
    ax2.grid(True, alpha=0.3)

    # 3. Inlet temperature time series
    ax3.plot(
        t_vals,
        np.array(T_inlet_vals) - ZERO_C,
        color=colors["T_f"],
        linewidth=2,
    )
    ax3.set_ylabel("Inlet Temp [°C]")
    ax3.set_title(f"{name} - {var_info['T_inlet']['long_name']}")
    ax3.grid(True, alpha=0.3)

    # 4. Outlet temperature time series
    ax4.plot(t_vals, outlet_temps, color=colors["T_f"], linewidth=2)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Outlet Temp [°C]")
    ax4.set_title(f"{name} - Collector Oil Outlet Temperature")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # FIGURE 2: Contour plot (time on x-axis, position on y-axis)
    fig2, ax_contour = plt.subplots(1, 1, figsize=figsize)

    contour = ax_contour.contourf(
        T_grid.T,
        X_grid.T,
        (temp_vals - ZERO_C).T,
        levels=temp_levels,
        cmap="viridis",
        extend="both",
    )
    ax_contour.set_xlabel("Time [s]")
    ax_contour.set_ylabel("Position [m]")
    ax_contour.set_title(f"{name} - Oil Temperature Field")
    ax_contour.axhline(
        y=model.L,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"collector end (x={model.L}m)",
    )
    ax_contour.legend()
    cbar = plt.colorbar(contour, ax=ax_contour)
    cbar.set_label("Temperature [°C]")

    return fig1, fig2


def print_temp_profiles(model, t_eval, x_eval):
    """
    Print temperature profiles at different times and positions
    """
    t_eval = np.array(t_eval).reshape(-1)
    x_eval = np.array(x_eval).reshape(-1)

    # Extract solution data
    t_vals = pd.Index(model.t)
    x_vals = pd.Index(model.x)

    print("\n" + "=" * 60)
    print("TEMPERATURE PROFILE ANALYSIS")
    print("=" * 60)

    # Temperature at different times along the pipe
    print("\nTemperature profiles at different times:")
    print(
        f"{'Time [s]':<8} {'Inlet [K]':<10} {'Outlet [K]':<10} "
        f"{'End [K]':<10} {'ΔT [K]':<10}"
    )
    print("-" * 50)

    time_indeces = t_vals.get_indexer(t_eval, method="nearest")
    pos_indeces = x_vals.get_indexer(x_eval, method="nearest")
    for i, t in zip(time_indeces, t_eval):  # Sample times
        T_inlet = model.T[t_vals[i], x_vals[0]].value
        T_penult = model.T[t_vals[i], x_vals[pos_indeces[-2]]].value
        T_outlet = model.T[t_vals[i], x_vals[-1]].value
        delta_T = T_outlet - T_inlet
        print(
            f"{t:<8.2f} {T_inlet:<10.1f} {T_penult:<10.1f} {T_outlet:<10.1f} "
            f"{delta_T:<10.1f}"
        )

    # Temperature evolution at different positions
    print("\nTemperature evolution at different positions:")
    print(
        f"{'Position [m]':<12} {'Initial [K]':<12} {'Final [K]':<12} "
        f"{'Change [K]':<12}"
    )
    print("-" * 50)

    for i, x in zip(pos_indeces, x_eval):  # Sample positions
        T_initial = model.T[t_vals[0], x_vals[i]].value
        T_final = model.T[t_vals[-1], x_vals[i]].value
        temp_change = T_final - T_initial
        print(
            f"{x:<12.2f} {T_initial:<12.1f} {T_final:<12.1f} "
            f"{temp_change:<12.1f}"
        )

    # Check for numerical issues
    print("\nNumerical diagnostics:")
    print("-" * 30)

    # Find min/max temperatures
    all_temps = []
    for t in t_vals:
        for x in x_vals:
            all_temps.append(model.T[t, x].value)

    min_temp = min(all_temps)
    max_temp = max(all_temps)
    print(f"Temperature range: {min_temp:.1f} K to {max_temp:.1f} K")

    # Check for instabilities (large gradients)
    max_gradient = 0
    worst_location = None
    for t in t_vals[1:]:  # Skip initial time
        for i, x in enumerate(x_vals[1:-1], 1):  # Skip boundaries
            if i < len(x_vals) - 1:
                # Approximate spatial gradient
                dx = x_vals[i + 1] - x_vals[i - 1]
                dT = (
                    model.T[t, x_vals[i + 1]].value
                    - model.T[t, x_vals[i - 1]].value
                )
                gradient = abs(dT / dx) if dx > 0 else 0
                if gradient > max_gradient:
                    max_gradient = gradient
                    worst_location = (t, x)

    print(f"Maximum spatial gradient: {max_gradient:.1f} K/m")
    if worst_location:
        print(
            f"Location of max gradient: t={worst_location[0]:.2f}s, "
            f"x={worst_location[1]:.2f}m"
        )

    if max_gradient > 50:  # Arbitrary threshold for concern
        print("⚠️  WARNING: Large temp. gradients.")

    print("=" * 60)
