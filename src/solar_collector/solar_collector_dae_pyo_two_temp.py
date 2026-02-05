"""Solar collector thermal dynamics simulation using Pyomo DAE with two
temperature variables: fluid and pipe wall.

This module implements a 1D partial differential equation (PDE) model for heat
transfer in a solar collector pipe using Pyomo's differential-algebraic
equation (DAE) framework. The model includes separate temperatures for the
fluid (T_f) and pipe wall (T_p) with heat transfer between them and to ambient.

Functions
---------
create_pipe_flow_model(...) -> ConcreteModel
    Creates Pyomo model with fluid (T_f) and pipe wall (T_p) temperature
    variables, derivative variables, physical parameters, and time-varying
    input parameters (v, I, T_inlet). Optionally uses Dittus-Boelter for h_int.

add_pde_constraints(model) -> ConcreteModel
    Adds coupled PDE constraints for fluid and wall, initial conditions,
    and boundary conditions.

solve_model(model, ...) -> Results
    Applies finite difference discretization and solves with IPOPT.

plot_results(model, ...) -> (Figure, Figure, Figure)
    Plots time series and contour plots for both fluid and wall temperatures.

print_temp_profiles(model, ...)
    Prints temperature profiles and heat transfer analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    Param,
    SolverFactory,
    TransformationFactory,
    Var,
    exp as pyo_exp,
)

from solar_collector.config import PLOT_COLORS, VAR_INFO
from solar_collector.fluid_properties import SYLTHERM800
from solar_collector.heat_transfer import (
    calculate_heat_transfer_coefficient_turbulent,
    calculate_prandtl_number,
    calculate_reynolds_number,
)

# Constants
ZERO_C = 273.15  # K
THERMAL_DIFFUSIVITY = 0.25  # m²/s
FLUID_DENSITY = 800  # kg/m³
FLUID_DYNAMIC_VISCOSITY = 0.001  # Pa·s
FLUID_THERMAL_CONDUCTIVITY = 0.12  # W/m·K
FLUID_SPECIFIC_HEAT = 2000.0  # J/kg·K
HEAT_TRANSFER_COEFF_INT = 10.0  # W/m²·K (internal, pipe-to-fluid)
HEAT_TRANSFER_COEFF_EXT = 20.0  # W/m²·K (external, pipe-to-ambient)
PIPE_DIAMETER = 0.07  # m
PIPE_WALL_THICKNESS = 0.006  # m
COLLECTOR_LENGTH = 100.0  # m
PIPE_THERMAL_CONDUCTIVITY = 50.0  # W/m·K (typical for steel)
PIPE_DENSITY = 7850.0  # kg/m³ (steel)
PIPE_SPECIFIC_HEAT = 450.0  # J/kg·K (steel)

# Solar collector parameters (based on Yebra & Rhinehart model)
MIRROR_WIDTH = 5.76  # m (width of parabolic mirrors)
CONCENTRATION_FACTOR = 26.0  # Solar concentration ratio
OPTICAL_EFFICIENCY = 0.8  # Efficiency factor for mirror/alignment losses


def create_pipe_flow_model(
    fluid_props,
    L=COLLECTOR_LENGTH,
    t_final=5.0,
    n_x=50,
    n_t=50,
    velocity_func=None,
    irradiance_func=None,
    inlet_func=None,
    T_ref=ZERO_C + 300.0,
    thermal_diffusivity=THERMAL_DIFFUSIVITY,
    T_ambient=ZERO_C + 20.0,
    pipe_diameter=PIPE_DIAMETER,
    pipe_wall_thickness=PIPE_WALL_THICKNESS,
    heat_transfer_coeff_ext=HEAT_TRANSFER_COEFF_EXT,
    pipe_thermal_conductivity=PIPE_THERMAL_CONDUCTIVITY,
    pipe_density=PIPE_DENSITY,
    pipe_specific_heat=PIPE_SPECIFIC_HEAT,
    concentration_factor=CONCENTRATION_FACTOR,
    optical_efficiency=OPTICAL_EFFICIENCY,
    constant_density=True,
    constant_viscosity=True,
    constant_thermal_conductivity=True,
    constant_specific_heat=True,
    constant_heat_transfer_coeff=True,
):
    """
    Create Pyomo model for pipe flow heat transport PDE with fluid and wall.

    Parameters
    ----------
    fluid_props : FluidProperties
        Fluid properties object (e.g., SYLTHERM800()) providing temperature-
        dependent correlations for density, viscosity, thermal conductivity,
        and specific heat.
    L : float, default=100.0
        Length of solar collector section of pipe [m] (domain with heat input)
    t_final : float, default=5.0
        Final simulation time [s]
    n_x : int, default=110
        Number of spatial discretization points
    n_t : int, default=50
        Number of temporal discretization points
    velocity_func : callable, optional
        Function for time-varying fluid velocity v(t) [m/s].
        If None, uses default constant velocity.
    irradiance_func : callable, optional
        Function for time-varying solar irradiance I(t) [W/m²]
        (natural/direct normal irradiance before concentration).
        If None, uses default zero irradiance.
    inlet_func : callable, optional
        Function for time-varying inlet temperature T_inlet(t) [K].
        If None, uses default constant inlet temperature.
    T_ref : float, default=ZERO_C + 300.0
        Reference temperature [K] for evaluating constant properties.
    thermal_diffusivity : float, default=THERMAL_DIFFUSIVITY
        Fluid thermal diffusivity α [m²/s].
    T_ambient : float, default=ZERO_C + 20.0
        Ambient temperature [K] for convective heat loss.
    pipe_diameter : float, default=0.07
        Inner pipe diameter D [m].
    pipe_wall_thickness : float, default=0.006
        Pipe wall thickness d [m].
    heat_transfer_coeff_ext : float, default=20.0
        External heat transfer coefficient h_ext [W/m²·K] (pipe-to-ambient).
    pipe_thermal_conductivity : float, default=50.0
        Pipe wall thermal conductivity k_p [W/m·K].
    pipe_density : float, default=7850.0
        Pipe wall density ρ_p [kg/m³].
    pipe_specific_heat : float, default=450.0
        Pipe wall specific heat capacity cp_p [J/kg·K].
    concentration_factor : float, default=26.0
        Solar concentration ratio c (mirror width / effective absorber width).
    optical_efficiency : float, default=0.8
        Optical efficiency ε accounting for mirror/alignment losses.
    constant_density : bool, default=True
        If True, use constant density evaluated at T_ref.
        If False, use temperature-dependent Expression.
    constant_viscosity : bool, default=True
        If True, use constant viscosity evaluated at T_ref.
        If False, use temperature-dependent Expression.
    constant_thermal_conductivity : bool, default=True
        If True, use constant thermal conductivity evaluated at T_ref.
        If False, use temperature-dependent Expression.
    constant_specific_heat : bool, default=True
        If True, use constant specific heat evaluated at T_ref.
        If False, use temperature-dependent Expression.
    constant_heat_transfer_coeff : bool, default=True
        If True, use constant h_int evaluated at T_ref using Dittus-Boelter.
        If False, use temperature-dependent h_int Expression based on local
        fluid properties at T_f[t, x] using Dittus-Boelter correlation.

    Returns
    -------
    model : pyomo.ConcreteModel
        Pyomo model with variables, parameters, and derivative variables
        defined.

    Notes
    -----
    - Creates extended domain: 0 to L_extended = L * 1.1
    - Solar collector section: 0 < x <= L (with heat input to wall)
    - Buffer extension: L < x <= L_extended (no heat input)
    - Two temperature variables: T_f (fluid) and T_p (pipe wall)
    - Fluid properties can be constant or temperature-dependent Expressions
    """

    model = ConcreteModel()

    # Store fluid properties object and settings on model
    model.fluid_props = fluid_props
    model.T_ref = T_ref
    model.constant_density = constant_density
    model.constant_viscosity = constant_viscosity
    model.constant_thermal_conductivity = constant_thermal_conductivity
    model.constant_specific_heat = constant_specific_heat

    # Extend domain by 10% beyond nominal length
    L_extended = L * 1.1
    model.x = ContinuousSet(bounds=(0, L_extended))
    model.t = ContinuousSet(bounds=(0, t_final))

    # Store both nominal and extended pipe lengths
    model.L = L  # Nominal length with heat input
    model.L_extended = L_extended  # Full domain length

    # Temperature variables with proper bounds
    model.T_f = Var(model.t, model.x, bounds=(0.0, None))  # Fluid
    model.T_p = Var(model.t, model.x, bounds=(0.0, None))  # Pipe wall

    # Derivative variables for fluid temperature
    model.dT_f_dt = DerivativeVar(model.T_f, wrt=model.t)
    model.dT_f_dx = DerivativeVar(model.T_f, wrt=model.x)
    model.d2T_f_dx2 = DerivativeVar(model.T_f, wrt=(model.x, model.x))

    # Derivative variables for pipe wall temperature
    model.dT_p_dt = DerivativeVar(model.T_p, wrt=model.t)
    model.dT_p_dx = DerivativeVar(model.T_p, wrt=model.x)
    model.d2T_p_dx2 = DerivativeVar(model.T_p, wrt=(model.x, model.x))

    # Fluid thermal diffusivity (always constant for now)
    model.alpha_f = Param(initialize=thermal_diffusivity)  # [m²/s]

    # Fluid density: constant or temperature-dependent
    if constant_density:
        model._rho_f_const = Param(initialize=fluid_props.density(T_ref))

        @model.Expression(model.t, model.x)
        def rho_f(m, t, x):
            return m._rho_f_const
    else:
        @model.Expression(model.t, model.x)
        def rho_f(m, t, x):
            return m.fluid_props.density(m.T_f[t, x])

    # Fluid dynamic viscosity: constant or temperature-dependent
    if constant_viscosity:
        model._eta_f_const = Param(initialize=fluid_props.viscosity(T_ref))

        @model.Expression(model.t, model.x)
        def eta_f(m, t, x):
            return m._eta_f_const
    else:
        @model.Expression(model.t, model.x)
        def eta_f(m, t, x):
            return m.fluid_props.viscosity(m.T_f[t, x], exp=pyo_exp)

    # Fluid thermal conductivity: constant or temperature-dependent
    if constant_thermal_conductivity:
        model._lam_f_const = Param(
            initialize=fluid_props.thermal_conductivity(T_ref)
        )

        @model.Expression(model.t, model.x)
        def lam_f(m, t, x):
            return m._lam_f_const
    else:
        @model.Expression(model.t, model.x)
        def lam_f(m, t, x):
            return m.fluid_props.thermal_conductivity(m.T_f[t, x])

    # Fluid specific heat: constant or temperature-dependent
    if constant_specific_heat:
        model._cp_f_const = Param(initialize=fluid_props.heat_capacity(T_ref))

        @model.Expression(model.t, model.x)
        def cp_f(m, t, x):
            return m._cp_f_const
    else:
        @model.Expression(model.t, model.x)
        def cp_f(m, t, x):
            return m.fluid_props.heat_capacity(m.T_f[t, x])

    # Pipe properties
    model.rho_p = Param(initialize=pipe_density)  # [kg/m³]
    model.cp_p = Param(initialize=pipe_specific_heat)  # [J/kg·K]
    model.k_p = Param(initialize=pipe_thermal_conductivity)  # [W/m·K]

    # Geometric parameters
    model.T_ambient = Param(initialize=T_ambient)  # [K]
    model.D = Param(initialize=pipe_diameter)  # [m]
    model.d = Param(initialize=pipe_wall_thickness)  # [m]
    model.h_ext = Param(initialize=heat_transfer_coeff_ext)  # [W/m²·K]

    # Default parameter functions if none provided
    if velocity_func is None:

        def velocity_func(t):
            return 0.2  # velocity [m/s]

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

    # Solar collector parameters (Yebra & Rhinehart model)
    model.c = Param(initialize=concentration_factor)  # Concentration ratio
    model.epsilon = Param(initialize=optical_efficiency)  # Optical efficiency

    # Store heat transfer coefficient settings
    model.constant_heat_transfer_coeff = constant_heat_transfer_coeff

    # Calculate h_int using Dittus-Boelter correlation
    # Get fluid properties at reference temperature for initial calculation
    rho_ref = fluid_props.density(T_ref)
    eta_ref = fluid_props.viscosity(T_ref)
    lam_ref = fluid_props.thermal_conductivity(T_ref)
    cp_ref = fluid_props.heat_capacity(T_ref)

    # Calculate h_int at T_ref for printing and constant case
    v_initial = velocity_func(0.0)
    h_int_ref, Re_ref, Pr_ref, Nu_ref = (
        calculate_heat_transfer_coefficient_turbulent(
            v_initial,
            pipe_diameter,
            rho_ref,
            eta_ref,
            lam_ref,
            cp_ref,
        )
    )

    if constant_heat_transfer_coeff:
        print("Using constant h_int (Dittus-Boelter at T_ref):")
    else:
        print("Using temperature-dependent h_int (Dittus-Boelter):")
    print(f"  Reference temperature: {T_ref - ZERO_C:.0f}°C")
    print(f"  Initial velocity: {v_initial:.3f} m/s")
    print(f"  Reynolds number at T_ref: {Re_ref:.0f}")
    print(f"  Prandtl number at T_ref: {Pr_ref:.1f}")
    print(f"  Nusselt number at T_ref: {Nu_ref:.1f}")
    print(f"  Heat transfer coefficient at T_ref: {h_int_ref:.1f} W/m²·K")

    # Internal heat transfer coefficient: constant or temperature-dependent
    if constant_heat_transfer_coeff:
        model._h_int_const = Param(initialize=h_int_ref)

        @model.Expression(model.t, model.x)
        def h_int(m, t, x):
            return m._h_int_const
    else:
        # Temperature-dependent h_int using Dittus-Boelter correlation:
        # Re = rho * v * D / mu
        # Pr = mu * cp / k
        # Nu = 0.023 * Re^0.8 * Pr^0.4
        # h = Nu * k / D
        @model.Expression(model.t, model.x)
        def h_int(m, t, x):
            # Get local fluid properties (may be constant or T-dependent)
            rho = m.rho_f[t, x]
            eta = m.eta_f[t, x]
            lam = m.lam_f[t, x]
            cp = m.cp_f[t, x]
            v = m.v[t]
            D = m.D

            # Reynolds number
            Re = rho * v * D / eta
            # Prandtl number
            Pr = eta * cp / lam
            # Nusselt number (Dittus-Boelter)
            Nu = 0.023 * Re**0.8 * Pr**0.4
            # Heat transfer coefficient
            return Nu * lam / D

    # Time-varying parameters (will be initialized after discretization)
    model.v = Param(model.t, mutable=True)
    model.I = Param(model.t, mutable=True)  # Solar irradiance [W/m²]
    model.T_inlet = Param(model.t, mutable=True)

    return model


def add_pde_constraints(model):
    """
    Add PDE constraints and boundary/initial conditions for both temperatures
    """

    # Fluid temperature PDE constraint
    @model.Constraint(model.t, model.x)
    def fluid_pde_constraint(m, t, x):
        # Skip initial time
        if t == 0:
            return Constraint.Skip
        if x == 0:  # Skip inlet
            return Constraint.Skip

        # Heat transfer from pipe wall to fluid per unit volume [W/m³]
        # Heat transfer area per unit length: π * D [m²/m]
        # Heat transfer per unit length:
        #   h_int * π * D * (T_p - T_f) [W/m]
        # Heat transfer per unit volume:
        #   h_int * π * D * (T_p - T_f) / (π * D² / 4)
        # Simplified:
        #   h_int * 4 * (T_p - T_f) / D [W/m³]
        heat_to_fluid = m.h_int[t, x] * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D

        # Fluid PDE: (energy balance)
        #   ρcp ∂T_f/∂t + ρcp v(t)∂T_f/∂x = ρcp α_f∂²T_f/∂x² + q_to_fluid
        rho_cp = m.rho_f[t, x] * m.cp_f[t, x]
        return (
            rho_cp * m.dT_f_dt[t, x] + rho_cp * m.v[t] * m.dT_f_dx[t, x]
            == rho_cp * m.alpha_f * m.d2T_f_dx2[t, x] + heat_to_fluid
        )

    # Pipe wall temperature PDE constraint
    @model.Constraint(model.t, model.x)
    def wall_pde_constraint(m, t, x):
        # Skip initial time
        if t == 0:
            return Constraint.Skip

        # Density * specific heat for pipe wall
        rho_cp = m.rho_p * m.cp_p

        # For nominal physical pipe (0 < x <= L): include heat input
        if x <= m.L:
            # Solar heat input based on Yebra & Rhinehart model:
            # q_eff = I * c * ε / 2 [W/m²]
            # where:
            #   I = natural solar irradiance [W/m²]
            #   c = concentration factor (mirror width / absorber width)
            #   ε = optical efficiency (mirror/alignment losses)
            #   /2 = accounts for 180° illumination of pipe surface
            q_eff = m.I[t] * m.c * m.epsilon / 2.0

            # Heat input to pipe wall per unit volume [W/m³]
            # Outer diameter: D + 2*d
            # Heat input per unit length: q_eff * π * (D + 2*d) [W/m]
            # Pipe wall volume per unit length:
            #   π * ((D+2*d)² - D²) / 4 [m³/m]
            # Simplified:
            #   q_eff * 4 * (D + 2*d) / ((D+2*d)² - D²) [W/m³]
            D_outer = m.D + 2 * m.d
            heat_input_volumetric = (
                q_eff * 4.0 * D_outer / (D_outer**2 - m.D**2)
            )

            # Heat transfer to fluid per unit volume [W/m³]
            heat_to_fluid = m.h_int[t, x] * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D

            # Heat loss to ambient per unit volume [W/m³]
            # Heat loss per unit length:
            #   h_ext * π * (D + 2*d) * (T_p - T_ambient) [W/m]
            # Heat loss per unit volume:
            #   h_ext * π * (D + 2*d) * (T_p - T_ambient) /
            #   wall_volume_per_length
            heat_loss_volumetric = (
                m.h_ext
                * 4.0
                * D_outer
                * (m.T_p[t, x] - m.T_ambient)
                / (D_outer**2 - m.D**2)
            )

            # Pipe thermal diffusivity: k_p / (ρ_p * cp_p)
            alpha_p = m.k_p / (m.rho_p * m.cp_p)

            # Wall PDE: (energy balance)
            #   ρcp ∂T_p/∂t = α_p ρcp ∂²T_p/∂x²
            #      + q_input - q_to_fluid - q_to_ambient
            return rho_cp * m.dT_p_dt[t, x] == (
                rho_cp * alpha_p * m.d2T_p_dx2[t, x]
                + heat_input_volumetric
                - heat_to_fluid
                - heat_loss_volumetric
            )

        else:
            # For extended section (L < x <= L_extended): no heat input
            heat_to_fluid = m.h_int[t, x] * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D
            D_outer = m.D + 2 * m.d
            heat_loss_volumetric = (
                m.h_ext
                * 4.0
                * D_outer
                * (m.T_p[t, x] - m.T_ambient)
                / (D_outer**2 - m.D**2)
            )
            alpha_p = m.k_p / rho_cp

            return rho_cp * m.dT_p_dt[t, x] == (
                rho_cp * alpha_p * m.d2T_p_dx2[t, x]
                - heat_to_fluid
                - heat_loss_volumetric
            )

    # Initial conditions
    @model.Constraint(model.x)
    def fluid_initial_condition(m, x):
        if x == 0:
            return Constraint.Skip
        T_0 = ZERO_C + 270.0  # initial fluid temperature
        return m.T_f[0, x] == T_0

    @model.Constraint(model.x)
    def wall_initial_condition(m, x):
        T_0 = ZERO_C + 210.0  # initial wall temperature
        return m.T_p[0, x] == T_0

    # Inlet boundary condition for fluid
    @model.Constraint(model.t)
    def inlet_bc(m, t):
        return m.T_f[t, 0] == m.T_inlet[t]

    # Dummy objective (required by Pyomo)
    model.obj = Objective(expr=1)

    return model


def solve_model(
    model, n_x=110, n_t=50, max_iter=1000, tol=1e-6, print_level=5, tee=True
):
    """
    Discretize and solve the PDE model using finite differences

    Parameters:
    -----------
    model : pyomo.ConcreteModel
        The Pyomo model created by create_pipe_flow_model()
    n_x : int, default=110
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

    # Outlet boundary conditions (zero gradient)
    @model.Constraint(model.t)
    def fluid_outlet_bc(m, t):
        if t == 0:  # Skip initial time
            return Constraint.Skip
        x_vals = sorted(m.x)
        x_outlet = x_vals[-1]
        x_before = x_vals[-2]
        return m.T_f[t, x_outlet] == m.T_f[t, x_before]

    @model.Constraint(model.t)
    def wall_outlet_bc(m, t):
        if t == 0:  # Skip initial time
            return Constraint.Skip
        x_vals = sorted(m.x)
        x_outlet = x_vals[-1]
        x_before = x_vals[-2]
        return m.T_p[t, x_outlet] == m.T_p[t, x_before]

    print(
        f"Discretized with {len(model.x)} x points and {len(model.t)} t points"
    )

    # Initialize time-varying parameters after discretization
    for t in model.t:
        velocity_t = model.velocity_func(t)
        model.v[t].set_value(velocity_t)
        model.I[t].set_value(model.irradiance_func(t))
        model.T_inlet[t].set_value(model.inlet_func(t))

        # Note: h_int was already calculated in create_pipe_flow_model using
        # Dittus-Boelter correlation if use_dittus_boelter=True. For now, h_int
        # is spatially uniform and constant in time (evaluated at T_ref).

    # Provide good initial guess for temperature fields
    for t in model.t:
        for x in model.x:
            if t == 0:
                T_guess = ZERO_C + 50.0
            else:
                T_guess = ZERO_C + 50.0
            model.T_f[t, x].set_value(T_guess)
            model.T_p[t, x].set_value(T_guess)

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
    name="Oil + Pipe Model",
    var_info=VAR_INFO,
    colors=PLOT_COLORS,
    figsize=(8, 7.5),
):
    """
    Plot temperature field solutions for both fluid and pipe wall temperatures

    Parameters:
    -----------
    model : pyomo.ConcreteModel
        Solved Pyomo model containing temperature solutions
    t_eval : array-like
        Times to evaluate (used for compatibility, not actively used)
    x_eval : array-like
        Positions to evaluate (used for compatibility, not actively used)
    figsize : tuple, default=(8, 7.5)
        Figure size as (width, height) in inches (not used in current layout)

    Returns:
    --------
    fig1 : matplotlib.figure.Figure
        Figure with 4 time series plots (velocity, heat input, inlet temp,
        outlet temps)
    fig2 : matplotlib.figure.Figure
        Figure with contour plot of fluid temperature field (time vs position)
    fig3 : matplotlib.figure.Figure
        Figure with contour plot of pipe wall temperature field (time vs
        position)

    Notes:
    ------
    Figure 1 (4x1 layout):
    - Row 1: Fluid velocity over time
    - Row 2: Solar heat input over time
    - Row 3: Inlet temperature over time
    - Row 4: Outlet temperatures (both fluid and wall) at collector end

    Figure 2 & 3:
    - Contour plots with time on x-axis and position on y-axis
    - Consistent colorbar range (0-400°C) for easy comparison
    - Red dashed line marks end of solar collector section (x = L)
    """
    t_eval = np.array(t_eval).reshape(-1)
    x_eval = np.array(x_eval).reshape(-1)

    # Extract solution data
    t_vals = pd.Index(model.t)
    x_vals = pd.Index(model.x)

    # Create meshgrid
    T_grid, X_grid = np.meshgrid(t_vals, x_vals, indexing="ij")

    # Extract fluid temperature values
    temp_f_vals = np.array(
        [[pyo.environ.value(model.T_f[t, x]) for x in x_vals] for t in t_vals]
    )

    # Extract pipe wall temperature values
    temp_p_vals = np.array(
        [[pyo.environ.value(model.T_p[t, x]) for x in x_vals] for t in t_vals]
    )

    # Extract input parameter values
    v_vals = [pyo.environ.value(model.v[t]) for t in t_vals]
    I_vals = [pyo.environ.value(model.I[t]) for t in t_vals]
    T_inlet_vals = [pyo.environ.value(model.T_inlet[t]) for t in t_vals]
    inlet_temps_p = temp_p_vals[:, 0] - ZERO_C

    # Find index closest to collector end (x = L) for outlet temperatures
    end_idx = np.argmin(np.abs(np.array(x_vals) - model.L))
    outlet_temps_f = temp_f_vals[:, end_idx] - ZERO_C
    outlet_temps_p = temp_p_vals[:, end_idx] - ZERO_C

    # Define consistent temperature range for colorbars (0-400°C)
    temp_levels = np.linspace(0, 400, 21)

    # FIGURE 1: Time series plots (4 rows, 1 column)
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)

    # 1. Velocity time series
    ax1.plot(t_vals, v_vals, color=colors["v"], linewidth=2)
    ax1.set_ylabel("Velocity [m/s]")
    ax1.set_title(f"{name} - Fluid Velocity")
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
        label="Oil",
    )
    ax3.plot(
        t_vals, inlet_temps_p, color=colors["T_p"], linewidth=2, label="Wall"
    )
    ax3.set_ylabel("Inlet Temp [°C]")
    ax3.set_title(f"{name} - Collector Inlet Temperatures")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Outlet temperatures time series (both fluid and wall)
    ax4.plot(
        t_vals, outlet_temps_f, color=colors["T_f"], linewidth=2, label="Oil"
    )
    ax4.plot(
        t_vals, outlet_temps_p, color=colors["T_p"], linewidth=2, label="Wall"
    )
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Outlet Temp [°C]")
    ax4.set_title(f"{name} - Collector Outlet Temperatures")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # FIGURE 2: Fluid temperature contour (time on x-axis, position on y-axis)
    fig2, ax_contour_f = plt.subplots(1, 1, figsize=figsize)

    contour_f = ax_contour_f.contourf(
        T_grid.T,
        X_grid.T,
        (temp_f_vals - ZERO_C).T,
        levels=temp_levels,
        cmap="viridis",
        extend="both",
    )
    ax_contour_f.set_xlabel("Time [s]")
    ax_contour_f.set_ylabel("Position [m]")
    ax_contour_f.set_title(f"{name} - Oil Temperature Field")
    ax_contour_f.axhline(
        y=model.L,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="collector end",
    )
    ax_contour_f.legend()
    cbar_f = plt.colorbar(contour_f, ax=ax_contour_f)
    cbar_f.set_label("Temperature [°C]")

    plt.tight_layout()

    # FIGURE 3: Pipe wall temperature contour (time on x-axis, position
    # on y-axis)
    fig3, ax_contour_p = plt.subplots(1, 1, figsize=figsize)

    contour_p = ax_contour_p.contourf(
        T_grid.T,
        X_grid.T,
        (temp_p_vals - ZERO_C).T,
        levels=temp_levels,
        cmap="viridis",
        extend="both",
    )
    ax_contour_p.set_xlabel("Time [s]")
    ax_contour_p.set_ylabel("Position [m]")
    ax_contour_p.set_title(f"{name} - Pipe Wall Temperature Field")
    ax_contour_p.axhline(
        y=model.L,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="collector end",
    )
    ax_contour_p.legend()
    cbar_p = plt.colorbar(contour_p, ax=ax_contour_p)
    cbar_p.set_label("Temperature [°C]")

    return fig1, fig2, fig3


def print_temp_profiles(model, t_eval, x_eval):
    """
    Print temperature profiles for both fluid and pipe wall temperatures
    """
    t_eval = np.array(t_eval).reshape(-1)
    x_eval = np.array(x_eval).reshape(-1)

    # Extract solution data
    t_vals = pd.Index(model.t)
    x_vals = pd.Index(model.x)

    print("\n" + "=" * 80)
    print("TWO-TEMPERATURE MODEL ANALYSIS")
    print("=" * 80)

    # Temperature profiles at different times
    print("\nFluid and pipe wall temperature profiles at different times:")
    print(
        f"{'Time [s]':<9} {'T_f Inlet [K]':<13} {'T_f End [K]':<12} "
        f"{'T_p End [K]':<12} {'ΔT_f [K]':<11} {'T_p-T_f [K]':<12}"
    )
    print("-" * 79)

    time_indeces = t_vals.get_indexer(t_eval, method="nearest")
    pos_indeces = x_vals.get_indexer(x_eval, method="nearest")

    for i, t in zip(time_indeces, t_eval):
        T_f_inlet = model.T_f[t_vals[i], x_vals[0]].value
        T_f_end = model.T_f[t_vals[i], x_vals[pos_indeces[-2]]].value
        T_p_end = model.T_p[t_vals[i], x_vals[pos_indeces[-2]]].value
        delta_T_f = T_f_end - T_f_inlet
        temp_diff = T_p_end - T_f_end
        print(
            f"{t:<9.2f} {T_f_inlet:<13.1f} {T_f_end:<12.1f} "
            f"{T_p_end:<12.1f} {delta_T_f:<11.1f} {temp_diff:<12.1f}"
        )

    # Temperature evolution at different positions
    print("\nFluid temperature evolution at different positions:")
    print(
        f"{'Position [m]':<12} {'Initial [K]':<12} {'Final [K]':<12} "
        f"{'Change [K]':<12}"
    )
    print("-" * 48)

    for i, x in zip(pos_indeces, x_eval):
        T_f_initial = model.T_f[t_vals[0], x_vals[i]].value
        T_f_final = model.T_f[t_vals[-1], x_vals[i]].value
        temp_change = T_f_final - T_f_initial
        print(
            f"{x:<12.2f} {T_f_initial:<12.1f} {T_f_final:<12.1f} "
            f"{temp_change:<12.1f}"
        )

    print("\nPipe wall temperature evolution at different positions:")
    print(
        f"{'Position [m]':<12} {'Initial [K]':<12} {'Final [K]':<12} "
        f"{'Change [K]':<12}"
    )
    print("-" * 48)

    for i, x in zip(pos_indeces, x_eval):
        T_p_initial = model.T_p[t_vals[0], x_vals[i]].value
        T_p_final = model.T_p[t_vals[-1], x_vals[i]].value
        temp_change = T_p_final - T_p_initial
        print(
            f"{x:<12.2f} {T_p_initial:<12.1f} {T_p_final:<12.1f} "
            f"{temp_change:<12.1f}"
        )

    # Numerical diagnostics
    print("\nNumerical diagnostics:")
    print("-" * 40)

    # Find min/max temperatures for both variables
    all_temps_f = []
    all_temps_p = []
    all_temp_diffs = []

    for t in t_vals:
        for x in x_vals:
            T_f = model.T_f[t, x].value
            T_p = model.T_p[t, x].value
            all_temps_f.append(T_f)
            all_temps_p.append(T_p)
            all_temp_diffs.append(T_p - T_f)

    min_temp_f = min(all_temps_f)
    max_temp_f = max(all_temps_f)
    min_temp_p = min(all_temps_p)
    min_diff = min(all_temp_diffs)
    max_diff = max(all_temp_diffs)

    print(f"Fluid temperature range: {min_temp_f:.1f} K to {max_temp_f:.1f} K")
    max_temp_p = max(all_temps_p)
    print(
        f"Pipe wall temperature range: {min_temp_p:.1f} K to "
        f"{max_temp_p:.1f} K"
    )
    print(
        f"Temperature difference range: {min_diff:.1f} K to {max_diff:.1f} K"
    )

    # Check for instabilities (large gradients)
    max_gradient_f = 0
    max_gradient_p = 0
    worst_location_f = None
    worst_location_p = None

    for t in t_vals[1:]:  # Skip initial time
        for i, x in enumerate(x_vals[1:-1], 1):  # Skip boundaries
            if i < len(x_vals) - 1:
                # Approximate spatial gradients
                dx = x_vals[i + 1] - x_vals[i - 1]

                # Fluid gradient
                dT_f = (
                    model.T_f[t, x_vals[i + 1]].value
                    - model.T_f[t, x_vals[i - 1]].value
                )
                gradient_f = abs(dT_f / dx) if dx > 0 else 0
                if gradient_f > max_gradient_f:
                    max_gradient_f = gradient_f
                    worst_location_f = (t, x)

                # Pipe wall gradient
                dT_p = (
                    model.T_p[t, x_vals[i + 1]].value
                    - model.T_p[t, x_vals[i - 1]].value
                )
                gradient_p = abs(dT_p / dx) if dx > 0 else 0
                if gradient_p > max_gradient_p:
                    max_gradient_p = gradient_p
                    worst_location_p = (t, x)

    print(f"Maximum fluid spatial gradient: {max_gradient_f:.1f} K/m")
    if worst_location_f:
        print(
            f"Location of max fluid gradient: "
            f"t={worst_location_f[0]:.2f}s, x={worst_location_f[1]:.2f}m"
        )

    print(f"Maximum pipe wall spatial gradient: {max_gradient_p:.1f} K/m")
    if worst_location_p:
        print(
            f"Location of max pipe wall gradient: "
            f"t={worst_location_p[0]:.2f}s, x={worst_location_p[1]:.2f}m"
        )

    if max_gradient_f > 50 or max_gradient_p > 50:
        print("⚠️  WARNING: Large temp. gradients.")

    # Heat transfer analysis
    print("\nHeat transfer analysis:")
    print("-" * 30)

    # Average temperature difference in collector section
    avg_temp_diff = 0
    count = 0
    for t in t_vals[-5:]:  # Use last few time points
        for x in x_vals:
            if x <= model.L:  # Only in collector section
                T_f = model.T_f[t, x].value
                T_p = model.T_p[t, x].value
                avg_temp_diff += T_p - T_f
                count += 1

    if count > 0:
        avg_temp_diff /= count
        print(
            "Average temperature difference (T_p - T_f) in collector: "
            f"{avg_temp_diff:.1f} K"
        )

        # Estimate heat transfer rate per unit length
        # h_int is an indexed Expression; use the underlying constant value
        h_int_val = pyo.environ.value(model._h_int_const)
        D_val = pyo.environ.value(model.D)
        heat_transfer_rate = h_int_val * np.pi * D_val * avg_temp_diff  # W/m
        print(
            "Estimated heat transfer rate from wall to fluid: "
            f"{heat_transfer_rate:.0f} W/m"
        )

    print("=" * 80)
