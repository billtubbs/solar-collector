"""Solar collector thermal dynamics simulation using Pyomo DAE with two
temperature variables: fluid and pipe wall.

This module implements a 1D partial differential equation (PDE) model for heat
transfer in a solar collector pipe using Pyomo's differential-algebraic
equation (DAE) framework. The model includes separate temperatures for the
fluid (T_f) and pipe wall (T_p) with heat transfer between them and to ambient.

Dynamic Model Functions
-----------------------
create_collector_model(...) -> ConcreteModel
    Creates Pyomo model with fluid (T_f) and pipe wall (T_p) temperature
    variables, derivative variables, physical parameters, and time-varying
    input parameters (v, I, T_inlet). Optionally uses Dittus-Boelter for h_int.

add_pde_constraints(model) -> ConcreteModel
    Adds coupled PDE constraints for fluid and wall, initial conditions,
    and boundary conditions.

solve_model(model, ...) -> Results
    Applies finite difference discretization and solves with IPOPT.

Steady-State Model Functions
----------------------------
create_collector_model_steady_state(...) -> ConcreteModel
    Creates Pyomo model for steady-state analysis with only spatial domain.
    Input parameters (mass_flow_rate, irradiance, T_inlet) are scalars.

add_steady_state_constraints(model) -> ConcreteModel
    Adds steady-state ODE constraints (time derivatives = 0) and boundary
    conditions.

solve_steady_state_model(model, ...) -> Results
    Discretizes spatial domain and solves steady-state model with IPOPT.

Plotting Functions
------------------
plot_time_series(t_vals, data_series, ...) -> (Figure, Axes)
    General-purpose function for plotting multiple time series on stacked
    subplots.

plot_temperature_field(t_vals, x_vals, temp_vals, ...) -> (Figure, Axes)
    General-purpose function for plotting 2D temperature contour plots
    (time vs position).

extract_model_data(model) -> dict
    Extracts time series and temperature field data from a solved model.

plot_results(model, ...) -> (Figure, Figure, Figure)
    Convenience function that creates time series and contour plots for
    both fluid and wall temperatures (uses the above functions).

plot_steady_state_results(model, ...) -> (Figure, Axes)
    Plots steady-state temperature profiles T_f(x) and T_p(x).

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
    Objective,
    Param,
    SolverFactory,
    TransformationFactory,
    Var,
)
from pyomo.environ import (
    exp as pyo_exp,
)

from solar_collector.config import PLOT_COLORS, VAR_INFO
from solar_collector.heat_transfer import (
    calculate_heat_transfer_coefficient_turbulent,
)

# Constants (Syltherm 800 at 300°C)
ZERO_C = 273.15  # K
FLUID_DENSITY = 671.0  # kg/m³
FLUID_DYNAMIC_VISCOSITY = 0.00047  # Pa·s
FLUID_THERMAL_CONDUCTIVITY = 0.0824  # W/m·K
FLUID_SPECIFIC_HEAT = 2086.0  # J/kg·K
AXIAL_DISPERSION_COEFF = 1e-4  # m²/s (turbulent axial dispersion)
HEAT_TRANSFER_COEFF_INT = 10.0  # W/m²·K (internal, pipe-to-fluid)
HEAT_TRANSFER_COEFF_EXT = 20.0  # W/m²·K (external, pipe-to-ambient)
PIPE_DIAMETER = 0.07  # m
PIPE_WALL_THICKNESS = 0.006  # m
COLLECTOR_LENGTH = 96.0  # m
PIPE_THERMAL_CONDUCTIVITY = 50.0  # W/m·K (typical for steel)
PIPE_DENSITY = 7850.0  # kg/m³ (steel)
PIPE_SPECIFIC_HEAT = 450.0  # J/kg·K (steel)

# Solar collector parameters (based on Yebra & Rhinehart model)
MIRROR_WIDTH = 5.76  # m (width of parabolic mirrors)
CONCENTRATION_FACTOR = 26.0  # Solar concentration ratio
OPTICAL_EFFICIENCY = 0.8  # Efficiency factor for mirror/alignment losses


def create_collector_model(
    fluid_props,
    L=COLLECTOR_LENGTH,
    t_final=5.0,
    n_x=50,
    n_t=50,
    mass_flow_rate_func=None,
    irradiance_func=None,
    T_inlet_func=None,
    T_ref=ZERO_C + 300.0,
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
    axial_dispersion_coeff=AXIAL_DISPERSION_COEFF,
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
    mass_flow_rate_func : callable, optional
        Function for time-varying mass flow rate m_dot(t) [kg/s].
        If None, uses default constant mass flow rate.
        Velocity is computed as v = m_dot / (rho_ref * A) where A = π*D²/4.
    irradiance_func : callable, optional
        Function for time-varying solar irradiance I(t) [W/m²]
        (natural/direct normal irradiance before concentration).
        If None, uses default zero irradiance.
    T_inlet_func : callable, optional
        Function for time-varying absorber inlet fluid temperature T_inlet(t) [K].
        If None, uses default constant inlet temperature.
    T_ref : float, default=ZERO_C + 300.0
        Reference temperature [K] for evaluating constant properties.
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
    axial_dispersion_coeff : float, default=1e-4
        Axial dispersion coefficient D_ax [m²/s] for turbulent mixing.
        This characterizes the effective axial spreading of heat due to
        turbulent eddies, typically 2-3 orders of magnitude larger than
        molecular thermal diffusivity.

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
        model._k_f_const = Param(
            initialize=fluid_props.thermal_conductivity(T_ref)
        )

        @model.Expression(model.t, model.x)
        def k_f(m, t, x):
            return m._k_f_const
    else:

        @model.Expression(model.t, model.x)
        def k_f(m, t, x):
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

    # Axial dispersion coefficient for turbulent flow [m²/s]
    # This characterizes effective axial heat spreading due to turbulent mixing,
    # typically 2-3 orders of magnitude larger than molecular thermal diffusivity
    model.D_ax = Param(initialize=axial_dispersion_coeff)

    # Pipe properties
    model.rho_p = Param(initialize=pipe_density)  # [kg/m³]
    model.cp_p = Param(initialize=pipe_specific_heat)  # [J/kg·K]
    model.k_p = Param(initialize=pipe_thermal_conductivity)  # [W/m·K]

    # Geometric parameters
    model.T_ambient = Param(initialize=T_ambient)  # [K]
    model.D = Param(initialize=pipe_diameter)  # [m]
    model.d = Param(initialize=pipe_wall_thickness)  # [m]
    model.h_ext = Param(initialize=heat_transfer_coeff_ext)  # [W/m²·K]

    # Pipe cross-sectional area for mass flow rate to velocity conversion
    pipe_area = np.pi * pipe_diameter**2 / 4.0
    model.A = Param(initialize=pipe_area)  # [m²]

    # Reference density for velocity calculation (constant)
    rho_ref = fluid_props.density(T_ref)

    # Default parameter functions if none provided
    if mass_flow_rate_func is None:
        # Default mass flow rate gives ~0.2 m/s at reference density
        # m_dot = v * rho * A = 0.2 * rho_ref * A
        default_m_dot = 0.2 * rho_ref * pipe_area

        def mass_flow_rate_func(t):
            return default_m_dot  # mass flow rate [kg/s]

    if irradiance_func is None:

        def irradiance_func(t):
            # Natural solar irradiance [W/m²] (before concentration)
            # Typical DNI values: 800-1000 W/m² on a clear day
            if t > 60.0 and t <= 240.0:
                return 800.0
            return 0.0

    if T_inlet_func is None:

        def T_inlet_func(t):
            return ZERO_C + 270.0  # Absorber inlet fluid temperature [K]

    # Store functions for later use
    model.mass_flow_rate_func = mass_flow_rate_func
    model.irradiance_func = irradiance_func
    model.T_inlet_func = T_inlet_func

    # Store reference density for velocity calculation
    model.rho_ref = Param(initialize=rho_ref)

    # Solar collector parameters (Yebra & Rhinehart model)
    model.c = Param(initialize=concentration_factor)  # Concentration ratio
    model.epsilon = Param(initialize=optical_efficiency)  # Optical efficiency

    # Store heat transfer coefficient settings
    model.constant_heat_transfer_coeff = constant_heat_transfer_coeff

    # Time-varying parameters (will be initialized after discretization)
    model.m_dot = Param(model.t, mutable=True)  # Mass flow rate [kg/s]
    model.I = Param(model.t, mutable=True)  # Solar irradiance [W/m²]
    model.T_inlet = Param(model.t, mutable=True)

    # Velocity as Expression: v(t, x) = m_dot(t) / (rho_f(t, x) * A)
    # This correctly handles mass conservation when density varies with temperature
    @model.Expression(model.t, model.x)
    def v(m, t, x):
        return m.m_dot[t] / (m.rho_f[t, x] * m.A)

    # Calculate h_int using Dittus-Boelter correlation
    # Get fluid properties at reference temperature for initial calculation
    eta_ref = fluid_props.viscosity(T_ref)
    lam_ref = fluid_props.thermal_conductivity(T_ref)
    cp_ref = fluid_props.heat_capacity(T_ref)

    # Calculate initial velocity from mass flow rate
    m_dot_initial = mass_flow_rate_func(0.0)
    v_initial = m_dot_initial / (rho_ref * pipe_area)

    # Calculate h_int at T_ref for printing and constant case
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
    print(f"  Initial mass flow rate: {m_dot_initial:.3f} kg/s")
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
            k = m.k_f[t, x]
            cp = m.cp_f[t, x]
            local_v = m.v[t, x]
            D = m.D

            # Reynolds number
            Re = rho * local_v * D / eta
            # Prandtl number
            Pr = eta * cp / k
            # Nusselt number (Dittus-Boelter)
            Nu = 0.023 * Re**0.8 * Pr**0.4
            # Heat transfer coefficient
            return Nu * k / D

    return model


def add_pde_constraints(
    model,
    T_f_initial=ZERO_C + 270.0,
    T_p_initial=ZERO_C + 210.0,
):
    """
    Add PDE constraints and boundary/initial conditions for both temperatures.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        The Pyomo model created by create_collector_model().
    T_f_initial : float or array-like, default=ZERO_C + 270.0
        Initial fluid temperature [K] at t=0. If float, uniform temperature
        for all x > 0. If array, must match the number of x points after
        discretization (values are mapped to x positions in order).
    T_p_initial : float or array-like, default=ZERO_C + 210.0
        Initial pipe wall temperature [K] at t=0. If float, uniform
        temperature for all x. If array, must match the number of x points.

    Returns
    -------
    model : pyomo.ConcreteModel
        The model with PDE constraints, initial conditions, and boundary
        conditions added.
    """
    # Store initial conditions on model for use in constraints
    model.T_f_initial = T_f_initial
    model.T_p_initial = T_p_initial

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
        #   ρcp ∂T_f/∂t + ρcp v(t,x)∂T_f/∂x = ρcp D_ax ∂²T_f/∂x² + q_to_fluid
        # Note: v(t,x) = m_dot(t) / (rho_f(t,x) * A) to conserve mass
        rho_cp = m.rho_f[t, x] * m.cp_f[t, x]
        return (
            rho_cp * m.dT_f_dt[t, x] + rho_cp * m.v[t, x] * m.dT_f_dx[t, x]
            == rho_cp * m.D_ax * m.d2T_f_dx2[t, x] + heat_to_fluid
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
            heat_to_fluid = (
                m.h_int[t, x] * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D
            )

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
            heat_to_fluid = (
                m.h_int[t, x] * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D
            )
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
    # Get sorted x values for array indexing
    x_vals = sorted(model.x)

    @model.Constraint(model.x)
    def fluid_initial_condition(m, x):
        if x == 0:
            return Constraint.Skip
        # Handle float or array initial condition
        if np.isscalar(m.T_f_initial):
            T_0 = m.T_f_initial
        else:
            idx = x_vals.index(x)
            T_0 = m.T_f_initial[idx]
        return m.T_f[0, x] == T_0

    @model.Constraint(model.x)
    def wall_initial_condition(m, x):
        # Handle float or array initial condition
        if np.isscalar(m.T_p_initial):
            T_0 = m.T_p_initial
        else:
            idx = x_vals.index(x)
            T_0 = m.T_p_initial[idx]
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
        The Pyomo model created by create_collector_model()
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
        model.m_dot[t].set_value(model.mass_flow_rate_func(t))
        model.I[t].set_value(model.irradiance_func(t))
        model.T_inlet[t].set_value(model.T_inlet_func(t))

        # Note: h_int was already calculated in create_collector_model using
        # Dittus-Boelter correlation. Velocity v(t,x) is computed as an
        # Expression from m_dot(t) and rho_f(t,x) to conserve mass.

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


# =============================================================================
# Steady-State Model Functions
# =============================================================================


def create_collector_model_steady_state(
    fluid_props,
    L=COLLECTOR_LENGTH,
    n_x=50,
    mass_flow_rate=7.5,
    irradiance=800.0,
    T_inlet=ZERO_C + 200.0,
    T_ref=ZERO_C + 300.0,
    T_ambient=ZERO_C + 25.0,
    D=PIPE_DIAMETER,
    d=PIPE_WALL_THICKNESS,
    h_ext=HEAT_TRANSFER_COEFF_EXT,
    concentration_factor=CONCENTRATION_FACTOR,
    optical_efficiency=OPTICAL_EFFICIENCY,
    axial_dispersion_coeff=AXIAL_DISPERSION_COEFF,
    constant_density=True,
    constant_viscosity=True,
    constant_thermal_conductivity=True,
    constant_specific_heat=True,
    constant_heat_transfer_coeff=True,
):
    """
    Create a Pyomo model for steady-state solar collector heat transfer.

    In steady state, all time derivatives are zero (∂T/∂t = 0), so we solve
    for the spatial temperature profiles T_f(x) and T_p(x) given constant
    operating conditions.

    Parameters
    ----------
    fluid_props : FluidProperties
        Object with methods: density(T), viscosity(T), thermal_conductivity(T),
        heat_capacity(T) for the heat transfer fluid.
    L : float, default=COLLECTOR_LENGTH
        Length of the solar collector pipe [m].
    n_x : int, default=50
        Number of initial spatial points (will be refined during solve).
    mass_flow_rate : float, default=7.5
        Mass flow rate [kg/s].
    irradiance : float, default=800.0
        Solar irradiance [W/m²].
    T_inlet : float, default=ZERO_C + 200.0
        Inlet fluid temperature [K].
    T_ref : float, default=ZERO_C + 300.0
        Reference temperature for constant fluid properties [K].
    T_ambient : float, default=ZERO_C + 25.0
        Ambient temperature for heat loss [K].
    D : float, default=PIPE_DIAMETER
        Inner diameter of the absorber pipe [m].
    d : float, default=PIPE_WALL_THICKNESS
        Pipe wall thickness [m].
    h_ext : float, default=HEAT_TRANSFER_COEFF_EXT
        External heat transfer coefficient [W/(m²·K)].
    concentration_factor : float, default=CONCENTRATION_FACTOR
        Solar concentration ratio.
    optical_efficiency : float, default=OPTICAL_EFFICIENCY
        Optical efficiency of collector.
    axial_dispersion_coeff : float, default=AXIAL_DISPERSION_COEFF
        Axial dispersion coefficient for turbulent flow [m²/s].
    constant_density : bool, default=True
        If True, use constant density at T_ref.
    constant_viscosity : bool, default=True
        If True, use constant viscosity at T_ref.
    constant_thermal_conductivity : bool, default=True
        If True, use constant thermal conductivity at T_ref.
    constant_specific_heat : bool, default=True
        If True, use constant specific heat at T_ref.
    constant_heat_transfer_coeff : bool, default=True
        If True, use constant h_int calculated at T_ref.

    Returns
    -------
    model : pyomo.ConcreteModel
        Pyomo model with variables, parameters, and expressions for
        steady-state analysis.
    """
    model = ConcreteModel()

    # Store reference temperature and fluid properties object
    model.T_ref = T_ref
    model.fluid_props = fluid_props

    # Store constant property flags for later reference
    model.constant_density = constant_density
    model.constant_viscosity = constant_viscosity
    model.constant_thermal_conductivity = constant_thermal_conductivity
    model.constant_specific_heat = constant_specific_heat
    model.constant_heat_transfer_coeff = constant_heat_transfer_coeff

    # Extended spatial domain for outlet boundary condition stability
    L_extended = L * 1.1

    # Spatial domain (continuous set) - no time domain for steady state
    model.x = ContinuousSet(bounds=(0.0, L_extended))

    # Temperature variables (functions of x only)
    model.T_f = Var(model.x, initialize=T_ref)  # Fluid temperature [K]
    model.T_p = Var(model.x, initialize=T_ref)  # Pipe wall temperature [K]

    # Derivative variables (spatial only)
    model.dT_f_dx = DerivativeVar(model.T_f, wrt=model.x)
    model.d2T_f_dx2 = DerivativeVar(model.T_f, wrt=(model.x, model.x))
    model.dT_p_dx = DerivativeVar(model.T_p, wrt=model.x)
    model.d2T_p_dx2 = DerivativeVar(model.T_p, wrt=(model.x, model.x))

    # Geometric parameters
    model.D = Param(initialize=D)  # Inner pipe diameter [m]
    model.d = Param(initialize=d)  # Pipe wall thickness [m]
    model.L = Param(initialize=L)  # Collector length [m]
    model.A = Param(initialize=np.pi * D**2 / 4.0)  # Flow cross-section [m²]

    # Pipe material properties
    model.rho_p = Param(initialize=PIPE_DENSITY)
    model.cp_p = Param(initialize=PIPE_SPECIFIC_HEAT)
    model.k_p = Param(initialize=PIPE_THERMAL_CONDUCTIVITY)
    model.h_ext = Param(initialize=h_ext)
    model.T_ambient = Param(initialize=T_ambient)

    # Axial dispersion coefficient for turbulent flow [m²/s]
    model.D_ax = Param(initialize=axial_dispersion_coeff)

    # Solar collector parameters
    model.c = Param(initialize=concentration_factor)
    model.epsilon = Param(initialize=optical_efficiency)

    # Operating condition parameters (scalar, not time-varying)
    model.m_dot = Param(initialize=mass_flow_rate)
    model.I = Param(initialize=irradiance)
    model.T_inlet = Param(initialize=T_inlet)

    # Fluid density: constant or temperature-dependent
    if constant_density:
        model._rho_f_const = Param(initialize=fluid_props.density(T_ref))

        @model.Expression(model.x)
        def rho_f(m, x):
            return m._rho_f_const
    else:

        @model.Expression(model.x)
        def rho_f(m, x):
            return m.fluid_props.density(m.T_f[x])

    # Fluid dynamic viscosity: constant or temperature-dependent
    if constant_viscosity:
        model._eta_f_const = Param(initialize=fluid_props.viscosity(T_ref))

        @model.Expression(model.x)
        def eta_f(m, x):
            return m._eta_f_const
    else:

        @model.Expression(model.x)
        def eta_f(m, x):
            return m.fluid_props.viscosity(m.T_f[x], exp=pyo_exp)

    # Fluid thermal conductivity: constant or temperature-dependent
    if constant_thermal_conductivity:
        model._k_f_const = Param(
            initialize=fluid_props.thermal_conductivity(T_ref)
        )

        @model.Expression(model.x)
        def k_f(m, x):
            return m._k_f_const
    else:

        @model.Expression(model.x)
        def k_f(m, x):
            return m.fluid_props.thermal_conductivity(m.T_f[x])

    # Fluid specific heat: constant or temperature-dependent
    if constant_specific_heat:
        model._cp_f_const = Param(initialize=fluid_props.heat_capacity(T_ref))

        @model.Expression(model.x)
        def cp_f(m, x):
            return m._cp_f_const
    else:

        @model.Expression(model.x)
        def cp_f(m, x):
            return m.fluid_props.heat_capacity(m.T_f[x])

    # Velocity expression from mass flow rate (scalar m_dot, x-dependent rho_f)
    @model.Expression(model.x)
    def v(m, x):
        return m.m_dot / (m.rho_f[x] * m.A)

    # Heat transfer coefficient: constant or computed from Dittus-Boelter
    if constant_heat_transfer_coeff:
        # Calculate h_int at reference temperature
        rho_ref = fluid_props.density(T_ref)
        eta_ref = fluid_props.viscosity(T_ref)
        k_ref = fluid_props.thermal_conductivity(T_ref)
        cp_ref = fluid_props.heat_capacity(T_ref)
        v_ref = mass_flow_rate / (rho_ref * np.pi * D**2 / 4.0)

        h_int_ref, _, _, _ = calculate_heat_transfer_coefficient_turbulent(
            v_ref, D, rho_ref, eta_ref, k_ref, cp_ref
        )

        model._h_int_const = Param(initialize=h_int_ref)

        @model.Expression(model.x)
        def h_int(m, x):
            return m._h_int_const
    else:
        # Temperature-dependent h_int via Dittus-Boelter
        @model.Expression(model.x)
        def h_int(m, x):
            rho = m.rho_f[x]
            eta = m.eta_f[x]
            k = m.k_f[x]
            cp = m.cp_f[x]
            local_v = m.v[x]
            D = m.D

            Re = rho * local_v * D / eta
            Pr = eta * cp / k
            Nu = 0.023 * Re**0.8 * Pr**0.4
            return Nu * k / D

    return model


def add_steady_state_constraints(model):
    """
    Add steady-state ODE constraints and boundary conditions.

    In steady state, the time derivatives are zero, so the PDEs become ODEs:
    - Fluid: ρ·cp·v·∂T_f/∂x = ρ·cp·D_ax·∂²T_f/∂x² + h_int·4·(T_p - T_f)/D
    - Wall: 0 = α_p·∂²T_p/∂x² + q_solar - q_to_fluid - q_to_ambient

    Parameters
    ----------
    model : pyomo.ConcreteModel
        The model created by create_collector_model_steady_state().

    Returns
    -------
    model : pyomo.ConcreteModel
        The model with ODE constraints and boundary conditions added.
    """

    # Fluid temperature ODE constraint (steady-state energy balance)
    @model.Constraint(model.x)
    def fluid_ode_constraint(m, x):
        if x == 0:  # Skip inlet (boundary condition applies there)
            return Constraint.Skip

        # Heat transfer from pipe wall to fluid per unit volume [W/m³]
        heat_to_fluid = m.h_int[x] * 4.0 * (m.T_p[x] - m.T_f[x]) / m.D

        # Steady-state fluid energy balance:
        # ρ·cp·v·∂T_f/∂x = ρ·cp·D_ax·∂²T_f/∂x² + heat_to_fluid
        rho_cp = m.rho_f[x] * m.cp_f[x]
        return (
            rho_cp * m.v[x] * m.dT_f_dx[x]
            == rho_cp * m.D_ax * m.d2T_f_dx2[x] + heat_to_fluid
        )

    # Pipe wall temperature ODE constraint (steady-state energy balance)
    @model.Constraint(model.x)
    def wall_ode_constraint(m, x):
        # Density * specific heat for pipe wall
        rho_cp = m.rho_p * m.cp_p

        # For nominal physical pipe (0 <= x <= L): include heat input
        if x <= m.L:
            # Solar heat input: q_eff = I * c * ε / 2 [W/m²]
            q_eff = m.I * m.c * m.epsilon / 2.0

            # Heat input to pipe wall per unit volume [W/m³]
            D_outer = m.D + 2 * m.d
            heat_input_volumetric = (
                q_eff * 4.0 * D_outer / (D_outer**2 - m.D**2)
            )

            # Heat transfer to fluid per unit volume [W/m³]
            heat_to_fluid = m.h_int[x] * 4.0 * (m.T_p[x] - m.T_f[x]) / m.D

            # Heat loss to ambient per unit volume [W/m³]
            heat_loss_volumetric = (
                m.h_ext
                * 4.0
                * D_outer
                * (m.T_p[x] - m.T_ambient)
                / (D_outer**2 - m.D**2)
            )

            # Pipe thermal diffusivity
            alpha_p = m.k_p / rho_cp

            # Steady-state wall energy balance (∂T_p/∂t = 0):
            # 0 = α_p·ρcp·∂²T_p/∂x² + q_in - q_to_fluid - q_loss
            return 0 == (
                rho_cp * alpha_p * m.d2T_p_dx2[x]
                + heat_input_volumetric
                - heat_to_fluid
                - heat_loss_volumetric
            )

        else:
            # For extended section (L < x <= L_extended): no heat input
            heat_to_fluid = m.h_int[x] * 4.0 * (m.T_p[x] - m.T_f[x]) / m.D
            D_outer = m.D + 2 * m.d
            heat_loss_volumetric = (
                m.h_ext
                * 4.0
                * D_outer
                * (m.T_p[x] - m.T_ambient)
                / (D_outer**2 - m.D**2)
            )
            alpha_p = m.k_p / rho_cp

            return 0 == (
                rho_cp * alpha_p * m.d2T_p_dx2[x]
                - heat_to_fluid
                - heat_loss_volumetric
            )

    # Inlet boundary condition for fluid temperature
    model.inlet_bc = Constraint(expr=model.T_f[0] == model.T_inlet)

    # Dummy objective (required by Pyomo)
    model.obj = Objective(expr=1)

    return model


def solve_steady_state_model(
    model, n_x=110, max_iter=1000, tol=1e-6, print_level=5, tee=True
):
    """
    Discretize and solve the steady-state model using finite differences.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        The model created by create_collector_model_steady_state() with
        constraints added by add_steady_state_constraints().
    n_x : int, default=110
        Number of finite elements for spatial discretization.
    max_iter : int, default=1000
        Maximum number of IPOPT solver iterations.
    tol : float, default=1e-6
        Solver tolerance for convergence.
    print_level : int, default=5
        IPOPT print level (0=no output, 5=detailed output).
    tee : bool, default=True
        Whether to display solver output to console.

    Returns
    -------
    results : pyomo solver results object
        Contains solver status, termination condition, and solution
        statistics.
    """
    # Apply finite difference discretization (CENTRAL for 2nd order accuracy)
    TransformationFactory("dae.finite_difference").apply_to(
        model, nfe=n_x, scheme="CENTRAL", wrt=model.x
    )

    # Outlet boundary conditions (zero gradient)
    x_vals = sorted(model.x)
    x_outlet = x_vals[-1]
    x_before = x_vals[-2]

    model.fluid_outlet_bc = Constraint(
        expr=model.T_f[x_outlet] == model.T_f[x_before]
    )
    model.wall_outlet_bc = Constraint(
        expr=model.T_p[x_outlet] == model.T_p[x_before]
    )

    print(f"Discretized with {len(model.x)} spatial points")

    # Provide good initial guess for temperature fields
    T_inlet_val = float(model.T_inlet.value)
    for x in model.x:
        model.T_f[x].set_value(T_inlet_val)
        model.T_p[x].set_value(T_inlet_val + 10.0)  # Slightly higher for wall

    # Configure solver
    solver = SolverFactory("ipopt")
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["print_level"] = print_level

    print("Solving steady-state model with IPOPT...")
    results = solver.solve(model, tee=tee)

    return results


def plot_steady_state_results(
    model,
    name="Steady-State Temperature Profiles",
    var_info=None,
    colors=None,
    figsize=(10, 6),
):
    """
    Plot steady-state temperature profiles for fluid and pipe wall.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        Solved steady-state model.
    name : str, default="Steady-State Temperature Profiles"
        Title for the plot.
    var_info : dict, optional
        Variable display information (labels, units).
    colors : dict, optional
        Color scheme for plots.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if var_info is None:
        var_info = VAR_INFO
    if colors is None:
        colors = PLOT_COLORS

    # Get spatial points and temperature values
    x_vals = np.array(sorted(model.x))
    T_f_vals = np.array([model.T_f[x].value for x in x_vals])
    T_p_vals = np.array([model.T_p[x].value for x in x_vals])

    # Convert to Celsius
    T_f_C = T_f_vals - ZERO_C
    T_p_C = T_p_vals - ZERO_C

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        x_vals, T_f_C, label="Fluid $T_f(x)$", color=colors.get("T_f", "blue")
    )
    ax.plot(
        x_vals,
        T_p_C,
        label="Pipe wall $T_p(x)$",
        color=colors.get("T_p", "red"),
        linestyle="--",
    )

    # Mark the collector length
    L_val = float(model.L.value)
    ax.axvline(
        L_val, color="gray", linestyle=":", alpha=0.7, label=f"L = {L_val} m"
    )

    ax.set_xlabel("Position $x$ [m]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add summary info as text
    T_inlet_C = float(model.T_inlet.value) - ZERO_C
    T_outlet_C = T_f_C[-1]
    delta_T = T_outlet_C - T_inlet_C
    info_text = (
        f"$T_{{inlet}}$ = {T_inlet_C:.1f}°C\n"
        f"$T_{{outlet}}$ = {T_outlet_C:.1f}°C\n"
        f"$\\Delta T$ = {delta_T:.1f}°C"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    return fig, ax


# =============================================================================
# General-Purpose Plotting Functions
# =============================================================================


def plot_time_series(
    t_vals,
    data_series,
    title="Time Series",
    colors=None,
    figsize=(8, 7.5),
):
    """
    Plot multiple time series on vertically stacked subplots.

    Parameters
    ----------
    t_vals : array-like
        Time values for x-axis.
    data_series : list of dict
        List of dictionaries, each containing:
        - 'y': array-like of y values (required)
        - 'ylabel': str, y-axis label (required)
        - 'label': str, legend label (optional, for multi-line plots)
        - 'title': str, subplot title (optional)
        - 'color': str, line color (optional)
        - 'lines': list of dict for multiple lines on same subplot,
          each with 'y', 'label', 'color' (optional)
    title : str, default="Time Series"
        Overall figure title (used as prefix for subplot titles).
    colors : dict, optional
        Color scheme dictionary.
    figsize : tuple, default=(8, 7.5)
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list of matplotlib.axes.Axes
        List of axes objects.

    Example
    -------
    >>> data = [
    ...     {'y': velocity, 'ylabel': 'Velocity [m/s]', 'title': 'Velocity'},
    ...     {'y': irradiance, 'ylabel': 'Irradiance [W/m²]'},
    ...     {'lines': [
    ...         {'y': T_inlet_f, 'label': 'Oil', 'color': 'blue'},
    ...         {'y': T_inlet_p, 'label': 'Wall', 'color': 'red'},
    ...     ], 'ylabel': 'Inlet Temp [°C]', 'title': 'Inlet Temperatures'},
    ... ]
    >>> fig, axes = plot_time_series(t, data, title="Model Results")
    """
    if colors is None:
        colors = PLOT_COLORS

    n_plots = len(data_series)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]

    for i, (ax, series) in enumerate(zip(axes, data_series)):
        subplot_title = series.get("title", "")
        if title and subplot_title:
            ax.set_title(f"{title} - {subplot_title}")
        elif subplot_title:
            ax.set_title(subplot_title)

        # Check if multiple lines on this subplot
        if "lines" in series:
            for line in series["lines"]:
                color = line.get("color", None)
                label = line.get("label", None)
                ax.plot(
                    t_vals, line["y"], color=color, linewidth=2, label=label
                )
            if any("label" in line for line in series["lines"]):
                ax.legend()
        else:
            color = series.get("color", None)
            label = series.get("label", None)
            ax.plot(t_vals, series["y"], color=color, linewidth=2, label=label)
            if label:
                ax.legend()

        ax.set_ylabel(series.get("ylabel", ""))
        ax.grid(True, alpha=0.3)

        # Only add x-label to bottom subplot
        if i == n_plots - 1:
            ax.set_xlabel("Time [s]")

    plt.tight_layout()
    return fig, axes


def plot_temperature_field(
    t_vals,
    x_vals,
    temp_vals,
    title="Temperature Field",
    temp_range=(0, 400),
    n_levels=21,
    cmap="viridis",
    collector_length=None,
    figsize=(8, 7.5),
):
    """
    Plot a 2D temperature field as a contour plot (time vs position).

    Parameters
    ----------
    t_vals : array-like
        Time values (shape: n_t).
    x_vals : array-like
        Position values (shape: n_x).
    temp_vals : array-like
        Temperature values in Celsius (shape: n_t x n_x).
    title : str, default="Temperature Field"
        Plot title.
    temp_range : tuple, default=(0, 400)
        Temperature range (min, max) for colorbar in °C.
    n_levels : int, default=21
        Number of contour levels.
    cmap : str, default="viridis"
        Colormap name.
    collector_length : float, optional
        If provided, draws a horizontal dashed line at this position
        to mark the collector end.
    figsize : tuple, default=(8, 7.5)
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    contour : matplotlib.contour.QuadContourSet
        The contour object (useful for additional customization).

    Example
    -------
    >>> fig, ax, contour = plot_temperature_field(
    ...     t_vals, x_vals, T_f_celsius,
    ...     title="Fluid Temperature",
    ...     collector_length=96.0,
    ... )
    """
    t_vals = np.asarray(t_vals)
    x_vals = np.asarray(x_vals)
    temp_vals = np.asarray(temp_vals)

    # Create meshgrid (time on x-axis, position on y-axis)
    T_grid, X_grid = np.meshgrid(t_vals, x_vals, indexing="ij")

    # Define contour levels
    temp_levels = np.linspace(temp_range[0], temp_range[1], n_levels)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    contour = ax.contourf(
        T_grid.T,
        X_grid.T,
        temp_vals.T,
        levels=temp_levels,
        cmap=cmap,
        extend="both",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title(title)

    # Mark collector end if specified
    if collector_length is not None:
        ax.axhline(
            y=collector_length,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="collector end",
        )
        ax.legend()

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Temperature [°C]")

    plt.tight_layout()
    return fig, ax, contour


def extract_model_data(model):
    """
    Extract time series and temperature field data from a solved dynamic model.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        Solved Pyomo model containing temperature solutions.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 't_vals': time values
        - 'x_vals': position values
        - 'T_f': fluid temperature array (n_t x n_x) in Kelvin
        - 'T_p': pipe wall temperature array (n_t x n_x) in Kelvin
        - 'v': velocity values at x=0
        - 'I': irradiance values
        - 'T_inlet': inlet temperature values
        - 'L': collector length
    """
    t_vals = np.array(sorted(model.t))
    x_vals = np.array(sorted(model.x))

    # Extract temperature fields
    T_f = np.array(
        [[pyo.environ.value(model.T_f[t, x]) for x in x_vals] for t in t_vals]
    )
    T_p = np.array(
        [[pyo.environ.value(model.T_p[t, x]) for x in x_vals] for t in t_vals]
    )

    # Extract time-varying inputs (at x=0 for velocity)
    v_vals = np.array(
        [pyo.environ.value(model.v[t, x_vals[0]]) for t in t_vals]
    )
    I_vals = np.array([pyo.environ.value(model.I[t]) for t in t_vals])
    T_inlet_vals = np.array(
        [pyo.environ.value(model.T_inlet[t]) for t in t_vals]
    )

    return {
        "t_vals": t_vals,
        "x_vals": x_vals,
        "T_f": T_f,
        "T_p": T_p,
        "v": v_vals,
        "I": I_vals,
        "T_inlet": T_inlet_vals,
        "L": float(pyo.environ.value(model.L)),
    }


def plot_results(
    model,
    t_eval=None,
    x_eval=None,
    name="Oil + Pipe Model",
    var_info=VAR_INFO,
    colors=PLOT_COLORS,
    figsize=(8, 7.5),
):
    """
    Plot temperature field solutions for both fluid and pipe wall temperatures.

    This is a convenience function that creates three figures:
    1. Time series plots (velocity, irradiance, inlet/outlet temperatures)
    2. Fluid temperature contour plot
    3. Pipe wall temperature contour plot

    Parameters
    ----------
    model : pyomo.ConcreteModel
        Solved Pyomo model containing temperature solutions.
    t_eval : array-like, optional
        Times to evaluate (deprecated, not used).
    x_eval : array-like, optional
        Positions to evaluate (deprecated, not used).
    name : str, default="Oil + Pipe Model"
        Name prefix for plot titles.
    var_info : dict, optional
        Variable display information.
    colors : dict, optional
        Color scheme dictionary.
    figsize : tuple, default=(8, 7.5)
        Figure size (width, height) in inches.

    Returns
    -------
    fig1 : matplotlib.figure.Figure
        Figure with 4 time series plots.
    fig2 : matplotlib.figure.Figure
        Figure with fluid temperature contour plot.
    fig3 : matplotlib.figure.Figure
        Figure with pipe wall temperature contour plot.
    """
    # Extract data from model
    data = extract_model_data(model)
    t_vals = data["t_vals"]
    x_vals = data["x_vals"]
    T_f = data["T_f"]
    T_p = data["T_p"]
    L = data["L"]

    # Find index closest to collector end (x = L) for outlet temperatures
    end_idx = np.argmin(np.abs(x_vals - L))

    # Prepare time series data
    time_series_data = [
        {
            "y": data["v"],
            "ylabel": "Velocity [m/s]",
            "title": "Fluid Velocity",
            "color": colors.get("v"),
        },
        {
            "y": data["I"],
            "ylabel": "Irradiance [W/m²]",
            "title": "Solar Irradiance (DNI)",
            "color": colors.get("q_solar_conc"),
        },
        {
            "lines": [
                {
                    "y": data["T_inlet"] - ZERO_C,
                    "label": "Oil",
                    "color": colors.get("T_f"),
                },
                {
                    "y": T_p[:, 0] - ZERO_C,
                    "label": "Wall",
                    "color": colors.get("T_p"),
                },
            ],
            "ylabel": "Inlet Temp [°C]",
            "title": "Collector Inlet Temperatures",
        },
        {
            "lines": [
                {
                    "y": T_f[:, end_idx] - ZERO_C,
                    "label": "Oil",
                    "color": colors.get("T_f"),
                },
                {
                    "y": T_p[:, end_idx] - ZERO_C,
                    "label": "Wall",
                    "color": colors.get("T_p"),
                },
            ],
            "ylabel": "Outlet Temp [°C]",
            "title": "Collector Outlet Temperatures",
        },
    ]

    # Create Figure 1: Time series
    fig1, _ = plot_time_series(
        t_vals, time_series_data, title=name, colors=colors, figsize=figsize
    )

    # Create Figure 2: Fluid temperature field
    fig2, _, _ = plot_temperature_field(
        t_vals,
        x_vals,
        T_f - ZERO_C,
        title=f"{name} - Oil Temperature Field",
        collector_length=L,
        figsize=figsize,
    )

    # Create Figure 3: Pipe wall temperature field
    fig3, _, _ = plot_temperature_field(
        t_vals,
        x_vals,
        T_p - ZERO_C,
        title=f"{name} - Pipe Wall Temperature Field",
        collector_length=L,
        figsize=figsize,
    )

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
