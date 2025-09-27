"""Solar collector thermal dynamics simulation using Pyomo DAE with two 
temperature variables: fluid and pipe wall.

This module implements a 1D partial differential equation (PDE) model for heat
transfer in a solar collector pipe using Pyomo's differential-algebraic
equation (DAE) framework. The model includes separate temperatures for the
fluid (T_f) and pipe wall (T_p) with heat transfer between them and to ambient.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo as pyo

from pyomo.dae import DerivativeVar, ContinuousSet
from pyomo.environ import (
    Var,
    Param,
    Constraint,
    Objective,
    SolverFactory,
    ConcreteModel,
    TransformationFactory,
)


# Constants
ZERO_C = 273.15  # K
THERMAL_DIFFUSIVITY = 0.25  # m²/s
FLUID_DENSITY = 800  # kg/m³
SPECIFIC_HEAT = 2000.0  # J/kg·K
HEAT_TRANSFER_COEFF = 20.0  # W/m²·K
PIPE_DIAMETER = 0.07  # m
PIPE_WALL_THICKNESS = 0.002  # m
COLLECTOR_LENGTH = 100.0  # m
PIPE_THERMAL_CONDUCTIVITY = 50.0  # W/m·K (typical for steel)
PIPE_DENSITY = 7850.0  # kg/m³ (steel)
PIPE_SPECIFIC_HEAT = 450.0  # J/kg·K (steel)


def create_pipe_flow_model(
    L=COLLECTOR_LENGTH,
    t_final=5.0,
    n_x=50,
    n_t=50,
    velocity_func=None,
    heat_func=None,
    inlet_func=None,
    thermal_diffusivity=THERMAL_DIFFUSIVITY,
    fluid_density=FLUID_DENSITY,
    specific_heat=SPECIFIC_HEAT,
    T_ambient=ZERO_C + 20.0,
    pipe_diameter=PIPE_DIAMETER,
    pipe_wall_thickness=PIPE_WALL_THICKNESS,
    heat_transfer_coeff=HEAT_TRANSFER_COEFF,
    pipe_thermal_conductivity=PIPE_THERMAL_CONDUCTIVITY,
    pipe_density=PIPE_DENSITY,
    pipe_specific_heat=PIPE_SPECIFIC_HEAT
):
    """
    Create Pyomo model for pipe flow heat transport PDE with fluid and wall

    Parameters:
    -----------
    L : float, default=100.0
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
    heat_func : callable, optional
        Function for time-varying heat input to pipe wall q(t) [W/m²]
        If None, uses default zero heat input
    inlet_func : callable, optional
        Function for time-varying inlet temperature T_inlet(t) [K]
        If None, uses default constant inlet temperature
    thermal_diffusivity : float, default=THERMAL_DIFFUSIVITY
        Fluid thermal diffusivity α [m²/s]
    fluid_density : float, default=FLUID_DENSITY
        Fluid density ρ_f [kg/m³]
    specific_heat : float, default=SPECIFIC_HEAT
        Fluid specific heat capacity cp_f [J/kg·K]
    T_ambient : float, default=ZERO_C + 20.0
        Ambient temperature [K] for convective heat loss
    pipe_diameter : float, default=0.07
        Inner pipe diameter D [m]
    pipe_wall_thickness : float, default=0.002
        Pipe wall thickness d [m]
    heat_transfer_coeff : float, default=20.0
        Convective heat transfer coefficient h [W/m²·K]
    pipe_thermal_conductivity : float, default=50.0
        Pipe wall thermal conductivity k_p [W/m·K]
    pipe_density : float, default=7850.0
        Pipe wall density ρ_p [kg/m³]
    pipe_specific_heat : float, default=450.0
        Pipe wall specific heat capacity cp_p [J/kg·K]

    Returns:
    --------
    model : pyomo.ConcreteModel
        Pyomo model with variables, parameters, and derivative variables
        defined.

    Notes:
    ------
    - Creates extended domain: 0 to L_extended = L * 1.1
    - Solar collector section: 0 < x <= L (with heat input to wall)
    - Buffer extension: L < x <= L_extended (no heat input)
    - Two temperature variables: T_f (fluid) and T_p (pipe wall)
    """

    model = ConcreteModel()

    # Extend domain by 10% beyond nominal length
    L_extended = L * 1.1
    model.x = ContinuousSet(bounds=(0, L_extended))
    model.t = ContinuousSet(bounds=(0, t_final))

    # Store both nominal and extended pipe lengths
    model.L = L  # Nominal length with heat input
    model.L_extended = L_extended  # Full domain length

    # Temperature variables with proper bounds
    model.T_f = Var(model.t, model.x, bounds=(0.0, None))  # Fluid temperature
    model.T_p = Var(model.t, model.x, bounds=(0.0, None))  # Pipe wall temperature

    # Derivative variables for fluid temperature
    model.dT_f_dt = DerivativeVar(model.T_f, wrt=model.t)
    model.dT_f_dx = DerivativeVar(model.T_f, wrt=model.x)
    model.d2T_f_dx2 = DerivativeVar(model.T_f, wrt=(model.x, model.x))

    # Derivative variables for pipe wall temperature
    model.dT_p_dt = DerivativeVar(model.T_p, wrt=model.t)
    model.dT_p_dx = DerivativeVar(model.T_p, wrt=model.x)
    model.d2T_p_dx2 = DerivativeVar(model.T_p, wrt=(model.x, model.x))

    # Fluid properties
    model.alpha_f = Param(initialize=thermal_diffusivity)  # [m²/s]
    model.rho_f = Param(initialize=fluid_density)          # [kg/m³]
    model.cp_f = Param(initialize=specific_heat)           # [J/kg·K]

    # Pipe properties
    model.rho_p = Param(initialize=pipe_density)           # [kg/m³]
    model.cp_p = Param(initialize=pipe_specific_heat)      # [J/kg·K]
    model.k_p = Param(initialize=pipe_thermal_conductivity) # [W/m·K]

    # Geometric parameters
    model.T_ambient = Param(initialize=T_ambient)          # [K]
    model.D = Param(initialize=pipe_diameter)              # [m]
    model.d = Param(initialize=pipe_wall_thickness)        # [m]
    model.h = Param(initialize=heat_transfer_coeff)        # [W/m²·K]

    # Default parameter functions if none provided
    if velocity_func is None:
        def velocity_func(t):
            return 0.2  # velocity [m/s]

    if heat_func is None:
        def heat_func(t):
            if t > 60.0 and t <= 240.0:
                return 1000.0  # heat flux to pipe wall [W/m²]
            return 0.0

    if inlet_func is None:
        def inlet_func(t):
            return ZERO_C + 270.0

    # Store functions for later use
    model.velocity_func = velocity_func
    model.heat_func = heat_func
    model.inlet_func = inlet_func

    # Time-varying parameters (will be initialized after discretization)
    model.v = Param(model.t, mutable=True)
    model.q = Param(model.t, mutable=True)
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
        # Heat transfer per unit length: h * π * D * (T_p - T_f) [W/m]
        # Heat transfer per unit volume: h * π * D * (T_p - T_f) / (π * D² / 4)
        # Simplified: h * 4 * (T_p - T_f) / D [W/m³]
        heat_from_wall = m.h * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D

        # Fluid PDE: ∂T_f/∂t + v(t)∂T_f/∂x = α_f∂²T_f/∂x² + q_wall_to_fluid/(ρ_f*cp_f)
        return (
            m.dT_f_dt[t, x] + m.v[t] * m.dT_f_dx[t, x] ==
            m.alpha_f * m.d2T_f_dx2[t, x] + heat_from_wall / (m.rho_f * m.cp_f)
        )

    # Pipe wall temperature PDE constraint
    @model.Constraint(model.t, model.x)
    def wall_pde_constraint(m, t, x):
        # Skip initial time
        if t == 0:
            return Constraint.Skip

        # For nominal physical pipe (0 < x <= L): include heat input
        if x <= m.L:
            # Heat input to pipe wall per unit volume [W/m³]
            # Heat flux q(t) [W/m²] applied to outer surface
            # Outer diameter: D + 2*d
            # Heat input per unit length: q(t) * π * (D + 2*d) [W/m]
            # Pipe wall volume per unit length: π * ((D+2*d)² - D²) / 4 [m³/m]
            # Heat input per unit volume: q(t) * π * (D + 2*d) / (π * ((D+2*d)² - D²) / 4)
            # Simplified: q(t) * 4 * (D + 2*d) / ((D+2*d)² - D²) [W/m³]
            D_outer = m.D + 2 * m.d
            heat_input_volumetric = m.q[t] * 4.0 * D_outer / (D_outer**2 - m.D**2)

            # Heat transfer to fluid per unit volume [W/m³]
            heat_to_fluid = m.h * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D

            # Heat loss to ambient per unit volume [W/m³]
            # Heat loss per unit length: h * π * (D + 2*d) * (T_p - T_ambient) [W/m]
            # Heat loss per unit volume: h * π * (D + 2*d) * (T_p - T_ambient) / wall_volume_per_length
            heat_loss_volumetric = m.h * 4.0 * D_outer * (m.T_p[t, x] - m.T_ambient) / (D_outer**2 - m.D**2)

            # Pipe thermal diffusivity: k_p / (ρ_p * cp_p)
            alpha_p = m.k_p / (m.rho_p * m.cp_p)

            # Wall PDE: ∂T_p/∂t = α_p∂²T_p/∂x² + (q_input - q_to_fluid - q_to_ambient)/(ρ_p*cp_p)
            return (
                m.dT_p_dt[t, x] ==
                alpha_p * m.d2T_p_dx2[t, x] +
                (heat_input_volumetric - heat_to_fluid - heat_loss_volumetric) / (m.rho_p * m.cp_p)
            )
        else:
            # For extended section (L < x <= L_extended): no heat input
            heat_to_fluid = m.h * 4.0 * (m.T_p[t, x] - m.T_f[t, x]) / m.D
            D_outer = m.D + 2 * m.d
            heat_loss_volumetric = m.h * 4.0 * D_outer * (m.T_p[t, x] - m.T_ambient) / (D_outer**2 - m.D**2)
            alpha_p = m.k_p / (m.rho_p * m.cp_p)

            return (
                m.dT_p_dt[t, x] ==
                alpha_p * m.d2T_p_dx2[t, x] -
                (heat_to_fluid + heat_loss_volumetric) / (m.rho_p * m.cp_p)
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
        T_0 = ZERO_C + 270.0  # initial wall temperature
        return m.T_p[0, x] == T_0

    # Inlet boundary condition for fluid
    @model.Constraint(model.t)
    def inlet_bc(m, t):
        return m.T_f[t, 0] == m.T_inlet[t]

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
    TransformationFactory('dae.finite_difference').apply_to(
        model, nfe=n_x, scheme='CENTRAL', wrt=model.x
    )

    # Temporal discretization (backward Euler for stability)
    TransformationFactory('dae.finite_difference').apply_to(
        model, nfe=n_t, scheme='BACKWARD', wrt=model.t
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
        model.v[t].set_value(model.velocity_func(t))
        model.q[t].set_value(model.heat_func(t))
        model.T_inlet[t].set_value(model.inlet_func(t))

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
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = max_iter
    solver.options['tol'] = tol
    solver.options['print_level'] = print_level

    print("Solving with IPOPT...")
    results = solver.solve(model, tee=tee)

    return results