"""
Heat transfer correlations for solar collector models.

This module provides empirical correlations for calculating convective heat
transfer coefficients in pipe flow, particularly the Dittus-Boelter correlation
for internal forced convection.
"""

# Default fluid properties for thermal oil
DEFAULT_FLUID_DENSITY = 800.0  # kg/m³
DEFAULT_FLUID_SPECIFIC_HEAT = 2000.0  # J/kg·K
DEFAULT_FLUID_THERMAL_CONDUCTIVITY = 0.12  # W/m·K (typical thermal oil)
DEFAULT_FLUID_DYNAMIC_VISCOSITY = 0.01  # Pa·s (typical thermal oil)


def calculate_heat_transfer_coefficient(
    velocity,
    pipe_diameter,
    fluid_density=DEFAULT_FLUID_DENSITY,
    fluid_viscosity=DEFAULT_FLUID_DYNAMIC_VISCOSITY,
    fluid_thermal_conductivity=DEFAULT_FLUID_THERMAL_CONDUCTIVITY,
    fluid_specific_heat=DEFAULT_FLUID_SPECIFIC_HEAT,
):
    """
    Calculate internal heat transfer coefficient using Dittus-Boelter correlation

    The Dittus-Boelter correlation is widely used for turbulent flow in smooth
    pipes with moderate temperature differences. It provides good accuracy
    (±25%) for most engineering applications.

    Parameters:
    -----------
    velocity : float
        Fluid velocity [m/s]
    pipe_diameter : float
        Pipe inner diameter [m]
    fluid_density : float, default=800.0
        Fluid density [kg/m³]
    fluid_viscosity : float, default=0.01
        Dynamic viscosity [Pa·s]
    fluid_thermal_conductivity : float, default=0.12
        Thermal conductivity [W/m·K]
    fluid_specific_heat : float, default=2000.0
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
    - Laminar flow (Re ≤ 4000): Nu = 4.36 (constant for uniform heat flux)

    Valid range:
    - 0.7 ≤ Pr ≤ 160
    - Re > 10,000 (but works reasonably well down to Re ≈ 4000)
    - L/D > 10 (fully developed flow)
    """
    # Reynolds number
    Re = fluid_density * velocity * pipe_diameter / fluid_viscosity

    # Prandtl number
    Pr = fluid_viscosity * fluid_specific_heat / fluid_thermal_conductivity

    # Nusselt number (Dittus-Boelter correlation)
    if Re > 4000:  # Turbulent flow
        Nu = 0.023 * Re**0.8 * Pr**0.4
    else:  # Laminar flow
        Nu = 4.36  # Constant Nu for uniform heat flux

    # Heat transfer coefficient
    h = Nu * fluid_thermal_conductivity / pipe_diameter

    return h, Re, Pr, Nu


def calculate_reynolds_number(velocity, pipe_diameter, fluid_density, fluid_viscosity):
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


def calculate_prandtl_number(fluid_viscosity, fluid_specific_heat, fluid_thermal_conductivity):
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

    Parameters:
    -----------
    reynolds_number : float
        Reynolds number [-]

    Returns:
    --------
    regime : str
        Flow regime: 'laminar', 'transitional', or 'turbulent'
    """
    if reynolds_number < 2300:
        return 'laminar'
    elif reynolds_number < 4000:
        return 'transitional'
    else:
        return 'turbulent'


def estimate_pressure_drop(velocity, pipe_length, pipe_diameter, fluid_density, fluid_viscosity):
    """
    Estimate pressure drop using Darcy-Weisbach equation

    Parameters:
    -----------
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

    Returns:
    --------
    pressure_drop : float
        Pressure drop [Pa]
    friction_factor : float
        Darcy friction factor [-]
    """
    Re = calculate_reynolds_number(velocity, pipe_diameter, fluid_density, fluid_viscosity)

    # Friction factor correlation
    if Re < 2300:  # Laminar flow
        f = 64 / Re
    else:  # Turbulent flow (Blasius correlation for smooth pipes)
        f = 0.316 / Re**0.25

    # Darcy-Weisbach equation
    pressure_drop = f * (pipe_length / pipe_diameter) * (fluid_density * velocity**2 / 2)

    return pressure_drop, f