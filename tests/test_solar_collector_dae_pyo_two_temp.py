"""Tests for solar_collector_dae_pyo_two_temp module.

Tests for model creation and constraint setup with both constant
and temperature-dependent fluid properties, including steady-state
model validation.
"""

import numpy as np
import pytest
from pyomo.dae import ContinuousSet
from pyomo.environ import value

from solar_collector.fluid_properties import SYLTHERM800
from solar_collector.solar_collector_dae_pyo_two_temp import (
    AXIAL_DISPERSION_COEFF,
    COLLECTOR_LENGTH,
    PIPE_DIAMETER,
    PIPE_WALL_THICKNESS,
    ZERO_C,
    add_pde_constraints,
    add_steady_state_constraints,
    compute_steady_state_initial_conditions,
    create_collector_model,
    create_collector_model_steady_state,
    get_final_temperatures,
    run_simulation,
    solve_model,
    solve_steady_state_model,
)


# =============================================================================
# Operating Conditions for Test Cases
# =============================================================================
# Dictionary of operating conditions for parameterized tests.
# Each case includes: T_inlet [°C], DNI [W/m²], T_ambient [°C],
# volumetric_flow_rate [L/min or m³/s - to be converted to mass flow]
#
# Note: Mass flow rate = volumetric_flow_rate * density
# For SYLTHERM800 at ~300°C: density ≈ 671 kg/m³

OPERATING_CONDITIONS = {
    # Case: Zero irradiance, inlet warmer than ambient
    # Expected: Heat loss to ambient, temperature decreases along pipe
    "zero_irradiance_warm_inlet": {
        "T_inlet_C": 300.0,  # °C
        "DNI": 0.0,  # W/m²
        "T_ambient_C": 25.0,  # °C
        "mass_flow_rate": 7.5,  # kg/s
        "description": "No solar input, inlet warmer than ambient",
    },
    # Case: Zero irradiance, thermal equilibrium
    # Expected: No temperature change
    "zero_irradiance_equilibrium": {
        "T_inlet_C": 300.0,  # °C
        "DNI": 0.0,  # W/m²
        "T_ambient_C": 300.0,  # °C (same as inlet)
        "mass_flow_rate": 7.5,  # kg/s
        "description": "No solar input, thermal equilibrium",
    },
    # Placeholder for researcher-provided conditions
    # "case_1": {
    #     "T_inlet_C": ...,
    #     "DNI": ...,
    #     "T_ambient_C": ...,
    #     "volumetric_flow_rate_Lmin": ...,  # L/min
    #     "description": "...",
    # },
}


@pytest.fixture
def fluid_props():
    """Create SYLTHERM800 fluid properties object."""
    return SYLTHERM800()


class TestCreatePipeFlowModelConstantProperties:
    """Tests for create_pipe_flow_model with constant fluid properties."""

    @pytest.fixture
    def model_constant(self, fluid_props):
        """Create model with constant fluid properties."""
        return create_collector_model(
            fluid_props,
            L=COLLECTOR_LENGTH,
            t_final=10.0,
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )

    def test_continuous_sets_exist(self, model_constant):
        """Test that spatial and temporal continuous sets are created."""
        assert hasattr(model_constant, "x")
        assert hasattr(model_constant, "t")
        assert isinstance(model_constant.x, ContinuousSet)
        assert isinstance(model_constant.t, ContinuousSet)

    def test_continuous_set_bounds(self, model_constant):
        """Test continuous set bounds are correct."""
        # Spatial domain: 0 to L_extended (L * 1.1)
        x_bounds = model_constant.x.bounds()
        assert x_bounds[0] == 0.0
        assert x_bounds[1] == pytest.approx(COLLECTOR_LENGTH * 1.1)

        # Temporal domain: 0 to t_final
        t_bounds = model_constant.t.bounds()
        assert t_bounds[0] == 0.0
        assert t_bounds[1] == 10.0

    def test_temperature_variables_exist(self, model_constant):
        """Test that fluid and wall temperature variables are created."""
        assert hasattr(model_constant, "T_f")
        assert hasattr(model_constant, "T_p")

    def test_derivative_variables_exist(self, model_constant):
        """Test that derivative variables are created."""
        # Fluid temperature derivatives
        assert hasattr(model_constant, "dT_f_dt")
        assert hasattr(model_constant, "dT_f_dx")
        assert hasattr(model_constant, "d2T_f_dx2")

        # Pipe wall temperature derivatives
        assert hasattr(model_constant, "dT_p_dt")
        assert hasattr(model_constant, "dT_p_dx")
        assert hasattr(model_constant, "d2T_p_dx2")

    def test_geometric_parameters(self, model_constant):
        """Test geometric parameters have correct values."""
        assert value(model_constant.D) == pytest.approx(PIPE_DIAMETER)
        assert value(model_constant.d) == pytest.approx(PIPE_WALL_THICKNESS)
        assert value(model_constant.L) == pytest.approx(COLLECTOR_LENGTH)

        # Cross-sectional area
        expected_A = np.pi * PIPE_DIAMETER**2 / 4.0
        assert value(model_constant.A) == pytest.approx(expected_A)

    def test_pipe_properties(self, model_constant):
        """Test pipe material property parameters exist."""
        assert hasattr(model_constant, "rho_p")
        assert hasattr(model_constant, "cp_p")
        assert hasattr(model_constant, "k_p")
        assert hasattr(model_constant, "h_ext")
        assert hasattr(model_constant, "T_ambient")

    def test_axial_dispersion_coefficient(self, model_constant):
        """Test axial dispersion coefficient parameter."""
        assert hasattr(model_constant, "D_ax")
        assert value(model_constant.D_ax) == pytest.approx(
            AXIAL_DISPERSION_COEFF
        )

    def test_fluid_property_expressions_exist(self, model_constant):
        """Test fluid property expressions are created."""
        assert hasattr(model_constant, "rho_f")
        assert hasattr(model_constant, "eta_f")
        assert hasattr(model_constant, "k_f")
        assert hasattr(model_constant, "cp_f")

    def test_constant_fluid_property_values(self, model_constant, fluid_props):
        """Test constant fluid property values match expected at T_ref."""
        T_ref = model_constant.T_ref

        # Get expected values from fluid properties at T_ref
        expected_rho = fluid_props.density(T_ref)
        expected_eta = fluid_props.viscosity(T_ref)
        expected_lam = fluid_props.thermal_conductivity(T_ref)
        expected_cp = fluid_props.heat_capacity(T_ref)

        # Check constant parameter values
        assert value(model_constant._rho_f_const) == pytest.approx(
            expected_rho
        )
        assert value(model_constant._eta_f_const) == pytest.approx(
            expected_eta
        )
        assert value(model_constant._k_f_const) == pytest.approx(expected_lam)
        assert value(model_constant._cp_f_const) == pytest.approx(expected_cp)

    def test_velocity_expression_exists(self, model_constant):
        """Test velocity expression is created."""
        assert hasattr(model_constant, "v")

    def test_heat_transfer_coefficient_expression_exists(self, model_constant):
        """Test internal heat transfer coefficient expression is created."""
        assert hasattr(model_constant, "h_int")

    def test_constant_heat_transfer_coefficient_value(self, model_constant):
        """Test constant h_int value is positive and reasonable."""
        # h_int should be calculated from Dittus-Boelter at T_ref
        h_int_val = value(model_constant._h_int_const)

        # Typical values for turbulent oil flow: 100-2000 W/(m²·K)
        assert h_int_val > 50.0
        assert h_int_val < 5000.0

    def test_time_varying_parameters_exist(self, model_constant):
        """Test time-varying input parameters are created."""
        assert hasattr(model_constant, "m_dot")
        assert hasattr(model_constant, "I")
        assert hasattr(model_constant, "T_inlet")

    def test_solar_collector_parameters(self, model_constant):
        """Test solar collector parameters exist."""
        assert hasattr(model_constant, "c")  # concentration factor
        assert hasattr(model_constant, "epsilon")  # optical efficiency

    def test_constant_property_flags_stored(self, model_constant):
        """Test that constant property flags are stored on model."""
        assert model_constant.constant_density is True
        assert model_constant.constant_viscosity is True
        assert model_constant.constant_thermal_conductivity is True
        assert model_constant.constant_specific_heat is True
        assert model_constant.constant_heat_transfer_coeff is True


class TestCreatePipeFlowModelTemperatureDependent:
    """Tests for create_pipe_flow_model with temperature-dependent properties."""

    @pytest.fixture
    def model_temp_dependent(self, fluid_props):
        """Create model with temperature-dependent fluid properties."""
        return create_collector_model(
            fluid_props,
            L=COLLECTOR_LENGTH,
            t_final=10.0,
            constant_density=False,
            constant_viscosity=False,
            constant_thermal_conductivity=False,
            constant_specific_heat=False,
            constant_heat_transfer_coeff=False,
        )

    def test_continuous_sets_exist(self, model_temp_dependent):
        """Test that spatial and temporal continuous sets are created."""
        assert hasattr(model_temp_dependent, "x")
        assert hasattr(model_temp_dependent, "t")

    def test_temperature_variables_exist(self, model_temp_dependent):
        """Test that fluid and wall temperature variables are created."""
        assert hasattr(model_temp_dependent, "T_f")
        assert hasattr(model_temp_dependent, "T_p")

    def test_fluid_property_expressions_exist(self, model_temp_dependent):
        """Test fluid property expressions are created for temp-dependent case."""
        assert hasattr(model_temp_dependent, "rho_f")
        assert hasattr(model_temp_dependent, "eta_f")
        assert hasattr(model_temp_dependent, "k_f")
        assert hasattr(model_temp_dependent, "cp_f")

    def test_velocity_expression_exists(self, model_temp_dependent):
        """Test velocity expression is created."""
        assert hasattr(model_temp_dependent, "v")

    def test_heat_transfer_coefficient_expression_exists(
        self, model_temp_dependent
    ):
        """Test internal heat transfer coefficient expression is created."""
        assert hasattr(model_temp_dependent, "h_int")

    def test_constant_property_flags_stored(self, model_temp_dependent):
        """Test that constant property flags are stored correctly."""
        assert model_temp_dependent.constant_density is False
        assert model_temp_dependent.constant_viscosity is False
        assert model_temp_dependent.constant_thermal_conductivity is False
        assert model_temp_dependent.constant_specific_heat is False
        assert model_temp_dependent.constant_heat_transfer_coeff is False

    def test_no_constant_params_for_temp_dependent(self, model_temp_dependent):
        """Test that constant parameter attributes don't exist when temp-dependent."""
        # These private attributes should not exist when properties are temp-dependent
        assert not hasattr(model_temp_dependent, "_rho_f_const")
        assert not hasattr(model_temp_dependent, "_eta_f_const")
        assert not hasattr(model_temp_dependent, "_k_f_const")
        assert not hasattr(model_temp_dependent, "_cp_f_const")
        assert not hasattr(model_temp_dependent, "_h_int_const")


class TestAddPdeConstraintsConstantProperties:
    """Tests for add_pde_constraints with constant fluid properties."""

    @pytest.fixture
    def model_with_constraints(self, fluid_props):
        """Create model with constraints added (constant properties)."""
        model = create_collector_model(
            fluid_props,
            L=COLLECTOR_LENGTH,
            t_final=10.0,
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )
        return add_pde_constraints(model)

    def test_fluid_pde_constraint_exists(self, model_with_constraints):
        """Test that fluid PDE constraint is added."""
        assert hasattr(model_with_constraints, "fluid_pde_constraint")

    def test_wall_pde_constraint_exists(self, model_with_constraints):
        """Test that wall PDE constraint is added."""
        assert hasattr(model_with_constraints, "wall_pde_constraint")

    def test_fluid_initial_condition_exists(self, model_with_constraints):
        """Test that fluid initial condition constraint is added."""
        assert hasattr(model_with_constraints, "fluid_initial_condition")

    def test_wall_initial_condition_exists(self, model_with_constraints):
        """Test that wall initial condition constraint is added."""
        assert hasattr(model_with_constraints, "wall_initial_condition")

    def test_inlet_boundary_condition_exists(self, model_with_constraints):
        """Test that inlet boundary condition is added."""
        assert hasattr(model_with_constraints, "inlet_bc")

    def test_objective_exists(self, model_with_constraints):
        """Test that objective function is added."""
        assert hasattr(model_with_constraints, "obj")


class TestAddPdeConstraintsTemperatureDependent:
    """Tests for add_pde_constraints with temperature-dependent properties."""

    @pytest.fixture
    def model_with_constraints_temp_dep(self, fluid_props):
        """Create model with constraints added (temp-dependent properties)."""
        model = create_collector_model(
            fluid_props,
            L=COLLECTOR_LENGTH,
            t_final=10.0,
            constant_density=False,
            constant_viscosity=False,
            constant_thermal_conductivity=False,
            constant_specific_heat=False,
            constant_heat_transfer_coeff=False,
        )
        return add_pde_constraints(model)

    def test_fluid_pde_constraint_exists(
        self, model_with_constraints_temp_dep
    ):
        """Test that fluid PDE constraint is added."""
        assert hasattr(model_with_constraints_temp_dep, "fluid_pde_constraint")

    def test_wall_pde_constraint_exists(self, model_with_constraints_temp_dep):
        """Test that wall PDE constraint is added."""
        assert hasattr(model_with_constraints_temp_dep, "wall_pde_constraint")

    def test_fluid_initial_condition_exists(
        self, model_with_constraints_temp_dep
    ):
        """Test that fluid initial condition constraint is added."""
        assert hasattr(
            model_with_constraints_temp_dep, "fluid_initial_condition"
        )

    def test_wall_initial_condition_exists(
        self, model_with_constraints_temp_dep
    ):
        """Test that wall initial condition constraint is added."""
        assert hasattr(
            model_with_constraints_temp_dep, "wall_initial_condition"
        )

    def test_inlet_boundary_condition_exists(
        self, model_with_constraints_temp_dep
    ):
        """Test that inlet boundary condition is added."""
        assert hasattr(model_with_constraints_temp_dep, "inlet_bc")

    def test_objective_exists(self, model_with_constraints_temp_dep):
        """Test that objective function is added."""
        assert hasattr(model_with_constraints_temp_dep, "obj")


class TestSolveModelZeroIrradiance:
    """Tests for solve_model with zero solar irradiance.

    With I(t) = 0 and uniform initial conditions matching T_inlet = T_amb,
    the system should maintain thermal equilibrium.
    """

    # Parameterize with different simulation durations
    @pytest.fixture(params=[10.0, 30.0, 60.0])
    def t_final(self, request):
        """Parameterized simulation duration."""
        return request.param

    @pytest.fixture
    def T_uniform(self):
        """Uniform temperature for initial conditions, inlet, and ambient."""
        return ZERO_C + 300.0  # 300°C

    @pytest.fixture
    def solved_model(self, fluid_props, t_final, T_uniform):
        """Create and solve model with zero irradiance and uniform temps."""

        # Define input functions for zero irradiance case
        def zero_irradiance(t):
            return 0.0

        def constant_inlet_temp(t):
            return T_uniform

        # Create model
        model = create_collector_model(
            fluid_props,
            L=COLLECTOR_LENGTH,
            t_final=t_final,
            irradiance_func=zero_irradiance,
            T_inlet_func=constant_inlet_temp,
            T_ambient=T_uniform,
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )

        # Add constraints with uniform initial conditions
        model = add_pde_constraints(
            model,
            T_f_initial=T_uniform,
            T_p_initial=T_uniform,
        )

        # Solve with reduced output
        results = solve_model(
            model,
            n_x=20,
            n_t=10,
            print_level=0,
            tee=False,
        )

        return model, results

    def test_solver_converges(self, solved_model):
        """Test that the solver converges to optimal solution."""
        model, results = solved_model
        assert results.solver.termination_condition.name in [
            "optimal",
            "locallyOptimal",
        ]

    def test_fluid_temperature_uniform_at_final_time(
        self, solved_model, T_uniform
    ):
        """Test fluid temperature is uniform along pipe at final time."""
        model, _ = solved_model
        t_vals = sorted(model.t)
        x_vals = sorted(model.x)
        t_final = t_vals[-1]

        # Get fluid temperatures at final time
        T_f_final = [value(model.T_f[t_final, x]) for x in x_vals]

        # Check all temperatures are close to uniform value
        for T_f in T_f_final:
            assert T_f == pytest.approx(T_uniform, rel=0.01)

    def test_wall_temperature_uniform_at_final_time(
        self, solved_model, T_uniform
    ):
        """Test wall temperature is uniform along pipe at final time."""
        model, _ = solved_model
        t_vals = sorted(model.t)
        x_vals = sorted(model.x)
        t_final = t_vals[-1]

        # Get wall temperatures at final time
        T_p_final = [value(model.T_p[t_final, x]) for x in x_vals]

        # Check all temperatures are close to uniform value
        for T_p in T_p_final:
            assert T_p == pytest.approx(T_uniform, rel=0.01)

    def test_fluid_temperature_constant_in_space(self, solved_model):
        """Test fluid temperature is constant along pipe at all times."""
        model, _ = solved_model
        t_vals = sorted(model.t)
        x_vals = sorted(model.x)

        for t in t_vals[1:]:  # Skip initial time
            T_f_at_t = [value(model.T_f[t, x]) for x in x_vals]
            T_f_mean = np.mean(T_f_at_t)
            T_f_std = np.std(T_f_at_t)

            # Standard deviation should be small relative to mean
            assert T_f_std / T_f_mean < 0.01

    def test_wall_temperature_constant_in_space(self, solved_model):
        """Test wall temperature is constant along pipe at all times."""
        model, _ = solved_model
        t_vals = sorted(model.t)
        x_vals = sorted(model.x)

        for t in t_vals[1:]:  # Skip initial time
            T_p_at_t = [value(model.T_p[t, x]) for x in x_vals]
            T_p_mean = np.mean(T_p_at_t)
            T_p_std = np.std(T_p_at_t)

            # Standard deviation should be small relative to mean
            assert T_p_std / T_p_mean < 0.01

    def test_fluid_wall_temperature_equilibrium(self, solved_model, T_uniform):
        """Test fluid and wall temperatures are in equilibrium."""
        model, _ = solved_model
        t_vals = sorted(model.t)
        x_vals = sorted(model.x)
        t_final = t_vals[-1]

        # At equilibrium, T_f ≈ T_p ≈ T_uniform at all positions
        for x in x_vals:
            T_f = value(model.T_f[t_final, x])
            T_p = value(model.T_p[t_final, x])
            assert T_f == pytest.approx(T_p, rel=0.01)
            assert T_f == pytest.approx(T_uniform, rel=0.01)

    def test_fluid_properties_constant_and_correct(
        self, solved_model, fluid_props
    ):
        """Test fluid properties are constant and match expected values at T_ref."""
        model, _ = solved_model
        t_vals = sorted(model.t)
        x_vals = sorted(model.x)
        T_ref = model.T_ref

        # Expected values from fluid properties at reference temperature
        expected_rho = fluid_props.density(T_ref)
        expected_eta = fluid_props.viscosity(T_ref)
        expected_k = fluid_props.thermal_conductivity(T_ref)
        expected_cp = fluid_props.heat_capacity(T_ref)

        # Check properties at multiple (t, x) points
        for t in t_vals[::3]:  # Sample every 3rd time point
            for x in x_vals[::3]:  # Sample every 3rd spatial point
                assert value(model.rho_f[t, x]) == pytest.approx(expected_rho)
                assert value(model.eta_f[t, x]) == pytest.approx(expected_eta)
                assert value(model.k_f[t, x]) == pytest.approx(expected_k)
                assert value(model.cp_f[t, x]) == pytest.approx(expected_cp)


# =============================================================================
# Steady-State Model Tests
# =============================================================================


class TestSteadyStateModelCreation:
    """Tests for steady-state model creation."""

    @pytest.fixture
    def steady_state_model(self, fluid_props):
        """Create a steady-state model with constant properties."""
        return create_collector_model_steady_state(
            fluid_props,
            L=COLLECTOR_LENGTH,
            mass_flow_rate=7.5,
            irradiance=800.0,
            T_inlet=ZERO_C + 200.0,
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )

    def test_spatial_set_exists(self, steady_state_model):
        """Test that spatial continuous set is created."""
        assert hasattr(steady_state_model, "x")
        assert isinstance(steady_state_model.x, ContinuousSet)

    def test_no_time_set(self, steady_state_model):
        """Test that no time set exists (steady-state)."""
        assert not hasattr(steady_state_model, "t")

    def test_temperature_variables_exist(self, steady_state_model):
        """Test that temperature variables are created."""
        assert hasattr(steady_state_model, "T_f")
        assert hasattr(steady_state_model, "T_p")

    def test_spatial_derivatives_exist(self, steady_state_model):
        """Test that spatial derivative variables are created."""
        assert hasattr(steady_state_model, "dT_f_dx")
        assert hasattr(steady_state_model, "d2T_f_dx2")
        assert hasattr(steady_state_model, "dT_p_dx")
        assert hasattr(steady_state_model, "d2T_p_dx2")

    def test_no_time_derivatives(self, steady_state_model):
        """Test that time derivatives don't exist (steady-state)."""
        assert not hasattr(steady_state_model, "dT_f_dt")
        assert not hasattr(steady_state_model, "dT_p_dt")

    def test_scalar_inputs(self, steady_state_model):
        """Test that inputs are scalar parameters (not time-indexed)."""
        # These should be Params, not indexed by time
        assert hasattr(steady_state_model, "m_dot")
        assert hasattr(steady_state_model, "I")
        assert hasattr(steady_state_model, "T_inlet")
        # Check they have scalar values
        assert value(steady_state_model.m_dot) == pytest.approx(7.5)
        assert value(steady_state_model.I) == pytest.approx(800.0)
        assert value(steady_state_model.T_inlet) == pytest.approx(ZERO_C + 200.0)


class TestSteadyStateSolve:
    """Tests for solving the steady-state model."""

    @pytest.fixture
    def solved_steady_state(self, fluid_props):
        """Create and solve a steady-state model."""
        model = create_collector_model_steady_state(
            fluid_props,
            L=COLLECTOR_LENGTH,
            mass_flow_rate=7.5,
            irradiance=800.0,
            T_inlet=ZERO_C + 200.0,
            T_ambient=ZERO_C + 25.0,
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )
        model = add_steady_state_constraints(model)
        results = solve_steady_state_model(
            model, n_x=50, print_level=0, tee=False
        )
        return model, results

    def test_solver_converges(self, solved_steady_state):
        """Test that the solver converges."""
        model, results = solved_steady_state
        assert results.solver.termination_condition.name in [
            "optimal",
            "locallyOptimal",
        ]

    def test_temperature_increases_with_irradiance(self, solved_steady_state):
        """Test that fluid temperature increases along pipe with solar input."""
        model, _ = solved_steady_state
        x_vals = sorted(model.x)

        T_inlet = value(model.T_f[x_vals[0]])
        T_outlet = value(model.T_f[x_vals[-1]])

        # With positive irradiance, outlet should be warmer than inlet
        assert T_outlet > T_inlet

    def test_wall_hotter_than_fluid(self, solved_steady_state):
        """Test that wall is hotter than fluid (heat flows wall -> fluid)."""
        model, _ = solved_steady_state
        x_vals = sorted(model.x)

        # Check at middle of collector (within heated section)
        x_mid = x_vals[len(x_vals) // 2]
        if x_mid <= float(model.L.value):
            T_f_mid = value(model.T_f[x_mid])
            T_p_mid = value(model.T_p[x_mid])
            assert T_p_mid > T_f_mid


class TestSteadyStateZeroIrradiance:
    """Tests for steady-state with zero irradiance."""

    @pytest.fixture
    def zero_irradiance_equilibrium(self, fluid_props):
        """Steady-state model with zero irradiance and T_inlet = T_ambient."""
        cond = OPERATING_CONDITIONS["zero_irradiance_equilibrium"]
        model = create_collector_model_steady_state(
            fluid_props,
            L=COLLECTOR_LENGTH,
            mass_flow_rate=cond["mass_flow_rate"],
            irradiance=cond["DNI"],
            T_inlet=ZERO_C + cond["T_inlet_C"],
            T_ambient=ZERO_C + cond["T_ambient_C"],
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )
        model = add_steady_state_constraints(model)
        results = solve_steady_state_model(
            model, n_x=50, print_level=0, tee=False
        )
        return model, results

    def test_uniform_temperature_equilibrium(self, zero_irradiance_equilibrium):
        """Test uniform temperature when I=0 and T_inlet = T_ambient."""
        model, _ = zero_irradiance_equilibrium
        x_vals = sorted(model.x)
        cond = OPERATING_CONDITIONS["zero_irradiance_equilibrium"]
        T_expected = ZERO_C + cond["T_inlet_C"]

        for x in x_vals:
            assert value(model.T_f[x]) == pytest.approx(T_expected, rel=0.001)
            assert value(model.T_p[x]) == pytest.approx(T_expected, rel=0.001)


class TestDynamicMatchesSteadyState:
    """Tests verifying dynamic model initialized at steady-state remains there.

    When the dynamic model is initialized with the steady-state solution,
    and the inputs remain constant, the system should stay at equilibrium.

    Uses the equilibrium case (T_inlet = T_ambient, I=0) where the steady-state
    temperature is uniform throughout the pipe.
    """

    @pytest.fixture
    def equilibrium_conditions(self):
        """Operating conditions: zero irradiance, T_inlet = T_ambient."""
        return OPERATING_CONDITIONS["zero_irradiance_equilibrium"]

    @pytest.fixture
    def steady_state_solution(self, fluid_props, equilibrium_conditions):
        """Solve steady-state model for equilibrium conditions."""
        cond = equilibrium_conditions
        model = create_collector_model_steady_state(
            fluid_props,
            L=COLLECTOR_LENGTH,
            mass_flow_rate=cond["mass_flow_rate"],
            irradiance=cond["DNI"],
            T_inlet=ZERO_C + cond["T_inlet_C"],
            T_ambient=ZERO_C + cond["T_ambient_C"],
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )
        model = add_steady_state_constraints(model)
        results = solve_steady_state_model(
            model, n_x=50, print_level=0, tee=False
        )
        return model, results

    @pytest.fixture
    def dynamic_from_steady_state(
        self, fluid_props, equilibrium_conditions, steady_state_solution
    ):
        """Create dynamic model initialized from steady-state solution."""
        cond = equilibrium_conditions
        ss_model, _ = steady_state_solution

        # Extract steady-state temperature profiles
        x_vals_ss = sorted(ss_model.x)
        T_f_ss = np.array([value(ss_model.T_f[x]) for x in x_vals_ss])
        T_p_ss = np.array([value(ss_model.T_p[x]) for x in x_vals_ss])

        # For equilibrium case, temperature should be uniform
        T_uniform = ZERO_C + cond["T_inlet_C"]

        # Create constant input functions
        def zero_irradiance(t):
            return cond["DNI"]

        def constant_inlet_temp(t):
            return T_uniform

        # Create dynamic model
        t_final = 60.0  # seconds
        dynamic_model = create_collector_model(
            fluid_props,
            L=COLLECTOR_LENGTH,
            t_final=t_final,
            mass_flow_rate_func=lambda t: cond["mass_flow_rate"],
            irradiance_func=zero_irradiance,
            T_inlet_func=constant_inlet_temp,
            T_ambient=T_uniform,
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )

        # Add constraints with uniform initial conditions (from steady-state)
        dynamic_model = add_pde_constraints(
            dynamic_model,
            T_f_initial=T_uniform,
            T_p_initial=T_uniform,
        )

        # Solve dynamic model
        results = solve_model(
            dynamic_model,
            n_x=50,
            n_t=20,
            print_level=0,
            tee=False,
        )

        return dynamic_model, results, T_uniform

    def test_dynamic_solver_converges(self, dynamic_from_steady_state):
        """Test that dynamic model solver converges."""
        model, results, _ = dynamic_from_steady_state
        assert results.solver.termination_condition.name in [
            "optimal",
            "locallyOptimal",
        ]

    def test_fluid_temperature_remains_uniform(
        self, dynamic_from_steady_state
    ):
        """Test fluid temperature stays uniform throughout simulation."""
        model, _, T_uniform = dynamic_from_steady_state

        t_vals = sorted(model.t)
        x_vals = sorted(model.x)

        # Check at all time points (after t=0)
        for t in t_vals[1:]:
            T_f_at_t = [value(model.T_f[t, x]) for x in x_vals]
            T_f_mean = np.mean(T_f_at_t)
            # Should stay within 0.5% of uniform temperature
            assert T_f_mean == pytest.approx(T_uniform, rel=0.005)

    def test_wall_temperature_remains_uniform(
        self, dynamic_from_steady_state
    ):
        """Test wall temperature stays uniform throughout simulation."""
        model, _, T_uniform = dynamic_from_steady_state

        t_vals = sorted(model.t)
        x_vals = sorted(model.x)

        # Check at all time points (after t=0)
        for t in t_vals[1:]:
            T_p_at_t = [value(model.T_p[t, x]) for x in x_vals]
            T_p_mean = np.mean(T_p_at_t)
            # Should stay within 0.5% of uniform temperature
            assert T_p_mean == pytest.approx(T_uniform, rel=0.005)

    def test_final_matches_initial(self, dynamic_from_steady_state):
        """Test temperatures at final time match initial conditions."""
        model, _, T_uniform = dynamic_from_steady_state

        t_vals = sorted(model.t)
        x_vals = sorted(model.x)
        t_final = t_vals[-1]

        # Compare final to initial (t=0)
        for x in x_vals:
            T_f_initial = value(model.T_f[0, x])
            T_f_final = value(model.T_f[t_final, x])
            T_p_initial = value(model.T_p[0, x])
            T_p_final = value(model.T_p[t_final, x])

            # Should stay within 0.5% of initial value
            assert T_f_final == pytest.approx(T_f_initial, rel=0.005)
            assert T_p_final == pytest.approx(T_p_initial, rel=0.005)


class TestDynamicWarmInletSteadyState:
    """Tests for dynamic model initialized at non-uniform steady-state.

    Uses the warm inlet case (T_inlet > T_ambient, I=0) where the
    steady-state has a declining temperature profile along the pipe.
    Verifies the mean temperature stays near steady-state.
    """

    @pytest.fixture
    def warm_inlet_conditions(self):
        """Operating conditions: zero irradiance, inlet warmer than ambient."""
        return OPERATING_CONDITIONS["zero_irradiance_warm_inlet"]

    @pytest.fixture
    def steady_state_warm(self, fluid_props, warm_inlet_conditions):
        """Solve steady-state model for warm inlet conditions."""
        cond = warm_inlet_conditions
        model = create_collector_model_steady_state(
            fluid_props,
            L=COLLECTOR_LENGTH,
            mass_flow_rate=cond["mass_flow_rate"],
            irradiance=cond["DNI"],
            T_inlet=ZERO_C + cond["T_inlet_C"],
            T_ambient=ZERO_C + cond["T_ambient_C"],
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
        )
        model = add_steady_state_constraints(model)
        results = solve_steady_state_model(
            model, n_x=50, print_level=0, tee=False
        )

        # Extract temperature profiles
        x_vals = sorted(model.x)
        T_f_profile = np.array([value(model.T_f[x]) for x in x_vals])
        T_p_profile = np.array([value(model.T_p[x]) for x in x_vals])

        return model, results, (T_f_profile, T_p_profile, x_vals)

    def test_steady_state_temperature_decreases(
        self, steady_state_warm, warm_inlet_conditions
    ):
        """Test that steady-state shows temperature decrease (heat loss)."""
        model, _, (T_f_profile, T_p_profile, x_vals) = steady_state_warm
        cond = warm_inlet_conditions

        T_inlet = ZERO_C + cond["T_inlet_C"]
        T_outlet = T_f_profile[-1]

        # With T_inlet > T_ambient and I=0, temperature should decrease
        assert T_outlet < T_inlet
        # But should still be above ambient
        T_ambient = ZERO_C + cond["T_ambient_C"]
        assert T_outlet > T_ambient

    def test_steady_state_heat_loss(
        self, steady_state_warm, warm_inlet_conditions
    ):
        """Verify temperature drop is reasonable for the heat loss conditions."""
        _, _, (T_f_profile, _, _) = steady_state_warm
        cond = warm_inlet_conditions

        T_inlet_C = cond["T_inlet_C"]
        T_outlet_C = T_f_profile[-1] - ZERO_C
        delta_T = T_inlet_C - T_outlet_C

        # Temperature should drop due to heat loss, but not dramatically
        # (high mass flow rate limits the cooling)
        assert delta_T > 0  # Temperature decreases
        assert delta_T < 50  # But not by more than 50°C


class TestInitialSteadyState:
    """Tests for initial_steady_state feature in run_simulation().

    The initial_steady_state option computes steady-state temperature profiles
    using the input functions evaluated at t=0, then uses those as initial
    conditions for the dynamic simulation.
    """

    @pytest.fixture
    def test_conditions(self):
        """Operating conditions for steady-state initialization test."""
        return {
            "T_inlet_C": 270.0,
            "DNI": 800.0,
            "T_ambient_C": 25.0,
            "mass_flow_rate": 1.0,  # kg/s (different from default 0.516)
        }

    def test_compute_steady_state_initial_conditions(
        self, fluid_props, test_conditions
    ):
        """Test compute_steady_state_initial_conditions function."""
        cond = test_conditions

        # Create a model with input functions
        # Pass initial_mass_flow_rate so h_int is calculated correctly
        model = create_collector_model(
            fluid_props,
            t_final=60.0,
            T_ambient=ZERO_C + cond["T_ambient_C"],
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
            initial_mass_flow_rate=cond["mass_flow_rate"],
        )

        # Set input functions
        model.mass_flow_rate_func = lambda t: cond["mass_flow_rate"]
        model.irradiance_func = lambda t: cond["DNI"]
        model.T_inlet_func = lambda t: ZERO_C + cond["T_inlet_C"]

        # Compute steady-state initial conditions
        T_f_ss, T_p_ss, x_vals = compute_steady_state_initial_conditions(
            model, n_x=50, print_level=0, tee=False
        )

        # Verify shapes
        assert len(T_f_ss) == len(x_vals)
        assert len(T_p_ss) == len(x_vals)

        # With I > 0, fluid temperature should increase along the pipe
        # (absorbing solar energy)
        assert T_f_ss[-1] > T_f_ss[0]

        # Wall temperature profile depends on inlet conditions and heat transfer
        # Near inlet: wall is hotter than cold incoming fluid
        # Near outlet: heated fluid approaches wall temperature
        # Just verify wall temperatures are above ambient
        assert T_p_ss.min() > ZERO_C + cond["T_ambient_C"]

        # Inlet fluid temperature should match T_inlet
        assert abs(T_f_ss[0] - (ZERO_C + cond["T_inlet_C"])) < 1.0

    def test_run_simulation_with_initial_steady_state(
        self, fluid_props, test_conditions
    ):
        """Test run_simulation with initial_steady_state=True."""
        cond = test_conditions

        # Pass initial_mass_flow_rate so h_int is calculated correctly
        model = create_collector_model(
            fluid_props,
            t_final=60.0,
            T_ambient=ZERO_C + cond["T_ambient_C"],
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
            initial_mass_flow_rate=cond["mass_flow_rate"],
        )

        # Run simulation with steady-state initialization
        results = run_simulation(
            model,
            mass_flow_rate_func=lambda t: cond["mass_flow_rate"],
            irradiance_func=lambda t: cond["DNI"],
            T_inlet_func=lambda t: ZERO_C + cond["T_inlet_C"],
            initial_steady_state=True,
            n_x=30,
            n_t=10,
            print_level=0,
            tee=False,
        )

        # Check solver succeeded
        assert results.solver.termination_condition.name in ["optimal", "locallyOptimal"]

        # Get final temperatures
        T_f_final, T_p_final = get_final_temperatures(model)

        # Since inputs are constant and we start from steady-state,
        # temperatures should not change significantly
        x_vals = sorted(model.x)
        T_f_initial = np.array([value(model.T_f_init_param[x]) for x in x_vals])
        T_p_initial = np.array([value(model.T_p_init_param[x]) for x in x_vals])

        # Final should be close to initial (within 2°C for a short simulation)
        assert np.allclose(T_f_final, T_f_initial, atol=2.0)
        assert np.allclose(T_p_final, T_p_initial, atol=2.0)

    def test_initial_steady_state_not_used_when_explicit_ic_given(
        self, fluid_props, test_conditions
    ):
        """Test that initial_steady_state is ignored when explicit ICs given."""
        cond = test_conditions

        # Pass initial_mass_flow_rate so h_int is calculated correctly
        model = create_collector_model(
            fluid_props,
            t_final=60.0,
            T_ambient=ZERO_C + cond["T_ambient_C"],
            constant_density=True,
            constant_viscosity=True,
            constant_thermal_conductivity=True,
            constant_specific_heat=True,
            constant_heat_transfer_coeff=True,
            initial_mass_flow_rate=cond["mass_flow_rate"],
        )

        T_f_explicit = ZERO_C + 300.0  # Different from steady-state
        T_p_explicit = ZERO_C + 280.0

        # Run with both initial_steady_state=True and explicit ICs
        results = run_simulation(
            model,
            mass_flow_rate_func=lambda t: cond["mass_flow_rate"],
            irradiance_func=lambda t: cond["DNI"],
            T_inlet_func=lambda t: ZERO_C + cond["T_inlet_C"],
            T_f_initial=T_f_explicit,
            T_p_initial=T_p_explicit,
            initial_steady_state=True,  # Should be ignored
            n_x=20,
            n_t=5,
            print_level=0,
            tee=False,
        )

        assert results.solver.termination_condition.name in ["optimal", "locallyOptimal"]

        # Verify explicit ICs were used (check initial param values)
        x_vals = sorted(model.x)
        for x in x_vals:
            assert abs(value(model.T_f_init_param[x]) - T_f_explicit) < 0.01
            assert abs(value(model.T_p_init_param[x]) - T_p_explicit) < 0.01
