"""Tests for solar_collector_dae_pyo_two_temp module.

Tests for model creation and constraint setup with both constant
and temperature-dependent fluid properties.
"""

import pytest
import numpy as np
from pyomo.environ import value
from pyomo.dae import ContinuousSet

from solar_collector.fluid_properties import SYLTHERM800
from solar_collector.solar_collector_dae_pyo_two_temp import (
    ZERO_C,
    COLLECTOR_LENGTH,
    PIPE_DIAMETER,
    PIPE_WALL_THICKNESS,
    AXIAL_DISPERSION_COEFF,
    create_pipe_flow_model,
    add_pde_constraints,
    solve_model,
)


@pytest.fixture
def fluid_props():
    """Create SYLTHERM800 fluid properties object."""
    return SYLTHERM800()


class TestCreatePipeFlowModelConstantProperties:
    """Tests for create_pipe_flow_model with constant fluid properties."""

    @pytest.fixture
    def model_constant(self, fluid_props):
        """Create model with constant fluid properties."""
        return create_pipe_flow_model(
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
        assert hasattr(model_constant, 'x')
        assert hasattr(model_constant, 't')
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
        assert hasattr(model_constant, 'T_f')
        assert hasattr(model_constant, 'T_p')

    def test_derivative_variables_exist(self, model_constant):
        """Test that derivative variables are created."""
        # Fluid temperature derivatives
        assert hasattr(model_constant, 'dT_f_dt')
        assert hasattr(model_constant, 'dT_f_dx')
        assert hasattr(model_constant, 'd2T_f_dx2')

        # Pipe wall temperature derivatives
        assert hasattr(model_constant, 'dT_p_dt')
        assert hasattr(model_constant, 'dT_p_dx')
        assert hasattr(model_constant, 'd2T_p_dx2')

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
        assert hasattr(model_constant, 'rho_p')
        assert hasattr(model_constant, 'cp_p')
        assert hasattr(model_constant, 'k_p')
        assert hasattr(model_constant, 'h_ext')
        assert hasattr(model_constant, 'T_ambient')

    def test_axial_dispersion_coefficient(self, model_constant):
        """Test axial dispersion coefficient parameter."""
        assert hasattr(model_constant, 'D_ax')
        assert value(model_constant.D_ax) == pytest.approx(AXIAL_DISPERSION_COEFF)

    def test_fluid_property_expressions_exist(self, model_constant):
        """Test fluid property expressions are created."""
        assert hasattr(model_constant, 'rho_f')
        assert hasattr(model_constant, 'eta_f')
        assert hasattr(model_constant, 'k_f')
        assert hasattr(model_constant, 'cp_f')

    def test_constant_fluid_property_values(self, model_constant, fluid_props):
        """Test constant fluid property values match expected at T_ref."""
        T_ref = model_constant.T_ref

        # Get expected values from fluid properties at T_ref
        expected_rho = fluid_props.density(T_ref)
        expected_eta = fluid_props.viscosity(T_ref)
        expected_lam = fluid_props.thermal_conductivity(T_ref)
        expected_cp = fluid_props.heat_capacity(T_ref)

        # Check constant parameter values
        assert value(model_constant._rho_f_const) == pytest.approx(expected_rho)
        assert value(model_constant._eta_f_const) == pytest.approx(expected_eta)
        assert value(model_constant._k_f_const) == pytest.approx(expected_lam)
        assert value(model_constant._cp_f_const) == pytest.approx(expected_cp)

    def test_velocity_expression_exists(self, model_constant):
        """Test velocity expression is created."""
        assert hasattr(model_constant, 'v')

    def test_heat_transfer_coefficient_expression_exists(self, model_constant):
        """Test internal heat transfer coefficient expression is created."""
        assert hasattr(model_constant, 'h_int')

    def test_constant_heat_transfer_coefficient_value(self, model_constant):
        """Test constant h_int value is positive and reasonable."""
        # h_int should be calculated from Dittus-Boelter at T_ref
        h_int_val = value(model_constant._h_int_const)

        # Typical values for turbulent oil flow: 100-2000 W/(m²·K)
        assert h_int_val > 50.0
        assert h_int_val < 5000.0

    def test_time_varying_parameters_exist(self, model_constant):
        """Test time-varying input parameters are created."""
        assert hasattr(model_constant, 'm_dot')
        assert hasattr(model_constant, 'I')
        assert hasattr(model_constant, 'T_inlet')

    def test_solar_collector_parameters(self, model_constant):
        """Test solar collector parameters exist."""
        assert hasattr(model_constant, 'c')  # concentration factor
        assert hasattr(model_constant, 'epsilon')  # optical efficiency

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
        return create_pipe_flow_model(
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
        assert hasattr(model_temp_dependent, 'x')
        assert hasattr(model_temp_dependent, 't')

    def test_temperature_variables_exist(self, model_temp_dependent):
        """Test that fluid and wall temperature variables are created."""
        assert hasattr(model_temp_dependent, 'T_f')
        assert hasattr(model_temp_dependent, 'T_p')

    def test_fluid_property_expressions_exist(self, model_temp_dependent):
        """Test fluid property expressions are created for temp-dependent case."""
        assert hasattr(model_temp_dependent, 'rho_f')
        assert hasattr(model_temp_dependent, 'eta_f')
        assert hasattr(model_temp_dependent, 'k_f')
        assert hasattr(model_temp_dependent, 'cp_f')

    def test_velocity_expression_exists(self, model_temp_dependent):
        """Test velocity expression is created."""
        assert hasattr(model_temp_dependent, 'v')

    def test_heat_transfer_coefficient_expression_exists(self, model_temp_dependent):
        """Test internal heat transfer coefficient expression is created."""
        assert hasattr(model_temp_dependent, 'h_int')

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
        assert not hasattr(model_temp_dependent, '_rho_f_const')
        assert not hasattr(model_temp_dependent, '_eta_f_const')
        assert not hasattr(model_temp_dependent, '_k_f_const')
        assert not hasattr(model_temp_dependent, '_cp_f_const')
        assert not hasattr(model_temp_dependent, '_h_int_const')


class TestAddPdeConstraintsConstantProperties:
    """Tests for add_pde_constraints with constant fluid properties."""

    @pytest.fixture
    def model_with_constraints(self, fluid_props):
        """Create model with constraints added (constant properties)."""
        model = create_pipe_flow_model(
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
        assert hasattr(model_with_constraints, 'fluid_pde_constraint')

    def test_wall_pde_constraint_exists(self, model_with_constraints):
        """Test that wall PDE constraint is added."""
        assert hasattr(model_with_constraints, 'wall_pde_constraint')

    def test_fluid_initial_condition_exists(self, model_with_constraints):
        """Test that fluid initial condition constraint is added."""
        assert hasattr(model_with_constraints, 'fluid_initial_condition')

    def test_wall_initial_condition_exists(self, model_with_constraints):
        """Test that wall initial condition constraint is added."""
        assert hasattr(model_with_constraints, 'wall_initial_condition')

    def test_inlet_boundary_condition_exists(self, model_with_constraints):
        """Test that inlet boundary condition is added."""
        assert hasattr(model_with_constraints, 'inlet_bc')

    def test_objective_exists(self, model_with_constraints):
        """Test that objective function is added."""
        assert hasattr(model_with_constraints, 'obj')


class TestAddPdeConstraintsTemperatureDependent:
    """Tests for add_pde_constraints with temperature-dependent properties."""

    @pytest.fixture
    def model_with_constraints_temp_dep(self, fluid_props):
        """Create model with constraints added (temp-dependent properties)."""
        model = create_pipe_flow_model(
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

    def test_fluid_pde_constraint_exists(self, model_with_constraints_temp_dep):
        """Test that fluid PDE constraint is added."""
        assert hasattr(model_with_constraints_temp_dep, 'fluid_pde_constraint')

    def test_wall_pde_constraint_exists(self, model_with_constraints_temp_dep):
        """Test that wall PDE constraint is added."""
        assert hasattr(model_with_constraints_temp_dep, 'wall_pde_constraint')

    def test_fluid_initial_condition_exists(self, model_with_constraints_temp_dep):
        """Test that fluid initial condition constraint is added."""
        assert hasattr(model_with_constraints_temp_dep, 'fluid_initial_condition')

    def test_wall_initial_condition_exists(self, model_with_constraints_temp_dep):
        """Test that wall initial condition constraint is added."""
        assert hasattr(model_with_constraints_temp_dep, 'wall_initial_condition')

    def test_inlet_boundary_condition_exists(self, model_with_constraints_temp_dep):
        """Test that inlet boundary condition is added."""
        assert hasattr(model_with_constraints_temp_dep, 'inlet_bc')

    def test_objective_exists(self, model_with_constraints_temp_dep):
        """Test that objective function is added."""
        assert hasattr(model_with_constraints_temp_dep, 'obj')


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
        model = create_pipe_flow_model(
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
            'optimal', 'locallyOptimal'
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

    def test_fluid_properties_constant_and_correct(self, solved_model, fluid_props):
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
