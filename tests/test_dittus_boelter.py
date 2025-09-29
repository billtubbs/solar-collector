#!/usr/bin/env python3
"""
Test script for Dittus-Boelter enhanced Pyomo models
"""

import pytest

from solar_collector.heat_transfer import calculate_heat_transfer_coefficient
from solar_collector.solar_collector_dae_pyo import (
    add_pde_constraints as add_single_constraints,
)
from solar_collector.solar_collector_dae_pyo import (
    create_pipe_flow_model as create_single_temp_model,
)
from solar_collector.solar_collector_dae_pyo import (
    solve_model as solve_single_model,
)
from solar_collector.solar_collector_dae_pyo_two_temp import (
    add_pde_constraints as add_two_constraints,
)
from solar_collector.solar_collector_dae_pyo_two_temp import (
    create_pipe_flow_model as create_two_temp_model,
)
from solar_collector.solar_collector_dae_pyo_two_temp import (
    solve_model as solve_two_model,
)


def test_heat_transfer_coefficient_calculations():
    """Test heat transfer coefficient calculations with different velocities"""
    velocities = [0.1, 0.2, 0.5, 1.0]
    expected_regimes = ["laminar", "laminar", "laminar", "turbulent"]

    for v, expected_regime in zip(velocities, expected_regimes):
        h, re, pr, nu = calculate_heat_transfer_coefficient(
            velocity=v,
            pipe_diameter=0.07,
        )

        # Basic sanity checks
        assert h > 0, f"Heat transfer coefficient should be positive, got {h}"
        assert re > 0, f"Reynolds number should be positive, got {re}"
        assert pr > 0, f"Prandtl number should be positive, got {pr}"
        assert nu > 0, f"Nusselt number should be positive, got {nu}"

        # Check flow regime transition
        if expected_regime == "laminar":
            assert re <= 4000, f"Expected laminar flow but Re={re} > 4000"
            assert nu == pytest.approx(4.36, rel=1e-3), (
                f"Laminar Nu should be 4.36, got {nu}"
            )
        else:
            assert re > 4000, f"Expected turbulent flow but Re={re} <= 4000"
            assert nu > 4.36, f"Turbulent Nu should be > 4.36, got {nu}"


def test_single_temperature_model():
    """Test single temperature model with Dittus-Boelter correlation"""
    # Create model with Dittus-Boelter correlation
    model = create_single_temp_model(
        t_final=60.0,  # Short simulation
        use_dittus_boelter=True,
    )

    # Add constraints
    model = add_single_constraints(model)

    # Solve
    results = solve_single_model(model, n_x=21, n_t=11, tol=1e-4)

    # Check solver status
    assert str(results.solver.status) == "ok", (
        f"Solver failed with status: {results.solver.status}"
    )
    assert str(results.solver.termination_condition) == "optimal", (
        f"Non-optimal termination: {results.solver.termination_condition}"
    )


def test_two_temperature_model():
    """Test two temperature model with Dittus-Boelter correlation"""
    # Create model with Dittus-Boelter correlation
    model = create_two_temp_model(
        t_final=60.0,  # Short simulation
        use_dittus_boelter=True,
    )

    # Add constraints
    model = add_two_constraints(model)

    # Solve
    results = solve_two_model(model, n_x=21, n_t=11, tol=1e-4)

    # Check solver status
    assert str(results.solver.status) == "ok", (
        f"Solver failed with status: {results.solver.status}"
    )
    assert str(results.solver.termination_condition) == "optimal", (
        f"Non-optimal termination: {results.solver.termination_condition}"
    )


def test_heat_transfer_coefficient_comparison():
    """Test that turbulent flow gives higher heat transfer than laminar"""
    # Laminar flow case
    h_laminar, _, _, _ = calculate_heat_transfer_coefficient(
        velocity=0.1,
        pipe_diameter=0.07,
    )

    # Turbulent flow case
    h_turbulent, _, _, _ = calculate_heat_transfer_coefficient(
        velocity=1.0,
        pipe_diameter=0.07,
    )

    # Turbulent should have much higher heat transfer
    assert h_turbulent > h_laminar * 10, (
        f"Turbulent h={h_turbulent} should be much higher than "
        f"laminar h={h_laminar}"
    )


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
