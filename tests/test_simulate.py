"""Tests for simulate module."""

import numpy as np
import pytest

from simulate import (
    ConstantInput,
    ForwardEuler,
    RungeKutta4,
    SimulationConfig,
    SimulationEngine,
    SinusoidalInput,
    StepInput,
)


def test_simple_pendulum():
    """Test simulation of simple pendulum with NumPy."""

    # Define pendulum dynamics
    def pendulum_dynamics(t, x, u, p):
        theta, omega = x
        g = p["g"]
        L = p["L"]
        tau = u.get("torque", 0.0)

        dtheta = omega
        domega = -g / L * np.sin(theta) + tau / (p.get("m", 1.0) * L**2)

        return np.array([dtheta, domega])

    # Create integrator and engine
    integrator = RungeKutta4(pendulum_dynamics)
    engine = SimulationEngine(integrator)

    # Configure simulation
    config = SimulationConfig(
        t_span=(0.0, 5.0),
        x0=np.array([np.pi / 4, 0.0]),
        inputs={"torque": ConstantInput(0.0)},
        parameters={"g": 9.81, "L": 1.0, "m": 1.0},
        dt=0.01,
    )

    # Run simulation
    result = engine.simulate(config)

    # Verify results
    assert result.time[0] == 0.0
    assert result.time[-1] == pytest.approx(5.0, rel=1e-6)
    assert result.states.shape[0] == len(result.time)
    assert result.states.shape[1] == 2
    assert (
        result.outputs.shape == result.states.shape
    )  # Default: outputs = states


def test_cartpole_casadi():
    """Test cart-pole (inverted pendulum) system using CasADi.

    State vector: [x, dx/dt, theta, dtheta/dt]
    - x: cart position
    - dx/dt: cart velocity
    - theta: pendulum angle from vertical (radians)
    - dtheta/dt: pendulum angular velocity

    This test demonstrates the framework-agnostic design by implementing
    the same cart-pole system in CasADi instead of pure Python/NumPy.
    """
    try:
        import casadi as cas

        from simulate import CasADiIntegrator
    except ImportError:
        pytest.skip("CasADi not installed")

    # Define symbolic variables
    x = cas.MX.sym("x", 4)  # State: [x, dx, theta, dtheta]
    u = cas.MX.sym("u", 1)  # Input: [force]
    p = cas.MX.sym("p", 5)  # Parameters: [m, M, L, g, d]

    # Extract states
    x_pos = x[0]  # Cart position
    x_vel = x[1]  # Cart velocity
    theta = x[2]  # Pendulum angle
    theta_dot = x[3]  # Pendulum angular velocity

    # Extract parameters
    m = p[0]  # Pendulum mass
    M = p[1]  # Cart mass
    L = p[2]  # Pendulum length
    g = p[3]  # Gravity
    d = p[4]  # Damping

    # Extract input
    force = u[0]

    # Cart-pendulum dynamics (symbolic CasADi expressions)
    sin_theta = cas.sin(theta)
    cos_theta = cas.cos(theta)
    mL = m * L
    D = 1 / (m * L * L * (M + m * (1 - cos_theta**2)))
    b = mL * theta_dot**2 * sin_theta - d * x_vel

    # State derivatives
    dx = cas.vertcat(
        x_vel,  # dx/dt = velocity
        D * (-mL * g * cos_theta * sin_theta + L * b)
        + (1 / (M + m)) * force,  # cart acceleration
        theta_dot,  # dtheta/dt = angular velocity
        D * ((m + M) * m * g * L * sin_theta - mL * cos_theta * b)
        - (cos_theta / (L * (M + m))) * force,  # angular acceleration
    )

    # Create CasADi function
    dynamics = cas.Function("cartpole", [x, u, p], [dx])

    # Create CasADi integrator
    integrator = CasADiIntegrator(
        dynamics, method="cvodes", options={"abstol": 1e-8, "reltol": 1e-6}
    )

    # Conversion function for CasADi -> NumPy
    def casadi_to_numpy(x_cas):
        if hasattr(x_cas, "full"):
            arr = np.array(x_cas.full()).flatten()
        else:
            arr = np.asarray(x_cas)
            if arr.ndim > 0:
                arr = arr.flatten()
        # Return Python scalar for single-element arrays to ensure
        # consistent 1D shapes when stacking
        return arr.item() if arr.ndim == 1 and arr.size == 1 else arr

    # Create simulation engine
    engine = SimulationEngine(integrator=integrator, to_numpy=casadi_to_numpy)

    # Initial condition: cart at origin, pendulum at small angle
    x0 = cas.DM(
        [
            0.0,  # cart position
            0.0,  # cart velocity
            0.1,  # pendulum angle (radians, ~5.7 degrees)
            0.0,  # pendulum angular velocity
        ]
    )

    # Parameters for inverted pendulum
    params = cas.DM(
        [
            1.0,  # m: Pendulum mass
            5.0,  # M: Cart mass
            2.0,  # L: Pendulum length
            10.0,  # g: Gravity (positive for inverted pendulum)
            1.0,  # d: Damping coefficient
        ]
    )

    # Configure simulation
    config = SimulationConfig(
        t_span=(0.0, 10.0),
        x0=x0,
        inputs={"force": ConstantInput(cas.DM(0.0))},  # No control force
        parameters=params,
        dt=0.02,
    )

    # Run simulation
    result = engine.simulate(config)

    # Verify results
    assert result.states.shape[0] == len(result.time)
    assert result.states.shape[1] == 4  # 4 states
    assert result.n_states == 4

    # Pendulum should fall (angle should increase in magnitude)
    # Since there's no control and it starts at small angle
    initial_angle = abs(float(x0[2]))
    final_angle = abs(result.states.iloc[-1]["x3"])  # x3 is the 3rd state (theta)
    assert final_angle > initial_angle  # Should have moved


def test_cartpole_casadi_with_control():
    """Test CasADi cart-pole with step input control force."""
    try:
        import casadi as cas

        from simulate import CasADiIntegrator
    except ImportError:
        pytest.skip("CasADi not installed")

    # Define symbolic cart-pole dynamics (same as above)
    x = cas.MX.sym("x", 4)
    u = cas.MX.sym("u", 1)
    p = cas.MX.sym("p", 5)

    m, M, L, g, d = p[0], p[1], p[2], p[3], p[4]
    force = u[0]

    sin_theta = cas.sin(x[2])
    cos_theta = cas.cos(x[2])
    mL = m * L
    D = 1 / (m * L * L * (M + m * (1 - cos_theta**2)))
    b = mL * x[3] ** 2 * sin_theta - d * x[1]

    dx = cas.vertcat(
        x[1],
        D * (-mL * g * cos_theta * sin_theta + L * b) + (1 / (M + m)) * force,
        x[3],
        D * ((m + M) * m * g * L * sin_theta - mL * cos_theta * b)
        - (cos_theta / (L * (M + m))) * force,
    )

    dynamics = cas.Function("cartpole", [x, u, p], [dx])

    # Create integrator and engine
    integrator = CasADiIntegrator(dynamics, method="cvodes")

    # Conversion function for CasADi -> NumPy
    def casadi_to_numpy(x_cas):
        if hasattr(x_cas, "full"):
            arr = np.array(x_cas.full()).flatten()
        else:
            arr = np.asarray(x_cas)
            if arr.ndim > 0:
                arr = arr.flatten()
        # Return Python scalar for single-element arrays to ensure
        # consistent 1D shapes when stacking
        return arr.item() if arr.ndim == 1 and arr.size == 1 else arr

    engine = SimulationEngine(integrator=integrator, to_numpy=casadi_to_numpy)

    # Step force: 10N applied at t=2s
    config = SimulationConfig(
        t_span=(0.0, 5.0),
        x0=cas.DM([0.0, 0.0, 0.1, 0.0]),
        inputs={"force": StepInput([2.0], [cas.DM(0.0), cas.DM(10.0)])},
        parameters=cas.DM(
            [1.0, 5.0, 2.0, 10.0, 1.0]
        ),  # g=10.0 for inverted pendulum
        dt=0.02,
    )

    result = engine.simulate(config)

    # Cart velocity should change after force is applied
    idx_before = np.argmax(result.time >= 1.8)
    idx_after = np.argmax(result.time >= 2.5)
    velocity_before = result.states.iloc[idx_before]["x2"]  # x2 is cart velocity
    velocity_after = result.states.iloc[idx_after]["x2"]
    assert abs(velocity_after) > abs(velocity_before)


def test_step_input():
    """Test simulation with step input."""

    # Simple first-order system: dx/dt = -x + u
    def dynamics(t, x, u, p):
        return -p["tau"] * x + u["step"]

    integrator = ForwardEuler(dynamics)
    engine = SimulationEngine(integrator)

    # Step input from 0 to 1 at t=2
    config = SimulationConfig(
        t_span=(0.0, 5.0),
        x0=np.array([0.0]),
        inputs={"step": StepInput([2.0], [0.0, 1.0])},
        parameters={"tau": 1.0},
        dt=0.1,
    )

    result = engine.simulate(config)

    # Check that state increases after step
    idx_before = np.argmax(result.time >= 1.9)
    idx_after = np.argmax(result.time >= 2.5)
    assert result.states.iloc[idx_after]["x1"] > result.states.iloc[idx_before]["x1"]


def test_sinusoidal_input():
    """Test simulation with sinusoidal input."""

    # Linear system: dx/dt = u
    def dynamics(t, x, u, p):
        return np.array([u["sin"]])

    integrator = RungeKutta4(dynamics)
    engine = SimulationEngine(integrator)

    # Sinusoidal input
    config = SimulationConfig(
        t_span=(0.0, 10.0),
        x0=np.array([0.0]),
        inputs={"sin": SinusoidalInput(amplitude=1.0, frequency=1.0)},
        parameters={},
        dt=0.01,
    )

    result = engine.simulate(config)

    # State should oscillate
    assert result.states["x1"].max() > 0
    assert result.states["x1"].min() < 0


def test_custom_output_function():
    """Test simulation with custom output function."""

    # Simple dynamics
    def dynamics(t, x, u, p):
        return -x

    # Custom output: y = x^2
    def output_func(t, x, u, p):
        return x**2

    integrator = RungeKutta4(dynamics)
    engine = SimulationEngine(integrator, output_func=output_func)

    config = SimulationConfig(
        t_span=(0.0, 2.0),
        x0=np.array([2.0]),
        inputs={},
        parameters={},
        dt=0.1,
    )

    result = engine.simulate(config)

    # Outputs should be squared states
    expected_outputs = result.states**2
    np.testing.assert_array_almost_equal(result.outputs.to_numpy(), expected_outputs.to_numpy())


def test_variable_time_steps():
    """Test simulation with variable time steps."""

    def dynamics(t, x, u, p):
        return -x

    integrator = RungeKutta4(dynamics)
    engine = SimulationEngine(integrator)

    # Variable time points
    time_points = np.concatenate(
        [
            np.linspace(0, 1, 10),
            np.linspace(1, 2, 5),
            np.linspace(2, 5, 20),
        ]
    )

    config = SimulationConfig(
        t_span=(0.0, 5.0),
        x0=np.array([1.0]),
        inputs={},
        parameters={},
        time_points=time_points,
    )

    result = engine.simulate(config)

    assert len(result.time) == len(time_points)
    np.testing.assert_array_equal(result.time, time_points)


def test_save_options():
    """Test selective saving of states, outputs, inputs."""

    def dynamics(t, x, u, p):
        return -x

    integrator = RungeKutta4(dynamics)
    engine = SimulationEngine(integrator)

    # Don't save states
    config = SimulationConfig(
        t_span=(0.0, 1.0),
        x0=np.array([1.0]),
        inputs={"u": ConstantInput(0.5)},
        parameters={},
        dt=0.1,
        save_states=False,
        save_outputs=True,
        save_inputs=True,
    )

    result = engine.simulate(config)

    assert result.states is None
    assert result.outputs is not None
    assert result.inputs is not None


def test_simulation_result_dataframe():
    """Test conversion to pandas DataFrame."""

    def dynamics(t, x, u, p):
        return -x

    integrator = RungeKutta4(dynamics)
    engine = SimulationEngine(integrator)

    config = SimulationConfig(
        t_span=(0.0, 1.0),
        x0=np.array([1.0, 2.0]),
        inputs={"u": ConstantInput(0.5)},
        parameters={},
        dt=0.1,
    )

    result = engine.simulate(config)
    df = result.to_dataframe()

    # Check columns
    assert "time" in df.columns
    assert "x1" in df.columns  # First state (using 1-based indexing now)
    assert "x2" in df.columns  # Second state
    assert "u" in df.columns  # Input name comes from dict key

    # Check length
    assert len(df) == len(result.time)


def test_configuration_validation():
    """Test that configuration validation catches errors."""

    # Should raise error: no dt or time_points
    with pytest.raises(
        ValueError, match="Must provide either dt or time_points"
    ):
        config = SimulationConfig(
            t_span=(0.0, 1.0), x0=np.array([1.0]), inputs={}, parameters={}
        )

    # Should raise error: both dt and time_points
    with pytest.raises(ValueError, match="Cannot provide both"):
        config = SimulationConfig(
            t_span=(0.0, 1.0),
            x0=np.array([1.0]),
            inputs={},
            parameters={},
            dt=0.1,
            time_points=np.linspace(0, 1, 10),
        )
