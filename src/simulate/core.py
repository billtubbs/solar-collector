"""Core simulation engine and protocols for framework-agnostic simulation.

This module provides a flexible simulation framework that works with
dynamical systems from any framework (CasADi, JAX, Pyomo, NumPy, etc.)
through duck-typing and protocol-based interfaces.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np


class Integrator(Protocol):
    """Protocol for integrators (framework-specific).

    The user provides an integrator that knows how to advance
    their specific system (CasADi, Pyomo, JAX, etc.) one time step.

    The integrator is responsible for:
    - Calling the appropriate solver for the framework
    - Handling the framework-specific data types
    - Returning the next state in the same format
    """

    def __call__(self, t: float, x: Any, dt: float, u: Any, p: Any) -> Any:
        """Advance state from t to t+dt.

        Parameters
        ----------
        t : float
            Current time
        x : array-like
            Current state (framework-specific type)
        dt : float
            Time step size
        u : dict or array-like
            Inputs at time t
        p : dict or array-like
            System parameters

        Returns
        -------
        x_next : array-like
            State at t+dt (same type as x)
        """
        ...


class OutputFunction(Protocol):
    """Protocol for output computation (optional).

    If not provided, states are used as outputs.
    """

    def __call__(self, t: float, x: Any, u: Any, p: Any) -> Any:
        """Compute outputs y = h(t, x, u, p).

        Parameters
        ----------
        t : float
            Current time
        x : array-like
            Current state
        u : dict or array-like
            Current inputs
        p : dict or array-like
            System parameters

        Returns
        -------
        y : array-like
            System outputs
        """
        ...


class InputSignal(Protocol):
    """Protocol for input signals.

    Input signals are callable objects that return the input value
    at a given time.
    """

    def __call__(self, t: float) -> Any:
        """Evaluate input at time t.

        Parameters
        ----------
        t : float
            Time at which to evaluate input

        Returns
        -------
        value : scalar or array-like
            Input value at time t
        """
        ...


@dataclass
class SimulationConfig:
    """Configuration for simulation.

    Parameters
    ----------
    t_span : (float, float)
        Start and end times (t0, tf)
    x0 : array-like
        Initial state (framework-specific type)
    inputs : dict of str to InputSignal
        Time-varying or constant inputs, keyed by name
    parameters : dict
        System parameters
    dt : float, optional
        Fixed time step size. Either dt or time_points must be provided.
    time_points : array-like, optional
        Specific time points to evaluate (for variable step sizes)
    state_names : list of str, optional
        Names for state variables. If None, defaults to ['x1', 'x2', ...]
    output_names : list of str, optional
        Names for output variables. If None, defaults to ['y1', 'y2', ...]
    save_states : bool, default=True
        Whether to save state trajectory
    save_outputs : bool, default=True
        Whether to save output trajectory
    save_inputs : bool, default=True
        Whether to save input trajectory

    Examples
    --------
    >>> config = SimulationConfig(
    ...     t_span=(0.0, 10.0),
    ...     x0=np.array([1.0, 0.0]),
    ...     inputs={'u': ConstantInput(0.0)},
    ...     parameters={'mass': 1.0},
    ...     state_names=['position', 'velocity'],
    ...     dt=0.01
    ... )
    """

    t_span: tuple[float, float]
    x0: Any
    inputs: Dict[str, InputSignal]
    parameters: Dict[str, Any]
    dt: Optional[float] = None
    time_points: Optional[np.ndarray] = None
    state_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
    save_states: bool = True
    save_outputs: bool = True
    save_inputs: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.dt is None and self.time_points is None:
            raise ValueError("Must provide either dt or time_points")
        if self.dt is not None and self.time_points is not None:
            raise ValueError("Cannot provide both dt and time_points")


class SimulationEngine:
    """Framework-agnostic discrete-time simulation engine.

    The engine handles:
    - Time-stepping loop with variable or fixed step sizes
    - Input signal evaluation at each time step
    - Data storage and organization
    - Conversion to numpy arrays for analysis

    The user provides:
    - Integrator (framework-specific)
    - Output function (optional, defaults to states)
    - Conversion to numpy (optional, defaults to np.asarray)

    Parameters
    ----------
    integrator : Integrator
        Framework-specific integrator that advances state from t to t+dt
    output_func : callable, optional
        Function to compute outputs y = h(t, x, u, p).
        If None, states are returned as outputs.
    to_numpy : callable, optional
        Function to convert framework-specific arrays to numpy arrays.
        Examples:
        - CasADi: lambda x: np.array(x.full()).flatten()
        - JAX: lambda x: np.asarray(x)
        - Pyomo: custom extraction from model
        If None, uses np.asarray()

    Examples
    --------
    >>> # With NumPy
    >>> def integrate(t, x, dt, u, p):
    ...     # Simple Euler step
    ...     dx = dynamics(t, x, u, p)
    ...     return x + dt * dx
    >>> engine = SimulationEngine(integrator=integrate)
    >>>
    >>> # With CasADi
    >>> integrator = CasADiIntegrator(dynamics_func)
    >>> engine = SimulationEngine(
    ...     integrator=integrator,
    ...     to_numpy=lambda x: np.array(x.full()).flatten()
    ... )
    """

    def __init__(
        self,
        integrator: Integrator,
        output_func: Optional[OutputFunction] = None,
        to_numpy: Optional[Callable[[Any], np.ndarray]] = None,
    ):
        self.integrator = integrator
        self.output_func = output_func
        self.to_numpy = to_numpy or (lambda x: np.asarray(x))

    def simulate(self, config: SimulationConfig) -> "SimulationResult":
        """Run simulation from t0 to tf.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration including time span, initial
            conditions, inputs, and parameters

        Returns
        -------
        result : SimulationResult
            Contains time points, states, outputs, and inputs

        Notes
        -----
        The simulation loop:
        1. Evaluates input signals at current time
        2. Computes outputs (if output_func provided)
        3. Calls integrator to advance state
        4. Stores results in numpy arrays

        Examples
        --------
        >>> config = SimulationConfig(
        ...     t_span=(0.0, 10.0),
        ...     x0=np.array([1.0, 0.0]),
        ...     inputs={'torque': ConstantInput(0.0)},
        ...     parameters={'g': 9.81, 'L': 1.0},
        ...     dt=0.01
        ... )
        >>> result = engine.simulate(config)
        >>> result.plot()
        """
        # Setup time vector
        if config.time_points is not None:
            t_vec = np.asarray(config.time_points)
        else:
            t0, tf = config.t_span
            t_vec = np.arange(t0, tf + config.dt / 2, config.dt)

        n_steps = len(t_vec)

        # Initialize storage
        states = [] if config.save_states else None
        outputs = [] if config.save_outputs else None
        input_vals = (
            {key: [] for key in config.inputs.keys()}
            if config.save_inputs
            else None
        )

        # Initial conditions
        x = config.x0
        if config.save_states:
            states.append(self.to_numpy(x))

        # Time-stepping loop
        for i in range(n_steps - 1):
            t = t_vec[i]
            dt_step = t_vec[i + 1] - t

            # Evaluate inputs at current time
            u = {key: sig(t) for key, sig in config.inputs.items()}

            if config.save_inputs:
                for key in config.inputs.keys():
                    input_vals[key].append(self.to_numpy(u[key]))

            # Compute output at current state (optional)
            if config.save_outputs and self.output_func is not None:
                y = self.output_func(t, x, u, config.parameters)
                outputs.append(self.to_numpy(y))
            elif config.save_outputs:
                # Default: outputs = states
                outputs.append(self.to_numpy(x))

            # Integration step (user's framework-specific integrator)
            x = self.integrator(t, x, dt_step, u, config.parameters)

            # Save state
            if config.save_states:
                states.append(self.to_numpy(x))

        # Final values at t_final
        t_final = t_vec[-1]
        u_final = {key: sig(t_final) for key, sig in config.inputs.items()}

        if config.save_outputs and self.output_func is not None:
            y_final = self.output_func(t_final, x, u_final, config.parameters)
            outputs.append(self.to_numpy(y_final))
        elif config.save_outputs:
            outputs.append(self.to_numpy(x))

        if config.save_inputs:
            for key in config.inputs.keys():
                input_vals[key].append(self.to_numpy(u_final[key]))

        # Import here to avoid circular dependency
        from simulate.results import SimulationResult
        import pandas as pd

        # Convert states to DataFrame with proper column names
        states_df = None
        if config.save_states:
            states_array = np.array(states)
            n_states = states_array.shape[1] if states_array.ndim > 1 else 1

            if config.state_names is not None:
                state_cols = config.state_names
            else:
                state_cols = [f"x{i+1}" for i in range(n_states)]

            if states_array.ndim == 1:
                states_df = pd.DataFrame(
                    {state_cols[0]: states_array}, index=t_vec
                )
            else:
                states_df = pd.DataFrame(
                    states_array, index=t_vec, columns=state_cols
                )
            states_df.index.name = "time"

        # Convert outputs to DataFrame with proper column names
        outputs_df = None
        if config.save_outputs:
            outputs_array = np.array(outputs)
            n_outputs = outputs_array.shape[1] if outputs_array.ndim > 1 else 1

            if config.output_names is not None:
                output_cols = config.output_names
            else:
                output_cols = [f"y{i+1}" for i in range(n_outputs)]

            if outputs_array.ndim == 1:
                outputs_df = pd.DataFrame(
                    {output_cols[0]: outputs_array}, index=t_vec
                )
            else:
                outputs_df = pd.DataFrame(
                    outputs_array, index=t_vec, columns=output_cols
                )
            outputs_df.index.name = "time"

        # Convert inputs to DataFrame (input names come from dict keys)
        inputs_df = None
        if config.save_inputs:
            inputs_dict = {k: np.array(v) for k, v in input_vals.items()}
            inputs_df = pd.DataFrame(inputs_dict, index=t_vec)
            inputs_df.index.name = "time"

        return SimulationResult(
            time=t_vec,
            states=states_df,
            outputs=outputs_df,
            inputs=inputs_df,
            config=config,
        )
