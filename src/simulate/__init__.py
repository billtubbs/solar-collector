"""Framework-agnostic simulation system for dynamical systems.

This package provides a flexible simulation framework that works with
dynamical systems from any framework (CasADi, JAX, Pyomo, NumPy, etc.)
through duck-typing and protocol-based interfaces.

The simulation engine handles time-stepping, input evaluation, and data
storage, while users provide framework-specific integrators that leverage
optimized solvers from their chosen framework.

Main Components
---------------
SimulationEngine : Core simulation orchestrator
SimulationConfig : Configuration dataclass
SimulationResult : Results container with plotting and analysis

Input Signals
-------------
ConstantInput : Constant value
StepInput : Piecewise constant steps
RampInput : Linear ramp
InterpolatedInput : Interpolated from data points
SinusoidalInput : Sine wave
FunctionInput : Custom function
CompositeInput : Combination of signals

Integrators
-----------
ForwardEuler, RungeKutta4 : Simple NumPy-based
SciPyIntegrator : scipy.integrate.solve_ivp wrapper
CasADiIntegrator, CasADiRK4 : CasADi wrappers
JAXIntegrator : JAX/diffrax wrapper

Examples
--------
>>> from simulate import (
...     SimulationEngine, SimulationConfig,
...     ConstantInput, RungeKutta4
... )
>>> # Define dynamics
>>> def pendulum(t, x, u, p):
...     theta, omega = x
...     g, L = p['g'], p['L']
...     return np.array([omega, -g/L * np.sin(theta)])
>>>
>>> # Setup simulation
>>> integrator = RungeKutta4(pendulum)
>>> engine = SimulationEngine(integrator)
>>>
>>> config = SimulationConfig(
...     t_span=(0, 10),
...     x0=np.array([np.pi/4, 0.0]),
...     inputs={'torque': ConstantInput(0.0)},
...     parameters={'g': 9.81, 'L': 1.0},
...     dt=0.01
... )
>>>
>>> result = engine.simulate(config)
>>> result.plot()
"""

# Core simulation components
from simulate.core import (
    SimulationEngine,
    SimulationConfig,
    Integrator,
    OutputFunction,
    InputSignal,
)

# Results
from simulate.results import SimulationResult

# Input signals
from simulate.inputs import (
    ConstantInput,
    StepInput,
    RampInput,
    InterpolatedInput,
    SinusoidalInput,
    FunctionInput,
    CompositeInput,
)

# Integrators
from simulate.integrators import (
    ForwardEuler,
    RungeKutta4,
    SciPyIntegrator,
    CasADiIntegrator,
    CasADiRK4,
    JAXIntegrator,
)

# Configuration setup utilities
from simulate.setup import read_param_values, read_param_values_pint

__all__ = [
    # Core
    "SimulationEngine",
    "SimulationConfig",
    "SimulationResult",
    "Integrator",
    "OutputFunction",
    "InputSignal",
    # Inputs
    "ConstantInput",
    "StepInput",
    "RampInput",
    "InterpolatedInput",
    "SinusoidalInput",
    "FunctionInput",
    "CompositeInput",
    # Integrators
    "ForwardEuler",
    "RungeKutta4",
    "SciPyIntegrator",
    "CasADiIntegrator",
    "CasADiRK4",
    "JAXIntegrator",
    # Setup utilities
    "read_param_values",
    "read_param_values_pint",
]
