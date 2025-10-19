"""Input signal classes for time-varying system inputs.

This module provides various input signal types that can be used
to drive dynamical systems during simulation.

Notes
-----
Input signals are evaluated by the SimulationEngine at discrete time
points and passed as constants to the integrator for each time step.
This means inputs are piecewise constant between time steps.

For true time-varying inputs within integration steps, users should
implement custom integrators specific to their framework.
"""

from typing import Any, Callable, Union

import numpy as np


class ConstantInput:
    """Constant input signal.

    Returns the same value at all times.

    Parameters
    ----------
    value : scalar or array-like
        Constant value to return

    Examples
    --------
    >>> u = ConstantInput(5.0)
    >>> u(0.0)
    5.0
    >>> u(10.0)
    5.0
    """

    def __init__(self, value: Any):
        self.value = value

    def __call__(self, t: float) -> Any:
        """Return constant value."""
        return self.value

    def __repr__(self):
        return f"ConstantInput(value={self.value})"


class StepInput:
    """Step input signal with piecewise constant values.

    The input changes value at specified times.

    Parameters
    ----------
    times : array-like
        Times at which the input changes value
    values : array-like
        Values corresponding to each time interval.
        Length should be len(times) + 1 or len(times).

    Examples
    --------
    >>> # Step from 0 to 1 at t=5
    >>> u = StepInput([5.0], [0.0, 1.0])
    >>> u(4.9)
    0.0
    >>> u(5.1)
    1.0

    >>> # Multiple steps
    >>> u = StepInput([0, 2, 5], [1.0, 2.0, 3.0])
    >>> u(1.0)  # t in [0, 2)
    1.0
    >>> u(3.0)  # t in [2, 5)
    2.0
    >>> u(6.0)  # t >= 5
    3.0
    """

    def __init__(
        self,
        times: Union[list, np.ndarray],
        values: Union[list, np.ndarray],
    ):
        self.times = np.asarray(times)
        self.values = np.asarray(values)

        if len(self.values) not in (len(self.times), len(self.times) + 1):
            raise ValueError(
                f"values must have length {len(self.times)} or "
                f"{len(self.times) + 1}, got {len(self.values)}"
            )

    def __call__(self, t: float) -> Any:
        """Return value at time t."""
        idx = np.searchsorted(self.times, t, side="right")
        return self.values[min(idx, len(self.values) - 1)]

    def __repr__(self):
        return (
            f"StepInput(times={self.times.tolist()}, "
            f"values={self.values.tolist()})"
        )


class RampInput:
    """Ramp input that changes linearly with time.

    Parameters
    ----------
    rate : float
        Rate of change (slope)
    offset : float, optional
        Initial value at t=0, by default 0.0

    Examples
    --------
    >>> u = RampInput(rate=2.0, offset=1.0)
    >>> u(0.0)
    1.0
    >>> u(1.0)
    3.0
    >>> u(2.0)
    5.0
    """

    def __init__(self, rate: float, offset: float = 0.0):
        self.rate = rate
        self.offset = offset

    def __call__(self, t: float) -> float:
        """Return ramp value at time t."""
        return self.offset + self.rate * t

    def __repr__(self):
        return f"RampInput(rate={self.rate}, offset={self.offset})"


class InterpolatedInput:
    """Linearly interpolated input from tabulated data.

    Parameters
    ----------
    times : array-like
        Time points for interpolation
    values : array-like
        Values at each time point
    kind : str, optional
        Interpolation kind ('linear', 'cubic', etc.), by default 'linear'
    fill_value : str or float, optional
        How to handle extrapolation, by default 'extrapolate'

    Examples
    --------
    >>> times = [0.0, 1.0, 2.0, 3.0]
    >>> values = [0.0, 1.0, 0.5, 0.0]
    >>> u = InterpolatedInput(times, values)
    >>> u(0.5)  # Linear interpolation
    0.5
    >>> u(1.5)
    0.75
    """

    def __init__(
        self,
        times: Union[list, np.ndarray],
        values: Union[list, np.ndarray],
        kind: str = "linear",
        fill_value: Union[str, float] = "extrapolate",
    ):
        from scipy.interpolate import interp1d

        self.times = np.asarray(times)
        self.values = np.asarray(values)
        self.kind = kind

        self.interp = interp1d(
            self.times, self.values, kind=kind, fill_value=fill_value
        )

    def __call__(self, t: float) -> Any:
        """Return interpolated value at time t."""
        return float(self.interp(t))

    def __repr__(self):
        return (
            f"InterpolatedInput(kind='{self.kind}', "
            f"n_points={len(self.times)})"
        )


class SinusoidalInput:
    """Sinusoidal input signal.

    u(t) = amplitude * sin(2*pi*frequency*t + phase) + offset

    Parameters
    ----------
    amplitude : float
        Amplitude of sine wave
    frequency : float
        Frequency in Hz
    phase : float, optional
        Phase offset in radians, by default 0.0
    offset : float, optional
        DC offset, by default 0.0

    Examples
    --------
    >>> u = SinusoidalInput(amplitude=1.0, frequency=1.0)
    >>> u(0.0)
    0.0
    >>> abs(u(0.25) - 1.0) < 1e-10  # Peak at quarter period
    True
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        offset: float = 0.0,
    ):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    def __call__(self, t: float) -> float:
        """Return sinusoidal value at time t."""
        return (
            self.amplitude
            * np.sin(2 * np.pi * self.frequency * t + self.phase)
            + self.offset
        )

    def __repr__(self):
        return (
            f"SinusoidalInput(amplitude={self.amplitude}, "
            f"frequency={self.frequency}, phase={self.phase}, "
            f"offset={self.offset})"
        )


class FunctionInput:
    """Custom function-based input.

    Wraps any callable as an input signal.

    Parameters
    ----------
    func : callable
        Function that takes time t and returns input value

    Examples
    --------
    >>> # Exponential decay
    >>> u = FunctionInput(lambda t: np.exp(-t))
    >>> u(0.0)
    1.0
    >>> abs(u(1.0) - np.exp(-1)) < 1e-10
    True

    >>> # Piecewise function
    >>> def custom(t):
    ...     if t < 5:
    ...         return t**2
    ...     else:
    ...         return 25 - (t-5)
    >>> u = FunctionInput(custom)
    >>> u(3.0)
    9.0
    >>> u(6.0)
    24.0
    """

    def __init__(self, func: Callable[[float], Any]):
        self.func = func

    def __call__(self, t: float) -> Any:
        """Return function value at time t."""
        return self.func(t)

    def __repr__(self):
        func_name = getattr(self.func, "__name__", repr(self.func))
        return f"FunctionInput(func={func_name})"


class CompositeInput:
    """Composite input formed by combining multiple signals.

    Supports addition, multiplication, and custom combinations.

    Parameters
    ----------
    signals : list of InputSignal
        Input signals to combine
    operation : callable, optional
        Function to combine signals. Takes list of values, returns combined
        value. Default is sum.

    Examples
    --------
    >>> # Sum of two signals
    >>> u1 = ConstantInput(1.0)
    >>> u2 = SinusoidalInput(amplitude=0.5, frequency=1.0)
    >>> u_sum = CompositeInput([u1, u2])
    >>> u_sum(0.0)
    1.0

    >>> # Custom combination (product)
    >>> u_prod = CompositeInput(
    ...     [u1, u2],
    ...     operation=lambda vals: vals[0] * vals[1]
    ... )
    """

    def __init__(self, signals: list, operation: Callable[[list], Any] = None):
        self.signals = signals
        self.operation = operation or sum

    def __call__(self, t: float) -> Any:
        """Return combined value at time t."""
        values = [sig(t) for sig in self.signals]
        return self.operation(values)

    def __repr__(self):
        return f"CompositeInput(n_signals={len(self.signals)})"
