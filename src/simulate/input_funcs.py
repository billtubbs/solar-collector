"""Callable function classes for defining input signals and profiles.

This module provides various callable function types that can be used
to define signals as a function of any independent variable (e.g.,
time, position, or other coordinates).

Notes
-----
When used with SimulationEngine, these functions are evaluated at
discrete time points and passed as constants to the integrator for
each time step, making inputs piecewise constant between time steps.

These classes can also be used independently to define spatial
profiles (e.g., initial temperature distributions as a function
of position) or any other function of a single variable.
"""

from typing import Any, Callable, Union

import numpy as np


class ConstantInput:
    """Constant-valued function.

    Returns the same value for all inputs.

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

    def __call__(self, x: float) -> Any:
        """Return constant value."""
        return self.value

    def __repr__(self):
        return f"ConstantInput(value={self.value})"


class StepInput:
    """Piecewise constant (step) function.

    The output changes value at specified breakpoints.

    Parameters
    ----------
    times : array-like
        Breakpoints at which the value changes.
    values : array-like
        Values for each interval.
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

    def __call__(self, x: float) -> Any:
        """Return value at x."""
        idx = np.searchsorted(self.times, x, side="right")
        return self.values[min(idx, len(self.values) - 1)]

    def __repr__(self):
        return (
            f"StepInput(times={self.times.tolist()}, "
            f"values={self.values.tolist()})"
        )


class RampInput:
    """Linear ramp function.

    Parameters
    ----------
    rate : float
        Rate of change (slope).
    offset : float, optional
        Value at the origin, by default 0.0.

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

    def __call__(self, x: float) -> float:
        """Return ramp value at x."""
        return self.offset + self.rate * x

    def __repr__(self):
        return f"RampInput(rate={self.rate}, offset={self.offset})"


class InterpolatedInput:
    """Interpolated function from tabulated data points.

    Parameters
    ----------
    points : array-like
        Independent variable values (e.g., time or position).
    values : array-like
        Dependent variable values at each point.
    kind : str, optional
        Interpolation kind ('linear', 'cubic', etc.), by default 'linear'.
    fill_value : str or float, optional
        How to handle extrapolation, by default 'extrapolate'.

    Examples
    --------
    >>> points = [0.0, 1.0, 2.0, 3.0]
    >>> values = [0.0, 1.0, 0.5, 0.0]
    >>> u = InterpolatedInput(points, values)
    >>> u(0.5)  # Linear interpolation
    0.5
    >>> u(1.5)
    0.75
    """

    def __init__(
        self,
        points: Union[list, np.ndarray] = None,
        values: Union[list, np.ndarray] = None,
        kind: str = "linear",
        fill_value: Union[str, float] = "extrapolate",
        *,
        times: Union[list, np.ndarray] = None,
    ):
        from scipy.interpolate import interp1d

        # Support 'times' as alias for backwards compatibility
        if points is None and times is not None:
            points = times
        elif points is None:
            raise TypeError(
                "InterpolatedInput requires 'points' (or 'times')"
            )

        self.times = np.asarray(points)
        self.values = np.asarray(values)
        self.kind = kind

        self.interp = interp1d(
            self.times, self.values, kind=kind, fill_value=fill_value
        )

    def __call__(self, x: float) -> Any:
        """Return interpolated value at x."""
        return float(self.interp(x))

    def __repr__(self):
        return (
            f"InterpolatedInput(kind='{self.kind}', "
            f"n_points={len(self.times)})"
        )


class SinusoidalInput:
    """Sinusoidal function.

    f(x) = amplitude * sin(2*pi*frequency*x + phase) + offset

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

    def __call__(self, x: float) -> float:
        """Return sinusoidal value at x."""
        return (
            self.amplitude
            * np.sin(2 * np.pi * self.frequency * x + self.phase)
            + self.offset
        )

    def __repr__(self):
        return (
            f"SinusoidalInput(amplitude={self.amplitude}, "
            f"frequency={self.frequency}, phase={self.phase}, "
            f"offset={self.offset})"
        )


class FunctionInput:
    """Custom function wrapper.

    Wraps any callable as an input function.

    Parameters
    ----------
    func : callable
        Function that takes a single argument and returns a value.

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

    def __call__(self, x: float) -> Any:
        """Return function value at x."""
        return self.func(x)

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

    def __call__(self, x: float) -> Any:
        """Return combined value at x."""
        values = [sig(x) for sig in self.signals]
        return self.operation(values)

    def __repr__(self):
        return f"CompositeInput(n_signals={len(self.signals)})"
