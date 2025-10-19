"""Framework-specific integrator implementations.

This module provides integrator wrappers for various frameworks (CasADi,
JAX, SciPy, etc.) as well as simple fallback integrators for prototyping.

Users create integrators that wrap their framework's native solvers. The
integrator is responsible for advancing the state from time t to t+dt
using the framework's optimized solvers.
"""

from typing import Any, Callable, Dict, Optional

# ============================================================================
# Simple NumPy-based Integrators (for prototyping)
# ============================================================================


class ForwardEuler:
    """Simple forward Euler integrator.

    Best for prototyping and testing. Not recommended for production use.

    Parameters
    ----------
    dynamics_func : callable
        Function f(t, x, u, p) -> xdot that computes state derivatives

    Examples
    --------
    >>> def dynamics(t, x, u, p):
    ...     return -x  # Simple decay
    >>> integrator = ForwardEuler(dynamics)
    >>> x_next = integrator(t=0, x=np.array([1.0]), dt=0.1, u={}, p={})
    """

    def __init__(self, dynamics_func: Callable):
        self.dynamics = dynamics_func

    def __call__(self, t: float, x: Any, dt: float, u: Any, p: Any) -> Any:
        """Integrate from t to t+dt using forward Euler."""
        dx = self.dynamics(t, x, u, p)
        return x + dt * dx

    def __repr__(self):
        return "ForwardEuler()"


class RungeKutta4:
    """Classic 4th-order Runge-Kutta integrator.

    Good for prototyping with moderate accuracy.

    Parameters
    ----------
    dynamics_func : callable
        Function f(t, x, u, p) -> xdot that computes state derivatives

    Examples
    --------
    >>> def dynamics(t, x, u, p):
    ...     return -x
    >>> integrator = RungeKutta4(dynamics)
    >>> x_next = integrator(t=0, x=np.array([1.0]), dt=0.1, u={}, p={})
    """

    def __init__(self, dynamics_func: Callable):
        self.dynamics = dynamics_func

    def __call__(self, t: float, x: Any, dt: float, u: Any, p: Any) -> Any:
        """Integrate from t to t+dt using RK4."""
        k1 = self.dynamics(t, x, u, p)
        k2 = self.dynamics(t + dt / 2, x + dt / 2 * k1, u, p)
        k3 = self.dynamics(t + dt / 2, x + dt / 2 * k2, u, p)
        k4 = self.dynamics(t + dt, x + dt * k3, u, p)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def __repr__(self):
        return "RungeKutta4()"


# ============================================================================
# SciPy Integrators (for NumPy-based systems)
# ============================================================================


class SciPyIntegrator:
    """Wrapper for scipy.integrate.solve_ivp solvers.

    Uses SciPy's adaptive step-size solvers for accurate integration.
    Note that this still takes a single step of size dt, but internally
    may use adaptive sub-stepping.

    Parameters
    ----------
    dynamics_func : callable
        Function f(t, x, u, p) -> xdot
    method : str, optional
        scipy solve_ivp method: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF',
        'LSODA'. Default is 'RK45'.
    **solve_ivp_kwargs
        Additional keyword arguments passed to solve_ivp

    Examples
    --------
    >>> def dynamics(t, x, u, p):
    ...     return -p['decay_rate'] * x
    >>> integrator = SciPyIntegrator(dynamics, method='RK45', rtol=1e-6)
    >>> x_next = integrator(
    ...     t=0, x=np.array([1.0]), dt=0.1,
    ...     u={}, p={'decay_rate': 1.0}
    ... )
    """

    def __init__(
        self, dynamics_func: Callable, method: str = "RK45", **solve_ivp_kwargs
    ):
        self.dynamics = dynamics_func
        self.method = method
        self.kwargs = solve_ivp_kwargs

    def __call__(self, t: float, x: Any, dt: float, u: Any, p: Any) -> Any:
        """Integrate from t to t+dt using scipy.integrate.solve_ivp."""
        from scipy.integrate import solve_ivp

        # Define ODE with frozen inputs/parameters
        def ode(t_local, x_local):
            return self.dynamics(t_local, x_local, u, p)

        sol = solve_ivp(ode, (t, t + dt), x, method=self.method, **self.kwargs)

        return sol.y[:, -1]

    def __repr__(self):
        return f"SciPyIntegrator(method='{self.method}')"


# ============================================================================
# CasADi Integrators
# ============================================================================


class CasADiIntegrator:
    """Wrapper for CasADi's built-in integrators.

    Uses CasADi's optimized integrators (cvodes, idas, rk, collocation).
    Suitable for both ODE and DAE systems.

    Parameters
    ----------
    dynamics_func : casadi.Function
        CasADi function f(x, u, p) -> xdot.
        Should take state, inputs, and parameters and return derivatives.
    method : str, optional
        Integration method: 'cvodes', 'idas', 'rk', 'collocation'.
        Default is 'cvodes'.
    options : dict, optional
        Integrator options (e.g., abstol, reltol, max_num_steps)

    Notes
    -----
    The dynamics function should be a casadi.Function that takes three
    arguments: state (x), inputs (u), and parameters (p).

    Examples
    --------
    >>> import casadi as cas
    >>> # Define dynamics symbolically
    >>> x = cas.MX.sym('x', 2)
    >>> u = cas.MX.sym('u', 1)
    >>> p = cas.MX.sym('p', 2)
    >>> xdot = cas.vertcat(-p[0]*x[0], x[0] - p[1]*x[1])
    >>> dynamics = cas.Function('dynamics', [x, u, p], [xdot])
    >>> # Create integrator
    >>> integrator = CasADiIntegrator(
    ...     dynamics,
    ...     method='cvodes',
    ...     options={'abstol': 1e-8, 'reltol': 1e-6}
    ... )
    """

    def __init__(
        self,
        dynamics_func,
        method: str = "cvodes",
        options: Optional[Dict] = None,
    ):
        try:
            import casadi as cas
        except ImportError:
            raise ImportError(
                "CasADi not installed. Install with: pip install casadi"
            )

        self.dynamics_func = dynamics_func
        self.method = method

        # Build CasADi integrator
        # Get dimensions from function
        n_x = dynamics_func.size1_out(0)
        n_u = dynamics_func.size1_in(1)
        n_p = dynamics_func.size1_in(2)

        # Define symbolic variables
        x = cas.MX.sym("x", n_x)
        u = cas.MX.sym("u", n_u)
        p_sym = cas.MX.sym("p", n_p)

        # Define DAE for integrator
        # Note: CasADi integrator integrates from 0 to 1, we'll scale by dt
        dae = {
            "x": x,
            "p": cas.vertcat(u, p_sym),
            "ode": dynamics_func(x, u, p_sym),
        }

        opts = options or {}
        self.integrator = cas.integrator("integrator", method, dae, 0, 1, opts)

    def __call__(self, t: float, x: Any, dt: float, u: Any, p: Any) -> Any:
        """Integrate from t to t+dt using CasADi integrator."""
        import casadi as cas

        # Convert inputs and parameters to CasADi format
        if isinstance(u, dict):
            u_vec = cas.vertcat(*[u[key] for key in sorted(u.keys())])
        else:
            u_vec = u

        if isinstance(p, dict):
            p_vec = cas.vertcat(*[p[key] for key in sorted(p.keys())])
        else:
            p_vec = p

        # Combine inputs and parameters
        params = cas.vertcat(u_vec, p_vec)

        # Integrate (CasADi integrates from 0 to dt when we pass dt as grid)
        result = self.integrator(x0=x, p=params)

        return result["xf"]

    def __repr__(self):
        return f"CasADiIntegrator(method='{self.method}')"


class CasADiRK4:
    """Explicit RK4 integrator using CasADi operations.

    More efficient than generic RK4 for CasADi symbolic systems.

    Parameters
    ----------
    dynamics_func : casadi.Function
        CasADi function f(x, u, p) -> xdot

    Examples
    --------
    >>> import casadi as cas
    >>> x = cas.MX.sym('x', 2)
    >>> u = cas.MX.sym('u', 1)
    >>> p = cas.MX.sym('p', 1)
    >>> xdot = cas.vertcat(-p*x[0] + u, x[0] - x[1])
    >>> dynamics = cas.Function('f', [x, u, p], [xdot])
    >>> integrator = CasADiRK4(dynamics)
    """

    def __init__(self, dynamics_func):
        self.dynamics = dynamics_func

    def __call__(self, t: float, x: Any, dt: float, u: Any, p: Any) -> Any:
        """Integrate using explicit RK4."""
        import casadi as cas

        # Convert inputs/parameters to format expected by dynamics
        if isinstance(u, dict):
            u_vec = cas.vertcat(*[u[key] for key in sorted(u.keys())])
        else:
            u_vec = u

        if isinstance(p, dict):
            p_vec = cas.vertcat(*[p[key] for key in sorted(p.keys())])
        else:
            p_vec = p

        # RK4 algorithm
        k1 = self.dynamics(x, u_vec, p_vec)
        k2 = self.dynamics(x + dt / 2 * k1, u_vec, p_vec)
        k3 = self.dynamics(x + dt / 2 * k2, u_vec, p_vec)
        k4 = self.dynamics(x + dt * k3, u_vec, p_vec)

        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def __repr__(self):
        return "CasADiRK4()"


# ============================================================================
# JAX Integrators
# ============================================================================


class JAXIntegrator:
    """Wrapper for JAX-based ODE integrators.

    Can use diffrax (recommended) or simple Euler fallback.

    Parameters
    ----------
    dynamics_func : callable
        JAX function f(t, x, u, p) -> xdot
    method : str, optional
        Solver method if using diffrax: 'dopri5', 'dopri8', 'tsit5', etc.
        Default is 'dopri5'.
    **solver_kwargs
        Additional keyword arguments for diffrax solver

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def dynamics(t, x, u, p):
    ...     return -p['decay'] * x
    >>> integrator = JAXIntegrator(dynamics, method='dopri5')
    >>> x_next = integrator(
    ...     t=0, x=jnp.array([1.0]), dt=0.1,
    ...     u={}, p={'decay': 1.0}
    ... )
    """

    def __init__(
        self, dynamics_func: Callable, method: str = "dopri5", **solver_kwargs
    ):
        self.dynamics = dynamics_func
        self.method = method
        self.solver_kwargs = solver_kwargs

    def __call__(self, t: float, x: Any, dt: float, u: Any, p: Any) -> Any:
        """Integrate using JAX (with diffrax if available)."""
        try:
            # Try using diffrax if available
            import diffrax

            term = diffrax.ODETerm(
                lambda t_local, y, args: self.dynamics(t_local, y, u, p)
            )

            # Select solver
            solver_class = getattr(diffrax, self.method.capitalize(), None)
            if solver_class is None:
                # Try uppercase
                solver_class = getattr(diffrax, self.method.upper())
            solver = solver_class()

            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=t,
                t1=t + dt,
                dt0=dt,
                y0=x,
                **self.solver_kwargs,
            )
            return solution.ys[-1]

        except ImportError:
            # Fallback to simple Euler for JAX

            dx = self.dynamics(t, x, u, p)
            return x + dt * dx

    def __repr__(self):
        return f"JAXIntegrator(method='{self.method}')"
