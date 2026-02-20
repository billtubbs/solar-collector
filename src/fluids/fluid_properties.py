import json
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


class PropertyCorrelation(ABC):
    """
    Abstract base class for property correlations
    """

    @abstractmethod
    def evaluate(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate property at temperature T"""
        pass

    @abstractmethod
    def get_valid_range(self) -> Tuple[float, float]:
        """Return valid temperature range (T_min, T_max)"""
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """Serialize correlation to dictionary"""
        pass


def _shared_fitting_solver(
    model_func: Callable,
    params: List[ca.MX],
    T_data: np.ndarray,
    y_data: np.ndarray,
    weights: np.ndarray = None,
    bounds: Dict = None,
    initial_guess: Dict = None,
) -> Tuple[Dict, Dict]:
    """
    Shared CasADi-based optimization for parameter fitting

    Parameters:
    -----------
    model_func : callable
        Function that takes (T, *parameters) and returns predicted values
    parameters : list of ca.MX
        CasADi variables for optimization parameters
    T_data : np.ndarray
        Temperature data points
    y_data : np.ndarray
        Property data points
    weights : np.ndarray, optional
        Weights for data points (default: equal weights)
    bounds : dict, optional
        Parameter bounds {param_name: (lower, upper)}
    initial_guess : dict, optional
        Initial parameter values {param_name: value}

    Returns:
    --------
    tuple
        (optimal_params, fit_stats)
    """
    n_data = len(T_data)

    if weights is None:
        weights = np.ones(n_data)

    # Create parameter vector
    param_vec = ca.vertcat(*params)
    param_names = [p.name() for p in params]
    n_params = len(params)

    # Create function for model evaluation
    T_sym = ca.MX.sym("T")
    y_model = model_func(T_sym, *params)
    model_func_ca = ca.Function("model", [T_sym, param_vec], [y_model])

    # Create objective function using vectorized operations
    T_data_ca = ca.DM(T_data)
    y_data_ca = ca.DM(y_data)
    weights_ca = ca.DM(weights)

    # Evaluate model for all data points
    y_pred_vec = model_func_ca.map(n_data)(
        T_data_ca, ca.repmat(param_vec, 1, n_data)
    )

    # Calculate weighted sum of squares
    residuals = y_pred_vec.T - y_data_ca  # Transpose to match dimensions
    objective = ca.sum1(weights_ca * residuals**2)

    # Create optimization problem
    nlp = {"x": param_vec, "f": objective}

    # Solver options
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Set bounds and initial guess
    x0 = np.zeros(n_params)
    lbx = -np.inf * np.ones(n_params)
    ubx = np.inf * np.ones(n_params)

    for i, name in enumerate(param_names):
        if initial_guess and name in initial_guess:
            x0[i] = initial_guess[name]
        if bounds and name in bounds:
            lbx[i], ubx[i] = bounds[name]

    # Solve optimization
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)

    # Extract results
    optimal_values = np.array(sol["x"]).flatten()
    optimal_params = dict(zip(param_names, optimal_values))

    # Calculate fit statistics
    y_pred_values = np.array(
        model_func_ca.map(n_data)(T_data_ca, ca.repmat(sol["x"], 1, n_data))
    ).flatten()

    residuals = y_data - y_pred_values
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / n_data)
    mae = np.mean(np.abs(residuals))

    fit_stats = {
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "objective_value": float(sol["f"]),
        "solver_status": solver.stats()["return_status"],
    }

    return optimal_params, fit_stats


class FittedPropertyCorrelation(PropertyCorrelation):
    """
    Abstract base class for property correlations with fitting capabilities

    Extends PropertyCorrelation to add parameter fitting functionality using CasADi
    """

    @abstractmethod
    def fit(
        self,
        T_data: np.ndarray,
        y_data: np.ndarray,
        weights: np.ndarray = None,
        bounds: Dict = None,
        initial_guess: Dict = None,
    ) -> Dict:
        """
        Fit correlation parameters to data

        Parameters:
        -----------
        T_data : np.ndarray
            Temperature data points [K]
        y_data : np.ndarray
            Property data points
        weights : np.ndarray, optional
            Weights for data points
        bounds : dict, optional
            Parameter bounds {param_name: (lower, upper)}
        initial_guess : dict, optional
            Initial parameter values {param_name: value}

        Returns:
        --------
        dict
            Fitting statistics and results
        """
        pass

    @abstractmethod
    def get_param_names(self) -> List[str]:
        """Get list of parameter names for this correlation"""
        pass

    @abstractmethod
    def update_params(self, params: Dict) -> None:
        """Update correlation parameters from fitted values"""
        pass

    def plot_correlation(
        self,
        T_data: np.ndarray,
        y_data: np.ndarray,
        title: str = None,
        xlabel: str = "Temperature [°C]",
        ylabel: str = "Property Value",
        s: float = 25,
        show_residuals: bool = True,
        figsize=None,
    ) -> tuple:
        """
        Plot correlation model predictions versus experimental data

        Parameters:
        -----------
        T_data : np.ndarray
            Temperature data points [K]
        y_data : np.ndarray
            Property data points
        title : str, optional
            Plot title (default: uses correlation name)
        xlabel : str, optional
            X-axis label (default: "Temperature [°C]")
        ylabel : str, optional
            Y-axis label (default: "Property Value")
        show_residuals : bool, optional
            Whether to show residuals subplot (default: True)

        Returns:
        --------
        tuple
            (fig, axes) - matplotlib figure and axes objects
            If show_residuals=True: axes is (ax1, ax2) tuple
            If show_residuals=False: axes is single axis object
        """
        import matplotlib.pyplot as plt

        # Convert inputs to numpy arrays
        T_data = np.array(T_data)
        y_data = np.array(y_data)
        T_C_data = T_data - 273.15

        # Generate model predictions
        y_pred = self.evaluate(T_data)

        # Calculate statistics
        residuals = y_data - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        mape = (
            np.mean(np.abs(residuals / y_data)) * 100
            if np.all(y_data != 0)
            else float("inf")
        )

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("-inf")

        # Create smooth curve for model
        T_smooth = np.linspace(T_data.min(), T_data.max(), 200)
        T_C_smooth = T_smooth - 273.15
        y_smooth = self.evaluate(T_smooth)

        # Set up plot
        if show_residuals:
            if figsize is None:
                figsize = (7, 4)
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
            ax1, ax2 = axes
        else:
            if figsize is None:
                figsize = (7, 2.5)
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            ax1 = axes

        # Main plot: data vs model
        ax = ax1
        ax.scatter(
            T_C_data,
            y_data,
            color="C0",
            alpha=0.7,
            s=s,
            label="Data",
            zorder=3,
        )
        ax.plot(
            T_C_smooth,
            y_smooth,
            color="C1",
            linewidth=2,
            label="Correlation Model",
            zorder=2,
        )

        # Formatting
        if not show_residuals:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Title
        if title is None:
            title = f"{getattr(self, 'name', 'Correlation')} - Model vs Data"
        ax.set_title(title)

        # Add statistics text box
        stats_text = (
            f"R² = {r_squared:.4f}\n"
            f"RMSE = {rmse:.3e}\n"
            f"MAE = {mae:.3e}\n"
            f"MAPE = {mape:.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Residuals plot (using valid data only)
        if show_residuals:
            ax = ax2
            ax.scatter(T_C_data, residuals, color="C0", alpha=0.7, s=s)
            ax.axhline(y=0, color="C1", linestyle="--")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Residuals (Data - Model)")
            ax.grid(True, alpha=0.3)
            ax.set_title("Residuals Plot")

        return fig, axes


class PolynomialCorrelation(FittedPropertyCorrelation):
    """
    Polynomial correlation: f(T) = a0 + a1*T + a2*T^2 + ... + an*T^n

    This corresponds to DIPPR correlation equation 100:
    Y = A + B*T + C*T² + D*T³ + E*T⁴

    Commonly used for:
    - Liquid heat capacity (DIPPR100)
    - Liquid density (DIPPR100)
    - Thermal conductivity (DIPPR100)
    """

    def __init__(
        self,
        coefficients: List[float],
        T_min: float,
        T_max: float,
        name: str = "Polynomial",
    ):
        self.coefficients = np.array(coefficients)
        self.T_min = T_min
        self.T_max = T_max
        self.name = name

    def evaluate(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        T = np.atleast_1d(T)
        result = np.polyval(self.coefficients[::-1], T)
        return float(result[0]) if len(result) == 1 else result

    def get_valid_range(self) -> Tuple[float, float]:
        return (self.T_min, self.T_max)

    def to_dict(self) -> Dict:
        return {
            "type": "polynomial",
            "coefficients": self.coefficients.tolist(),
            "T_min": self.T_min,
            "T_max": self.T_max,
            "name": self.name,
        }

    def get_param_names(self) -> List[str]:
        """Get list of parameter names for polynomial correlation"""
        return [f"a{i}" for i in range(len(self.coefficients))]

    def update_params(self, params: Dict) -> None:
        """Update polynomial coefficients from fitted values"""
        n_coeffs = len(self.coefficients)
        new_coeffs = np.zeros(n_coeffs)
        for i in range(n_coeffs):
            param_name = f"a{i}"
            if param_name in params:
                new_coeffs[i] = params[param_name]
        self.coefficients = new_coeffs

    def fit(
        self,
        T_data: np.ndarray,
        y_data: np.ndarray,
        weights: np.ndarray = None,
        bounds: Dict = None,
        initial_guess: Dict = None,
    ) -> Dict:
        """
        Fit polynomial coefficients to data using CasADi optimization

        Parameters:
        -----------
        T_data : np.ndarray
            Temperature data points [K]
        y_data : np.ndarray
            Property data points
        weights : np.ndarray, optional
            Weights for data points
        bounds : dict, optional
            Parameter bounds {param_name: (lower, upper)}
        initial_guess : dict, optional
            Initial parameter values {param_name: value}

        Returns:
        --------
        dict
            Fitting statistics and results
        """
        n_coeffs = len(self.coefficients)

        # Create CasADi parameter variables
        coeffs = [ca.MX.sym(f"a{i}") for i in range(n_coeffs)]

        # Define polynomial model function
        def polynomial_model(T, *params):
            result = 0
            for i, coeff in enumerate(params):
                result += coeff * T**i
            return result

        # Use shared fitting solver
        optimal_params, fit_stats = _shared_fitting_solver(
            polynomial_model,
            coeffs,
            T_data,
            y_data,
            weights,
            bounds,
            initial_guess,
        )

        # Update correlation parameters
        self.update_params(optimal_params)

        # Add parameter values to fit stats
        fit_stats.update(
            {
                "fitted_parameters": optimal_params,
                "correlation_type": "polynomial",
                "n_parameters": n_coeffs,
            }
        )

        return fit_stats


class ExponentialCorrelation(FittedPropertyCorrelation):
    """
    Exponential correlation: f(T) = A * exp(B/T) + C

    This implements a simplified form of DIPPR correlation equation 101:
    Y = exp(A + B/T + C*ln(T) + D*T^E)

    Commonly used for:
    - Dynamic viscosity (DIPPR101 - Andrade equation)
    - Vapor pressure (DIPPR101 - Antoine-type)
    - Surface tension (DIPPR101)

    Note: This simplified form (A*exp(B/T) + C) is often used for
    viscosity correlations where the Andrade equation applies.
    """

    def __init__(
        self,
        A: float,
        B: float,
        C: float,
        T_min: float,
        T_max: float,
        name: str = "Exponential",
    ):
        self.A = A
        self.B = B
        self.C = C
        self.T_min = T_min
        self.T_max = T_max
        self.name = name

    def evaluate(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self.A * np.exp(self.B / T) + self.C

    def get_valid_range(self) -> Tuple[float, float]:
        return (self.T_min, self.T_max)

    def to_dict(self) -> Dict:
        return {
            "type": "exponential",
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "name": self.name,
        }

    def get_param_names(self) -> List[str]:
        """Get list of parameter names for exponential correlation"""
        return ["A", "B", "C"]

    def update_params(self, params: Dict) -> None:
        """Update exponential correlation parameters from fitted values"""
        if "A" in params:
            self.A = params["A"]
        if "B" in params:
            self.B = params["B"]
        if "C" in params:
            self.C = params["C"]

    def fit(
        self,
        T_data: np.ndarray,
        y_data: np.ndarray,
        weights: np.ndarray = None,
        bounds: Dict = None,
        initial_guess: Dict = None,
    ) -> Dict:
        """
        Fit exponential correlation parameters to data using CasADi
        optimization.

        Parameters:
        -----------
        T_data : np.ndarray
            Temperature data points [K]
        y_data : np.ndarray
            Property data points
        weights : np.ndarray, optional
            Weights for data points
        bounds : dict, optional
            Parameter bounds {param_name: (lower, upper)}
        initial_guess : dict, optional
            Initial parameter values {param_name: value}

        Returns:
        --------
        dict
            Fitting statistics and results
        """
        # Create CasADi parameter variables
        A = ca.MX.sym("A")
        B = ca.MX.sym("B")
        C = ca.MX.sym("C")

        # Define exponential model function
        def exponential_model(T, A_param, B_param, C_param):
            return A_param * ca.exp(B_param / T) + C_param

        # Use shared fitting solver
        optimal_params, fit_stats = _shared_fitting_solver(
            exponential_model,
            [A, B, C],
            T_data,
            y_data,
            weights,
            bounds,
            initial_guess,
        )

        # Update correlation parameters
        self.update_params(optimal_params)

        # Add parameter values to fit stats
        fit_stats.update(
            {
                "fitted_parameters": optimal_params,
                "correlation_type": "exponential",
                "n_parameters": 3,
            }
        )

        return fit_stats


class AntoineCorrelation(FittedPropertyCorrelation):
    """
    Antoine equation for vapor pressure: log10(P) = A - B/(T + C)

    This is a classical form of DIPPR correlation equation 101:
    Y = exp(A + B/T + C*ln(T) + D*T^E)

    Specifically for vapor pressure (DIPPR101):
    - A, B, C are Antoine constants
    - T is temperature [K]
    - P is pressure [Pa]

    The Antoine equation is widely used for:
    - Vapor pressure of pure components
    - Boiling point calculations
    - Phase equilibrium modeling

    Note: This implementation uses T in Kelvin and returns P in Pascal
    """

    def __init__(
        self,
        A: float,
        B: float,
        C: float,
        T_min: float,
        T_max: float,
        name: str = "Antoine",
    ):
        self.A = A
        self.B = B
        self.C = C
        self.T_min = T_min
        self.T_max = T_max
        self.name = name

    def evaluate(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # Convert from Kelvin to Celsius for Antoine equation
        T_C = T - 273.15
        log10_P = self.A - self.B / (self.C + T_C)

        return 10**log10_P

    def get_valid_range(self) -> Tuple[float, float]:
        return (self.T_min, self.T_max)

    def to_dict(self) -> Dict:
        return {
            "type": "antoine",
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "name": self.name,
        }

    def get_param_names(self) -> List[str]:
        """Get list of parameter names for Antoine correlation"""
        return ["A", "B", "C"]

    def update_params(self, params: Dict) -> None:
        """Update Antoine correlation parameters from fitted values"""
        if "A" in params:
            self.A = params["A"]
        if "B" in params:
            self.B = params["B"]
        if "C" in params:
            self.C = params["C"]

    def fit(
        self,
        T_data: np.ndarray,
        y_data: np.ndarray,
        weights: np.ndarray = None,
        bounds: Dict = None,
        initial_guess: Dict = None,
    ) -> Dict:
        """
        Fit Antoine equation parameters to data using CasADi optimization

        Parameters:
        -----------
        T_data : np.ndarray
            Temperature data points [K]
        y_data : np.ndarray
            Pressure data points [Pa]
        weights : np.ndarray, optional
            Weights for data points
        bounds : dict, optional
            Parameter bounds {param_name: (lower, upper)}
        initial_guess : dict, optional
            Initial parameter values {param_name: value}

        Returns:
        --------
        dict
            Fitting statistics and results
        """
        # Create CasADi parameter variables
        A = ca.MX.sym("A")
        B = ca.MX.sym("B")
        C = ca.MX.sym("C")

        # Filter out zero vapor pressure data points (due to rounding)
        # Antoine equation requires log(P), so P must be > 0
        valid_mask = y_data > 0
        T_data_filtered = T_data[valid_mask]
        y_data_filtered = y_data[valid_mask]
        weights_filtered = weights[valid_mask] if weights is not None else None

        # Define Antoine model function
        def antoine_model(T, A_param, B_param, C_param):
            T_C = T - 273.15  # Convert to Celsius
            log10_P = A_param - B_param / (C_param + T_C)
            return 10**log10_P

        # Use shared fitting solver with filtered data
        optimal_params, fit_stats = _shared_fitting_solver(
            antoine_model,
            [A, B, C],
            T_data_filtered,
            y_data_filtered,
            weights_filtered,
            bounds,
            initial_guess,
        )

        # Update correlation parameters
        self.update_params(optimal_params)

        # Add parameter values to fit stats
        fit_stats.update(
            {
                "fitted_parameters": optimal_params,
                "correlation_type": "antoine",
                "n_parameters": 3,
            }
        )

        return fit_stats


class PowerLawCorrelation(FittedPropertyCorrelation):
    """
    Power law: f(T) = A * T^n + B
    """

    def __init__(
        self,
        A: float,
        n: float,
        B: float,
        T_min: float,
        T_max: float,
        name: str = "PowerLaw",
    ):
        self.A = A
        self.n = n
        self.B = B
        self.T_min = T_min
        self.T_max = T_max
        self.name = name

    def evaluate(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self.A * T**self.n + self.B

    def get_valid_range(self) -> Tuple[float, float]:
        return (self.T_min, self.T_max)

    def to_dict(self) -> Dict:
        return {
            "type": "powerlaw",
            "A": self.A,
            "n": self.n,
            "B": self.B,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "name": self.name,
        }

    def get_param_names(self) -> List[str]:
        """Get list of parameter names for power law correlation"""
        return ["A", "n", "B"]

    def update_params(self, params: Dict) -> None:
        """Update power law correlation parameters from fitted values"""
        if "A" in params:
            self.A = params["A"]
        if "n" in params:
            self.n = params["n"]
        if "B" in params:
            self.B = params["B"]

    def fit(
        self,
        T_data: np.ndarray,
        y_data: np.ndarray,
        weights: np.ndarray = None,
        bounds: Dict = None,
        initial_guess: Dict = None,
    ) -> Dict:
        """
        Fit power law correlation parameters to data using CasADi optimization

        Parameters:
        -----------
        T_data : np.ndarray
            Temperature data points [K]
        y_data : np.ndarray
            Property data points
        weights : np.ndarray, optional
            Weights for data points
        bounds : dict, optional
            Parameter bounds {param_name: (lower, upper)}
        initial_guess : dict, optional
            Initial parameter values {param_name: value}

        Returns:
        --------
        dict
            Fitting statistics and results
        """
        # Create CasADi parameter variables
        A = ca.MX.sym("A")
        n = ca.MX.sym("n")
        B = ca.MX.sym("B")

        # Define power law model function
        def powerlaw_model(T, A_param, n_param, B_param):
            return A_param * T**n_param + B_param

        # Use shared fitting solver
        optimal_params, fit_stats = _shared_fitting_solver(
            powerlaw_model,
            [A, n, B],
            T_data,
            y_data,
            weights,
            bounds,
            initial_guess,
        )

        # Update correlation parameters
        self.update_params(optimal_params)

        # Add parameter values to fit stats
        fit_stats.update(
            {
                "fitted_parameters": optimal_params,
                "correlation_type": "powerlaw",
                "n_parameters": 3,
            }
        )

        return fit_stats


class TableCorrelation(PropertyCorrelation):
    """
    Table-based correlation with linear interpolation
    """

    def __init__(
        self,
        T_data: np.ndarray,
        property_data: np.ndarray,
        name: str = "Table",
    ):
        self.T_data = np.array(T_data)
        self.property_data = np.array(property_data)
        self.T_min = self.T_data.min()
        self.T_max = self.T_data.max()
        self.name = name

    def evaluate(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return np.interp(T, self.T_data, self.property_data)

    def get_valid_range(self) -> Tuple[float, float]:
        return (self.T_min, self.T_max)

    def to_dict(self) -> Dict:
        return {
            "type": "table",
            "T_data": self.T_data.tolist(),
            "property_data": self.property_data.tolist(),
            "name": self.name,
        }


class CustomCorrelation(PropertyCorrelation):
    """
    Custom user-defined function
    """

    def __init__(
        self,
        func: Callable,
        T_min: float,
        T_max: float,
        name: str = "Custom",
        description: str = "",
    ):
        self.func = func
        self.T_min = T_min
        self.T_max = T_max
        self.name = name
        self.description = description

    def evaluate(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self.func(T)

    def get_valid_range(self) -> Tuple[float, float]:
        return (self.T_min, self.T_max)

    def to_dict(self) -> Dict:
        return {
            "type": "custom",
            "description": self.description,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "name": self.name,
        }


class FluidProperties:
    """
    Complete fluid property database with temperature-dependent correlations

    Properties:
    - density (rho) [kg/m³]
    - heat_capacity (cp) [J/kg·K]
    - thermal_conductivity (k) [W/m·K]
    - dynamic_viscosity (mu) [Pa·s]
    - vapor_pressure (Pv) [Pa]
    """

    def __init__(self, name: str, molecular_weight: float = None):
        self.name = name
        self.molecular_weight = molecular_weight  # kg/kmol

        # Property correlations
        self.correlations: Dict[str, PropertyCorrelation] = {}

        # Reference conditions
        self.T_reference = 298.15  # K
        self.P_reference = 101325  # Pa

    def set_density(self, correlation: PropertyCorrelation):
        """Set density correlation"""
        self.correlations["density"] = correlation

    def set_heat_capacity(self, correlation: PropertyCorrelation):
        """Set heat capacity correlation"""
        self.correlations["heat_capacity"] = correlation

    def set_thermal_conductivity(self, correlation: PropertyCorrelation):
        """Set thermal conductivity correlation"""
        self.correlations["thermal_conductivity"] = correlation

    def set_viscosity(self, correlation: PropertyCorrelation):
        """Set dynamic viscosity correlation"""
        self.correlations["viscosity"] = correlation

    def set_vapor_pressure(self, correlation: PropertyCorrelation):
        """Set vapor pressure correlation"""
        self.correlations["vapor_pressure"] = correlation

    def density(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Get density at temperature T [K]"""
        if "density" not in self.correlations:
            raise ValueError(
                f"Density correlation not defined for {self.name}"
            )
        return self.correlations["density"].evaluate(T)

    def heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get heat capacity at temperature T [K]"""
        if "heat_capacity" not in self.correlations:
            raise ValueError(
                f"Heat capacity correlation not defined for {self.name}"
            )
        return self.correlations["heat_capacity"].evaluate(T)

    def thermal_conductivity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get thermal conductivity at temperature T [K]"""
        if "thermal_conductivity" not in self.correlations:
            raise ValueError(
                f"Thermal conductivity correlation not defined for {self.name}"
            )
        return self.correlations["thermal_conductivity"].evaluate(T)

    def viscosity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get dynamic viscosity at temperature T [K]"""
        if "viscosity" not in self.correlations:
            raise ValueError(
                f"Viscosity correlation not defined for {self.name}"
            )
        return self.correlations["viscosity"].evaluate(T)

    def vapor_pressure(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get vapor pressure at temperature T [K]"""
        if "vapor_pressure" not in self.correlations:
            raise ValueError(
                f"Vapor pressure correlation not defined for {self.name}"
            )
        return self.correlations["vapor_pressure"].evaluate(T)

    def kinematic_viscosity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get kinematic viscosity [m²/s]"""
        return self.viscosity(T) / self.density(T)

    def thermal_diffusivity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get thermal diffusivity [m²/s]"""
        return self.thermal_conductivity(T) / (
            self.density(T) * self.heat_capacity(T)
        )

    def prandtl_number(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get Prandtl number"""
        return (
            self.viscosity(T)
            * self.heat_capacity(T)
            / self.thermal_conductivity(T)
        )

    def get_all_properties(self, T: Union[float, np.ndarray]) -> Dict:
        """Get all properties at temperature T"""
        props = {}

        if "density" in self.correlations:
            props["density"] = self.density(T)
        if "heat_capacity" in self.correlations:
            props["heat_capacity"] = self.heat_capacity(T)
        if "thermal_conductivity" in self.correlations:
            props["thermal_conductivity"] = self.thermal_conductivity(T)
        if "viscosity" in self.correlations:
            props["viscosity"] = self.viscosity(T)
        if "vapor_pressure" in self.correlations:
            props["vapor_pressure"] = self.vapor_pressure(T)

        # Derived properties
        if "density" in props and "viscosity" in props:
            props["kinematic_viscosity"] = (
                props["viscosity"] / props["density"]
            )
        if all(
            k in props
            for k in ["thermal_conductivity", "density", "heat_capacity"]
        ):
            props["thermal_diffusivity"] = props["thermal_conductivity"] / (
                props["density"] * props["heat_capacity"]
            )
        if all(
            k in props
            for k in ["viscosity", "heat_capacity", "thermal_conductivity"]
        ):
            props["prandtl_number"] = (
                props["viscosity"]
                * props["heat_capacity"]
                / props["thermal_conductivity"]
            )

        return props

    def plot_properties(
        self, T_range: Tuple[float, float] = None, n_points: int = 100
    ):
        """Plot all properties vs temperature"""

        if T_range is None:
            # Find common valid range
            ranges = [
                corr.get_valid_range() for corr in self.correlations.values()
            ]
            T_min = max(r[0] for r in ranges)
            T_max = min(r[1] for r in ranges)
            T_range = (T_min, T_max)

        T = np.linspace(T_range[0], T_range[1], n_points)

        # Count available properties
        n_props = len(self.correlations)
        if n_props == 0:
            print("No properties defined")
            return

        # Create subplots
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        plot_idx = 0
        prop_info = [
            ("density", "Density", "kg/m³"),
            ("heat_capacity", "Heat Capacity", "J/kg·K"),
            ("thermal_conductivity", "Thermal Conductivity", "W/m·K"),
            ("viscosity", "Dynamic Viscosity", "Pa·s"),
            ("vapor_pressure", "Vapor Pressure", "Pa"),
        ]

        for prop_name, title, units in prop_info:
            if prop_name in self.correlations:
                row = plot_idx // n_cols
                col = plot_idx % n_cols
                ax = axes[row, col]

                try:
                    values = self.correlations[prop_name].evaluate(T)
                    ax.plot(T - 273.15, values, "b-", linewidth=2)
                    ax.set_xlabel("Temperature [°C]")
                    ax.set_ylabel(f"{title} [{units}]")
                    ax.set_title(f"{title} vs Temperature")
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error: {str(e)}",
                        transform=ax.transAxes,
                        ha="center",
                    )

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.suptitle(
            f"Fluid Properties: {self.name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def compare_with_data(
        self,
        T_data: Union[List, np.ndarray],
        property_data: Dict[str, Union[List, np.ndarray]],
        property_name: str = None,
        save_fig: bool = False,
        filename: str = None,
    ):
        """
        Compare correlation models with experimental/manufacturer data

        Parameters:
        -----------
        T_data : array-like
            Temperature data points [K]
        property_data : dict
            Dictionary with property names as keys and measurement data as
                values
            Keys should match: 'density', 'heat_capacity',
                'thermal_conductivity', 'viscosity', 'vapor_pressure'
        property_name : str, optional
            Specific property to plot. If None, plots all available properties
        save_fig : bool, default False
            Whether to save the figure
        filename : str, optional
            Filename for saved figure

        Returns:
        --------
        dict
            Statistics for each property (RMSE, MAE, MAPE, R²)
        """

        T_data = np.array(T_data)
        T_C_data = T_data - 273.15

        # Define properties to compare
        prop_info = [
            ("density", "Density", "kg/m³"),
            ("heat_capacity", "Heat Capacity", "J/kg·K"),
            ("thermal_conductivity", "Thermal Conductivity", "W/m·K"),
            ("viscosity", "Dynamic Viscosity", "Pa·s"),
            ("vapor_pressure", "Vapor Pressure", "Pa"),
        ]

        # Filter properties based on what's available
        available_props = []
        for prop_key, prop_title, prop_units in prop_info:
            if prop_key in self.correlations and prop_key in property_data:
                if property_name is None or prop_key == property_name:
                    available_props.append((prop_key, prop_title, prop_units))

        if not available_props:
            print("No matching properties found for comparison")
            return {}

        # Create plots
        n_props = len(available_props)
        n_cols = 2 if n_props > 1 else 1
        n_rows = (n_props + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(16, 6 * n_rows))
        if n_rows == 1:
            if n_props == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 2)

        results = {}

        for idx, (prop_key, prop_title, prop_units) in enumerate(
            available_props
        ):
            # Get experimental data
            exp_data = np.array(property_data[prop_key])

            # Clip to correlation's valid temperature range
            T_min_corr, T_max_corr = self.correlations[prop_key].get_valid_range()
            valid_mask = (T_data >= T_min_corr) & (T_data <= T_max_corr)
            T_valid = T_data[valid_mask]
            T_C_valid = T_valid - 273.15
            exp_valid = exp_data[valid_mask]

            # Get correlation predictions within valid range
            pred_data = self.correlations[prop_key].evaluate(T_valid)

            # Calculate statistics
            residuals = exp_valid - pred_data
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            mape = (
                np.mean(np.abs(residuals / exp_valid)) * 100
                if np.all(exp_valid != 0)
                else float("inf")
            )

            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((exp_valid - np.mean(exp_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("-inf")

            results[prop_key] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r_squared": r_squared,
            }

            # Plotting
            if n_props == 1:
                ax1 = axes[0]
                ax2 = axes[1]
            else:
                row = idx // n_cols
                col = idx % n_cols
                ax1 = axes[row, col * 2]
                ax2 = axes[row, col * 2 + 1]

            # Plot 1: Data vs Correlation (within valid range only)
            T_plot = np.linspace(T_min_corr, T_max_corr, 100)
            T_C_plot = T_plot - 273.15
            pred_plot = self.correlations[prop_key].evaluate(T_plot)

            ax1.plot(
                T_C_valid,
                exp_valid,
                "bo",
                label="Data",
                markersize=6,
                alpha=0.7,
            )
            ax1.plot(
                T_C_plot, pred_plot, "r-", label="Correlation", linewidth=2
            )
            ax1.set_xlabel("Temperature [°C]")
            ax1.set_ylabel(f"{prop_title} [{prop_units}]")
            ax1.set_title(f"{prop_title} Comparison")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = (
                f"R² = {r_squared:.4f}\nRMSE = {rmse:.2e}\nMAPE = {mape:.1f}%"
            )
            ax1.text(
                0.05,
                0.95,
                stats_text,
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Plot 2: Residuals
            ax2.plot(T_C_valid, residuals, "go", markersize=6, alpha=0.7)
            ax2.axhline(y=0, color="k", linestyle="--", linewidth=1)
            ax2.set_xlabel("Temperature [°C]")
            ax2.set_ylabel(f"Residual [{prop_units}]")
            ax2.set_title(f"{prop_title} Residuals")
            ax2.grid(True, alpha=0.3)

            # Print statistics
            print(f"\n{prop_title.upper()} Validation Statistics:")
            print(f"  RMSE: {rmse:.4e}")
            print(f"  MAE:  {mae:.4e}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²:   {r_squared:.6f}")

        # Hide unused subplots
        total_subplots = n_rows * n_cols * 2
        used_subplots = len(available_props) * 2
        if total_subplots > used_subplots:
            for idx in range(used_subplots, total_subplots):
                row = idx // (n_cols * 2)
                col = idx % (n_cols * 2)
                if n_props == 1:
                    continue  # Already handled
                else:
                    axes[row, col].axis("off")

        plt.suptitle(
            f"Property Validation: {self.name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_fig:
            if filename is None:
                filename = (
                    f"{self.name.replace(' ', '_')}_property_validation.png"
                )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"\nFigure saved as: {filename}")

        plt.show()

        return results

    def save_to_json(self, filename: str):
        """Save fluid properties to JSON file"""
        data = {
            "name": self.name,
            "molecular_weight": self.molecular_weight,
            "T_reference": self.T_reference,
            "P_reference": self.P_reference,
            "correlations": {
                name: corr.to_dict()
                for name, corr in self.correlations.items()
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Fluid properties saved to {filename}")

    def summary(self):
        """Print summary of defined properties"""
        print(f"Fluid: {self.name}")
        if self.molecular_weight:
            print(f"  Molecular weight: {self.molecular_weight} kg/kmol")
        print(
            f"  Reference conditions: T = {self.T_reference:.1f} K, "
            f"P = {self.P_reference / 1000:.1f} kPa"
        )
        print("\nDefined properties:")

        for prop_name, corr in self.correlations.items():
            T_min, T_max = corr.get_valid_range()
            print(
                f"  • {prop_name}: {corr.name} "
                f"({T_min - 273.15:.1f}°C to {T_max - 273.15:.1f}°C)"
            )


# Example fluid databases
def create_water_properties() -> FluidProperties:
    """
    Create water property database with standard correlations
    """

    water = FluidProperties("Water", molecular_weight=18.015)

    # Density: polynomial fit (273-373 K)
    # rho = 999.8 - 0.0676*T + ... (approximate)
    water.set_density(
        PolynomialCorrelation(
            coefficients=[999.8, -0.0676, -0.0096, 0.0],
            T_min=273.15,
            T_max=373.15,
            name="Polynomial fit",
        )
    )

    # Heat capacity: nearly constant for liquid water
    water.set_heat_capacity(
        PolynomialCorrelation(
            coefficients=[4180, 0.5],
            T_min=273.15,
            T_max=373.15,
            name="Linear approximation",
        )
    )

    # Thermal conductivity: linear approximation
    water.set_thermal_conductivity(
        PolynomialCorrelation(
            coefficients=[0.56, 0.002],
            T_min=273.15,
            T_max=373.15,
            name="Linear approximation",
        )
    )

    # Viscosity: exponential (Andrade equation)
    water.set_viscosity(
        ExponentialCorrelation(
            A=2.414e-5,
            B=247.8,
            C=-140e-6,
            T_min=273.15,
            T_max=373.15,
            name="Andrade equation",
        )
    )

    # Vapor pressure: Antoine equation
    water.set_vapor_pressure(
        AntoineCorrelation(
            A=8.07131,
            B=1730.63,
            C=233.426,
            T_min=273.15,
            T_max=373.15,
            name="Antoine equation",
        )
    )

    return water


def create_thermal_oil_properties() -> FluidProperties:
    """
    Create thermal oil (Therminol VP-1) property database
    """

    oil = FluidProperties("Therminol VP-1", molecular_weight=230)

    # Density: linear decrease with temperature
    oil.set_density(
        PolynomialCorrelation(
            coefficients=[1060, -0.65],
            T_min=273.15,
            T_max=673.15,
            name="Linear fit",
        )
    )

    # Heat capacity: increases with temperature
    oil.set_heat_capacity(
        PolynomialCorrelation(
            coefficients=[1500, 2.5],
            T_min=273.15,
            T_max=673.15,
            name="Linear fit",
        )
    )

    # Thermal conductivity: decreases with temperature
    oil.set_thermal_conductivity(
        PolynomialCorrelation(
            coefficients=[0.137, -0.00015],
            T_min=273.15,
            T_max=673.15,
            name="Linear fit",
        )
    )

    # Viscosity: strong exponential decrease
    oil.set_viscosity(
        ExponentialCorrelation(
            A=0.0001,
            B=2500,
            C=0.0,
            T_min=273.15,
            T_max=673.15,
            name="Exponential fit",
        )
    )

    # Vapor pressure: low (thermal oils have low volatility)
    oil.set_vapor_pressure(
        AntoineCorrelation(
            A=7.0,
            B=2000,
            C=200,
            T_min=273.15,
            T_max=673.15,
            name="Antoine equation",
        )
    )

    return oil


def demonstration():
    """
    Demonstrate usage of FluidProperties class
    """

    print("Fluid Properties Class Demonstration")
    print("=" * 40)

    # Create water properties
    print("\n1. Creating Water Properties Database:")
    water = create_water_properties()
    water.summary()

    # Evaluate at specific temperature
    T_test = 323.15  # 50°C
    print(f"\nProperties at {T_test - 273.15:.1f}°C:")
    props = water.get_all_properties(T_test)
    for name, value in props.items():
        print(f"  {name}: {value:.4e}")

    # Plot properties
    print("\nPlotting water properties...")
    water.plot_properties()

    # Create thermal oil
    print("\n2. Creating Thermal Oil Properties Database:")
    oil = create_thermal_oil_properties()
    oil.summary()

    # Compare properties at different temperatures
    print("\nProperty comparison at different temperatures:")
    T_range = [298.15, 323.15, 373.15, 423.15]
    print(
        f"{'T[°C]':<10} {'ρ[kg/m³]':<12} {'cp[J/kg·K]':<12} "
        f"{'k[W/m·K]':<12} {'μ[Pa·s]':<12} {'Pr':<10}"
    )
    print("-" * 78)

    for T in T_range:
        rho = oil.density(T)
        cp = oil.heat_capacity(T)
        k = oil.thermal_conductivity(T)
        mu = oil.viscosity(T)
        Pr = oil.prandtl_number(T)

        print(
            f"{T - 273.15:<10.1f} {rho:<12.1f} {cp:<12.1f} "
            f"{k:<12.4f} {mu:<12.4e} {Pr:<10.2f}"
        )

    # Plot thermal oil
    print("\nPlotting thermal oil properties...")
    oil.plot_properties()

    # Save to file
    oil.save_to_json("thermal_oil_properties.json")

    print("\nDemonstration complete!")


if __name__ == "__main__":
    demonstration()
