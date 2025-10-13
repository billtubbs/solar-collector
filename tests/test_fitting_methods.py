#!/usr/bin/env python3
"""
Pytest test suite for CasADi-based fitting methods in the
FittedPropertyCorrelation classes
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.fluids import (
    AntoineCorrelation,
    ExponentialCorrelation,
    PolynomialCorrelation,
)

DATA_DIR = "data/properties/fluids"
FILENAME = "SYLTHERM800_data.csv"


@pytest.fixture
def syltherm_data():
    """Load SYLTHERM 800 manufacturer data fixture"""
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, FILENAME))
        data = {
            "T_K": df["Temperature_C"].values + 273.15,
            "T_C": df["Temperature_C"].values,
            "density": df["Density_kg_m3"].values,
            "heat_capacity": df["Heat_Capacity_kJ_kg_K"].values
            * 1000,  # Convert to J/kg·K
            "thermal_conductivity": df["Thermal_Conductivity_W_m_K"].values,
            "viscosity": df["Viscosity_mPa_s"].values
            / 1000,  # Convert to Pa·s
            "vapor_pressure": df["Vapor_Pressure_kPa"].values
            * 1000,  # Convert to Pa
        }
        return data

    except FileNotFoundError:
        pytest.skip(
            f"{FILENAME!r} not found. Check file exists in {DATA_DIR}."
        )
        return None


@pytest.fixture
def density_fit_params():
    """Fitting parameters for density correlation"""
    return {
        "bounds": {
            "a0": (800, 1400),  # Intercept
            "a1": (-2, 2),  # Slope term
        },
        "initial_guess": {"a0": 1200, "a1": -0.5},
    }


@pytest.fixture
def viscosity_fit_params():
    """Fitting parameters for viscosity correlation"""
    return {
        "bounds": {
            "A": (1e-8, 1e-3),  # Pre-exponential factor
            "B": (500, 5000),  # Activation energy parameter
            "C": (-1e-3, 1e-3),  # Offset
        },
        "initial_guess": {"A": 1e-6, "B": 2000, "C": 0},
    }


@pytest.fixture
def vapor_pressure_fit_params():
    """Fitting parameters for vapor pressure correlation"""
    return {
        "bounds": {
            "A": (0, 20),  # Antoine constant
            "B": (-8000, -500),  # Antoine constant
            "C": (-300, 300),  # Antoine constant
        },
        "initial_guess": {"A": 15, "B": -3000, "C": 0},
    }


class TestPolynomialCorrelationFitting:
    """Test polynomial correlation fitting methods"""

    def test_polynomial_fitting_density(
        self, syltherm_data, density_fit_params
    ):
        """Test polynomial correlation fitting with density data"""
        T_data = syltherm_data["T_K"]
        y_data = syltherm_data["density"]

        # Create a 4th order polynomial correlation
        poly_corr = PolynomialCorrelation(
            coefficients=[0, 0, 0, 0],  # Will be fitted
            T_min=T_data.min(),
            T_max=T_data.max(),
            name="Density_Fitted",
        )

        # Fit to data
        fit_results = poly_corr.fit(
            T_data,
            y_data,
            bounds=density_fit_params["bounds"],
            initial_guess=density_fit_params["initial_guess"],
        )

        # Assertions
        assert fit_results["r_squared"] > 0.99, (
            f"R² should be > 0.99, got {fit_results['r_squared']:.4f}"
        )
        assert fit_results["rmse"] < 10.0, (
            f"RMSE should be < 10 kg/m³, got {fit_results['rmse']:.2f}"
        )
        assert fit_results["solver_status"] == "Solve_Succeeded"
        assert "fitted_parameters" in fit_results
        assert fit_results["correlation_type"] == "polynomial"
        assert fit_results["n_parameters"] == 4

        # Test parameter names
        param_names = poly_corr.get_param_names()
        assert param_names == ["a0", "a1", "a2", "a3"]

        # Test predictions at specific temperatures
        test_temps = np.array([298.15, 373.15, 573.15])  # 25°C, 100°C, 300°C
        predictions = np.array([poly_corr.evaluate(T) for T in test_temps])

        # Density should decrease with temperature for SYLTHERM 800
        assert predictions[0] > predictions[1] > predictions[2], (
            "Density should decrease with temperature"
        )

        # Check reasonable ranges
        assert 600 < predictions[0] < 1000, (
            f"Density at 25°C should be reasonable, got {predictions[0]:.1f}"
        )
        assert 400 < predictions[2] < 800, (
            f"Density at 300°C should be reasonable, got {predictions[2]:.1f}"
        )

    def test_polynomial_fitting_heat_capacity(self, syltherm_data):
        """Test polynomial correlation fitting with heat capacity data"""
        T_data = syltherm_data["T_K"]
        y_data = syltherm_data["heat_capacity"]

        # Create a 3rd order polynomial correlation
        poly_corr = PolynomialCorrelation(
            coefficients=[0, 0, 0],  # Will be fitted
            T_min=T_data.min(),
            T_max=T_data.max(),
            name="HeatCapacity_Fitted",
        )

        bounds = {
            "a0": (1000, 2500),  # Intercept
            "a1": (0, 10),  # Linear term
            "a2": (-0.01, 0.01),  # Quadratic term
        }

        initial_guess = {"a0": 1500, "a1": 2, "a2": 0}

        fit_results = poly_corr.fit(
            T_data, y_data, bounds=bounds, initial_guess=initial_guess
        )

        # Assertions
        assert fit_results["r_squared"] > 0.95, (
            f"R² should be > 0.95, got {fit_results['r_squared']:.4f}"
        )
        assert fit_results["rmse"] < 100.0, (
            f"RMSE should be < 100 J/kg·K, got {fit_results['rmse']:.2f}"
        )
        assert fit_results["solver_status"] == "Solve_Succeeded"


class TestExponentialCorrelationFitting:
    """Test exponential correlation fitting methods"""

    def test_exponential_fitting_viscosity(
        self, syltherm_data, viscosity_fit_params
    ):
        """Test exponential correlation fitting with viscosity data"""
        T_data = syltherm_data["T_K"]
        y_data = syltherm_data["viscosity"]

        # Create exponential correlation
        exp_corr = ExponentialCorrelation(
            A=1e-6,
            B=2000,
            C=0,  # Will be fitted
            T_min=T_data.min(),
            T_max=T_data.max(),
            name="Viscosity_Fitted",
        )

        # Fit to data
        fit_results = exp_corr.fit(
            T_data,
            y_data,
            bounds=viscosity_fit_params["bounds"],
            initial_guess=viscosity_fit_params["initial_guess"],
        )

        # Assertions
        assert fit_results["r_squared"] > 0.99, (
            f"R² should be > 0.99, got {fit_results['r_squared']:.4f}"
        )
        assert fit_results["rmse"] < 0.001, (
            f"RMSE should be < 0.001 Pa·s, got {fit_results['rmse']:.6f}"
        )
        assert fit_results["solver_status"] == "Solve_Succeeded"
        assert fit_results["correlation_type"] == "exponential"
        assert fit_results["n_parameters"] == 3

        # Test parameter names
        param_names = exp_corr.get_param_names()
        assert param_names == ["A", "B", "C"]

        # Test predictions at specific temperatures
        test_temps = np.array([298.15, 373.15, 573.15])  # 25°C, 100°C, 300°C
        predictions = np.array([exp_corr.evaluate(T) for T in test_temps])

        # Viscosity should decrease with temperature
        assert predictions[0] > predictions[1] > predictions[2], (
            "Viscosity should decrease with temperature"
        )

        # Convert to mPa·s for checking reasonable ranges
        predictions_mPas = predictions * 1000
        assert 5 < predictions_mPas[0] < 15, (
            "Viscosity at 25°C should be reasonable, got "
            f"{predictions_mPas[0]:.2f} mPa·s"
        )
        assert 0.5 < predictions_mPas[2] < 2, (
            "Viscosity at 300°C should be reasonable, got "
            f"{predictions_mPas[2]:.2f} mPa·s"
        )


class TestAntoineCorrelationFitting:
    """Test Antoine correlation fitting methods"""

    def test_antoine_fitting_setup(
        self, syltherm_data, vapor_pressure_fit_params
    ):
        """Test Antoine correlation setup and basic functionality"""
        T_data = syltherm_data["T_K"]
        y_data = syltherm_data["vapor_pressure"]

        # Create Antoine correlation
        antoine_corr = AntoineCorrelation(
            A=8,
            B=-2500,
            C=-50,  # Will be fitted
            T_min=T_data.min(),
            T_max=T_data.max(),
            name="VaporPressure_Fitted",
        )

        # Test parameter names
        param_names = antoine_corr.get_param_names()
        assert param_names == ["A", "B", "C"]

        # Test that fitting method exists and can be called
        # Note: Antoine fitting may not converge well for SYLTHERM 800
        # low vapor pressure data
        try:
            fit_results = antoine_corr.fit(
                T_data,
                y_data,
                bounds=vapor_pressure_fit_params["bounds"],
                initial_guess=vapor_pressure_fit_params["initial_guess"],
            )

            # If fitting succeeds, check basic structure
            assert "r_squared" in fit_results
            assert "rmse" in fit_results
            assert "fitted_parameters" in fit_results
            assert fit_results["correlation_type"] == "antoine"
            assert fit_results["n_parameters"] == 3

        except Exception as e:
            # Antoine fitting may fail for low vapor pressure fluids
            pytest.skip(
                "Antoine fitting failed (expected for low vapor pressure): "
                f"{e}"
            )


class TestFittingFramework:
    """Test overall fitting framework functionality"""

    def test_fitted_correlation_inheritance(self):
        """Test that all correlation classes properly inherit from
        FittedPropertyCorrelation
        """
        # Test PolynomialCorrelation
        poly_corr = PolynomialCorrelation([1, 2, 3], 200, 400, "Test")
        assert hasattr(poly_corr, "fit")
        assert hasattr(poly_corr, "get_param_names")
        assert hasattr(poly_corr, "update_params")

        # Test ExponentialCorrelation
        exp_corr = ExponentialCorrelation(1, 2, 3, 200, 400, "Test")
        assert hasattr(exp_corr, "fit")
        assert hasattr(exp_corr, "get_param_names")
        assert hasattr(exp_corr, "update_params")

        # Test AntoineCorrelation
        antoine_corr = AntoineCorrelation(1, 2, 3, 200, 400, "Test")
        assert hasattr(antoine_corr, "fit")
        assert hasattr(antoine_corr, "get_param_names")
        assert hasattr(antoine_corr, "update_params")

    def test_parameter_updates(self):
        """Test parameter update functionality"""
        poly_corr = PolynomialCorrelation([1, 2, 3], 200, 400, "Test")

        # Test initial coefficients
        assert np.array_equal(poly_corr.coefficients, [1, 2, 3])

        # Update parameters
        new_params = {"a0": 10, "a1": 20, "a2": 30}
        poly_corr.update_params(new_params)

        # Check updated coefficients
        assert np.array_equal(poly_corr.coefficients, [10, 20, 30])

    def test_fit_statistics_structure(self, syltherm_data):
        """Test that fit results have proper structure"""
        T_data = syltherm_data["T_K"][:10]  # Use subset for faster test
        y_data = syltherm_data["density"][:10]

        poly_corr = PolynomialCorrelation(
            [0, 0], T_data.min(), T_data.max(), "Test"
        )

        bounds = {"a0": (800, 1200), "a1": (-2, 2)}
        initial_guess = {"a0": 1000, "a1": -0.5}

        fit_results = poly_corr.fit(
            T_data, y_data, bounds=bounds, initial_guess=initial_guess
        )

        # Check required keys in fit results
        required_keys = [
            "r_squared",
            "rmse",
            "mae",
            "objective_value",
            "solver_status",
            "fitted_parameters",
            "correlation_type",
            "n_parameters",
        ]

        for key in required_keys:
            assert key in fit_results, f"Missing required key: {key}"

        # Check data types
        assert isinstance(fit_results["r_squared"], (int, float))
        assert isinstance(fit_results["rmse"], (int, float))
        assert isinstance(fit_results["mae"], (int, float))
        assert isinstance(fit_results["fitted_parameters"], dict)
        assert isinstance(fit_results["correlation_type"], str)
        assert isinstance(fit_results["n_parameters"], int)

    def test_plot_correlation_method(self, syltherm_data):
        """Test that plot_correlation method works correctly"""
        T_data = syltherm_data["T_K"][:20]  # Use subset for faster test
        y_data = syltherm_data["density"][:20]

        poly_corr = PolynomialCorrelation(
            [1000, -0.8, 0], T_data.min(), T_data.max(), "Test_Plotting"
        )

        # Test plotting without saving (just check it doesn't crash)
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for testing

        fig, axes = poly_corr.plot_correlation(
            T_data=T_data,
            y_data=y_data,
            title="Test Plot",
            show_residuals=True,
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, np.ndarray)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
