#!/usr/bin/env python3
"""
Example usage of the plot_correlation method for FittedPropertyCorrelation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fluids import PolynomialCorrelation


def main():
    """Demonstrate how to use the plot_correlation method"""
    print("Example: Using plot_correlation method")
    print("=" * 50)

    # Load some data (using SYLTHERM 800 density data as example)
    try:
        df = pd.read_csv("data/properties/fluids/SYLTHERM800_data.csv")
        T_data = df["Temperature_C"].values + 273.15  # Convert to Kelvin
        density_data = df["Density_kg_m3"].values
        print(f"Loaded {len(T_data)} data points from SYLTHERM800_data.csv")
    except FileNotFoundError:
        print(
            "Data file not found. Creating synthetic data for demonstration."
        )
        T_data = np.linspace(233.15, 673.15, 45)
        density_data = (
            1000 - 0.8 * (T_data - 273.15) - 0.0004 * (T_data - 273.15) ** 2
        )

    # 1. Create a correlation with initial coefficients
    correlation = PolynomialCorrelation(
        coefficients=[1000, -0.8, -0.0004],  # Initial guess coefficients
        T_min=T_data.min(),
        T_max=T_data.max(),
        name="Density_Example",
    )

    # 2. Plot the correlation vs data (before fitting)
    print("\n1. Plotting correlation with initial coefficients...")
    fig, axes = correlation.plot_correlation(
        T_data=T_data,
        y_data=density_data,
        title="Initial Correlation vs Data",
        ylabel="Density [kg/m³]",
    )
    plt.tight_layout()
    plt.show()

    # 3. Fit the correlation to the data
    print("\n2. Fitting correlation to data...")
    bounds = {"a0": (800, 1200), "a1": (-2, 2), "a2": (-0.01, 0.01)}
    initial_guess = {"a0": 1000, "a1": -0.8, "a2": -0.0004}

    fit_results = correlation.fit(
        T_data, density_data, bounds=bounds, initial_guess=initial_guess
    )

    print(f"Fitting completed! R² = {fit_results['r_squared']:.4f}")

    # 4. Plot the fitted correlation vs data
    print("\n3. Plotting fitted correlation...")
    fig, axes = correlation.plot_correlation(
        T_data=T_data,
        y_data=density_data,
        title="Fitted Correlation vs Data",
        ylabel="Density [kg/m³]",
    )
    plt.tight_layout()
    plt.show()

    # 6. Example of plotting without residuals
    print("\n5. Example: Plot without residuals subplot...")
    fig, ax = correlation.plot_correlation(
        T_data=T_data,
        y_data=density_data,
        title="Fitted Correlation (Main Plot Only)",
        ylabel="Density [kg/m³]",
        show_residuals=False,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
