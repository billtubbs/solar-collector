#!/usr/bin/env python3
"""
Script to compare fluid property correlations with experimental/manufacturer data
and generate validation plots for SYLTHERM 800
"""

import numpy as np
import pandas as pd
from src.fluids import FluidProperties, PolynomialCorrelation, ExponentialCorrelation


def load_syltherm_data():
    """Load SYLTHERM 800 manufacturer data"""
    try:
        df = pd.read_csv("data/properties/fluids/SYLTHERM800_data.csv")

        data = {
            "T_K": df["Temperature_C"].values + 273.15,
            "density": df["Density_kg_m3"].values,
            "heat_capacity": df["Heat_Capacity_kJ_kg_K"].values * 1000,  # Convert to J/kg·K
            "thermal_conductivity": df["Thermal_Conductivity_W_m_K"].values,
            "viscosity": df["Viscosity_mPa_s"].values / 1000,  # Convert to Pa·s
            "vapor_pressure": df["Vapor_Pressure_kPa"].values * 1000,  # Convert to Pa
        }

        return data
    except FileNotFoundError:
        print("SYLTHERM800_data.csv not found. Please ensure the file exists in data/properties/fluids/.")
        return None


def create_syltherm_fluid():
    """Create a FluidProperties object with SYLTHERM 800 correlations"""
    fluid = FluidProperties("SYLTHERM 800", molecular_weight=400.0)

    # Set temperature range
    T_min, T_max = 233.15, 673.15  # -40°C to 400°C

    # Density correlation: ρ = a + b*T_C + c*T_C²
    # These are the coefficients from the original code - they may not be optimal
    density_coeffs = [1000.0, -0.8652, -0.000401]  # [a, b, c]
    density_corr = PolynomialCorrelation(density_coeffs, T_min, T_max, "Density")
    fluid.set_density(density_corr)

    # Heat capacity correlation: cp = a + b*T_C
    cp_coeffs = [1506.2, 2.938]  # [a, b]
    cp_corr = PolynomialCorrelation(cp_coeffs, T_min, T_max, "Heat Capacity")
    fluid.set_heat_capacity(cp_corr)

    # Thermal conductivity correlation: k = a + b*T_C
    k_coeffs = [0.14636, -0.00006185]  # [a, b]
    k_corr = PolynomialCorrelation(k_coeffs, T_min, T_max, "Thermal Conductivity")
    fluid.set_thermal_conductivity(k_corr)

    # Viscosity correlation: μ = A * exp(B/T) + C (Andrade equation)
    # From original code: μ = a * exp(b/T) where a=9.85e-7, b=2396.8
    viscosity_corr = ExponentialCorrelation(
        A=9.85e-7, B=2396.8, C=0.0, T_min=T_min, T_max=T_max, name="Viscosity"
    )
    fluid.set_viscosity(viscosity_corr)

    # Vapor pressure correlation: P = A * exp(B/T) + C
    # For Antoine equation: log10(P_kPa) = a + b/(T_C + c)
    # This doesn't directly fit the ExponentialCorrelation form, so we'll approximate
    # Using simplified exponential form for demonstration
    vapor_pressure_corr = ExponentialCorrelation(
        A=1e-6, B=5000, C=0.0, T_min=T_min, T_max=T_max, name="Vapor Pressure"
    )
    fluid.set_vapor_pressure(vapor_pressure_corr)

    return fluid


def generate_validation_plots():
    """Generate comprehensive validation plots comparing correlations with data"""
    print("SYLTHERM 800 Correlation Validation")
    print("=" * 50)

    # Load manufacturer data
    mfr_data = load_syltherm_data()
    if mfr_data is None:
        print("Cannot proceed without manufacturer data")
        return

    print(f"Loaded {len(mfr_data['T_K'])} temperature points")
    print(f"Temperature range: {mfr_data['T_K'].min() - 273.15:.0f}°C to {mfr_data['T_K'].max() - 273.15:.0f}°C")

    # Create fluid with correlations
    fluid = create_syltherm_fluid()

    # Prepare data for comparison
    property_data = {
        "density": mfr_data["density"],
        "heat_capacity": mfr_data["heat_capacity"],
        "thermal_conductivity": mfr_data["thermal_conductivity"],
        "viscosity": mfr_data["viscosity"],
        "vapor_pressure": mfr_data["vapor_pressure"],
    }

    # Compare correlations with manufacturer data
    print("\nComparing correlations with manufacturer data...")

    try:
        results = fluid.compare_with_data(
            T_data=mfr_data["T_K"],
            property_data=property_data,
            save_fig=True,
            filename="SYLTHERM800_validation.png",
        )

        print("\nValidation completed successfully!")
        print("Results summary:")
        print("-" * 60)

        # Format results in a nice table
        prop_format = "{name:>21s}: R² = {r2:>7.4f}, RMSE = {rmse:>10.3e}, MAPE = {mape:>6.1f}%"

        for prop, stats in results.items():
            print(prop_format.format(
                name=prop.replace("_", " ").title(),
                r2=stats["r_squared"],
                rmse=stats["rmse"],
                mape=stats["mape"]
            ))

        print(f"\nValidation plots saved as: SYLTHERM800_validation.png")

        # Additional analysis
        print("\nCorrelation Quality Assessment:")
        print("-" * 40)

        excellent_props = []
        good_props = []
        poor_props = []

        for prop, stats in results.items():
            r2 = stats["r_squared"]
            if r2 > 0.95:
                excellent_props.append(prop)
            elif r2 > 0.80:
                good_props.append(prop)
            else:
                poor_props.append(prop)

        if excellent_props:
            print(f"Excellent fit (R² > 0.95): {', '.join(excellent_props)}")
        if good_props:
            print(f"Good fit (0.80 < R² ≤ 0.95): {', '.join(good_props)}")
        if poor_props:
            print(f"Poor fit (R² ≤ 0.80): {', '.join(poor_props)}")

        print("\nNote: These correlations use placeholder coefficients.")
        print("Consider fitting optimized coefficients to improve accuracy.")

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()


def test_individual_properties():
    """Test individual property correlations at specific temperatures"""
    print("\n" + "=" * 50)
    print("Individual Property Testing")
    print("=" * 50)

    # Load data and create fluid
    mfr_data = load_syltherm_data()
    if mfr_data is None:
        return

    fluid = create_syltherm_fluid()

    # Test temperatures
    test_temps = [298.15, 373.15, 473.15, 573.15]  # 25°C, 100°C, 200°C, 300°C
    test_temps_C = [T - 273.15 for T in test_temps]

    print(f"\nTesting correlations at selected temperatures:")
    print(f"Temperatures: {', '.join([f'{T:.0f}°C' for T in test_temps_C])}")
    print("-" * 60)

    # Test each property
    properties = [
        ("density", "kg/m³", 1),
        ("heat_capacity", "J/kg·K", 0),
        ("thermal_conductivity", "W/m·K", 5),
        ("viscosity", "mPa·s", 2),
        ("vapor_pressure", "Pa", 2),
    ]

    for prop_name, units, decimals in properties:
        if prop_name in fluid.correlations:
            print(f"\n{prop_name.replace('_', ' ').title()}:")

            values = []
            for T in test_temps:
                if prop_name == "viscosity":
                    # Convert to mPa·s for display
                    value = getattr(fluid, prop_name)(T) * 1000
                else:
                    value = getattr(fluid, prop_name)(T)
                values.append(value)

            # Print values
            for T_C, value in zip(test_temps_C, values):
                print(f"  {T_C:>6.0f}°C: {value:>10.{decimals}f} {units}")


def main():
    """Main function to run all validation and testing"""
    print("SYLTHERM 800 Property Correlation Analysis")
    print("=" * 80)
    print(__doc__.strip())
    print("=" * 80)

    # Generate validation plots
    generate_validation_plots()

    # Test individual properties
    test_individual_properties()

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("\nFiles generated:")
    print("  - SYLTHERM800_validation.png (comparison plots)")
    print("\nNext steps:")
    print("  1. Review correlation accuracy in the plots")
    print("  2. Consider implementing parameter fitting for better accuracy")
    print("  3. Validate fitted correlations against manufacturer data")


if __name__ == "__main__":
    main()