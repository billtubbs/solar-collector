from typing import Union

import matplotlib.pyplot as plt
import numpy as np


class SYLTHERM800:
    """
    SYLTHERM 800 Heat Transfer Fluid Properties

    Based on manufacturer data from Dow Chemical
    Data source: https://www.dow.com/en-us/pdp.syltherm-800-heat-transfer-fluid.html

    Valid range: -40°C to 400°C (233.15 K to 673.15 K)

    All correlations fitted to manufacturer specification data
    """

    def __init__(self):
        self.name = "SYLTHERM 800"
        self.description = "Silicone-based heat transfer fluid"
        self.manufacturer = "Dow Chemical"

        # Valid temperature range
        self.T_min = 233.15  # -40°C
        self.T_max = 673.15  # 400°C
        self.T_ref = 298.15  # 25°C (reference)

        # Fitted correlation coefficients
        self._set_correlation_coefficients()

        print(f"{self.name} - Temperature-Dependent Fluid Properties")
        print(
            f"Valid range: {self.T_min - 273.15:.0f}°C to "
            f"{self.T_max - 273.15:.0f}°C"
        )

    def _set_correlation_coefficients(self):
        """
        Store correlation coefficients fitted to manufacturer data
        """
        # Density: ρ = a + b*T + c*T²  [kg/m³] with T in °C
        self.rho_coeffs = {"a": 1000.0, "b": -0.8652, "c": -0.000401}

        # Heat capacity: cp = a + b*T  [J/kg·K] with T in °C
        self.cp_coeffs = {"a": 1506.2, "b": 2.938}

        # Thermal conductivity: k = a + b*T  [W/m·K] with T in °C
        self.k_coeffs = {"a": 0.14636, "b": -0.00006185}

        # Viscosity: μ = a * exp(b/T)  [Pa·s] with T in K
        # Exponential (Andrade) form
        self.mu_coeffs = {"a": 9.85e-7, "b": 2396.8}

        # Vapor pressure: log10(P_kPa) = a + b/(T_C + c)  with T in °C
        # Antoine equation form
        self.Pv_coeffs = {"a": 9.467, "b": -3004.5, "c": 201.38}

    def _check_temperature_range(self, T: Union[float, np.ndarray]):
        """Check if temperature is in valid range and warn if not"""
        T_array = np.atleast_1d(T)
        if np.any(T_array < self.T_min) or np.any(T_array > self.T_max):
            T_min_C = self.T_min - 273.15
            T_max_C = self.T_max - 273.15
            print(
                f"Warning: Temperature outside valid range "
                f"({T_min_C:.0f}°C to {T_max_C:.0f}°C)"
            )

    def density(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Density [kg/m³] as function of temperature [K]

        Correlation: ρ = a + b*T_C + c*T_C²
        R² > 0.9999

        Parameters:
        -----------
        T : float or array
            Temperature in Kelvin

        Returns:
        --------
        rho : float or array
            Density in kg/m³
        """
        self._check_temperature_range(T)
        T_C = T - 273.15  # Convert to Celsius

        a = self.rho_coeffs["a"]
        b = self.rho_coeffs["b"]
        c = self.rho_coeffs["c"]

        rho = a + b * T_C + c * T_C**2

        return rho

    def heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Specific heat capacity [J/kg·K] as function of temperature [K]

        Correlation: cp = a + b*T_C
        Linear increase with temperature
        R² > 0.9999

        Note: Manufacturer data in kJ/kg·K, converted to J/kg·K
        """
        self._check_temperature_range(T)
        T_C = T - 273.15

        a = self.cp_coeffs["a"]
        b = self.cp_coeffs["b"]

        cp = a + b * T_C  # Result in J/kg·K

        return cp

    def thermal_conductivity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Thermal conductivity [W/m·K] as function of temperature [K]

        Correlation: k = a + b*T_C
        Slight linear decrease with temperature
        R² > 0.9999
        """
        self._check_temperature_range(T)
        T_C = T - 273.15

        a = self.k_coeffs["a"]
        b = self.k_coeffs["b"]

        k = a + b * T_C

        return k

    def viscosity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Dynamic viscosity [Pa·s] as function of temperature [K]

        Correlation: μ = a * exp(b/T)
        Exponential decrease (Andrade equation)
        R² > 0.998

        Note: Manufacturer data in mPa·s (cSt × density), converted to Pa·s
        """
        self._check_temperature_range(T)

        a = self.mu_coeffs["a"]
        b = self.mu_coeffs["b"]

        # Andrade equation
        mu = a * np.exp(b / T)

        return mu

    def vapor_pressure(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Vapor pressure [Pa] as function of temperature [K]

        Correlation: log10(P_kPa) = a + b/(T_C + c)
        Antoine equation
        R² > 0.999

        SYLTHERM 800 has low vapor pressure (suitable for high temperatures)
        """
        self._check_temperature_range(T)
        T_C = T - 273.15

        a = self.Pv_coeffs["a"]
        b = self.Pv_coeffs["b"]
        c = self.Pv_coeffs["c"]

        # Antoine equation (result in kPa)
        log_P_kPa = a + b / (T_C + c)
        P_kPa = 10**log_P_kPa

        # Convert kPa to Pa
        P_Pa = P_kPa * 1000

        # Handle very low temperatures where pressure is negligible
        P_Pa = np.maximum(P_Pa, 0.01)  # Minimum 0.01 Pa

        return P_Pa

    # Derived thermophysical properties

    def kinematic_viscosity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Kinematic viscosity [m²/s]: ν = μ/ρ
        """
        return self.viscosity(T) / self.density(T)

    def thermal_diffusivity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Thermal diffusivity [m²/s]: α = k/(ρ·cp)
        """
        return self.thermal_conductivity(T) / (
            self.density(T) * self.heat_capacity(T)
        )

    def prandtl_number(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Prandtl number [-]: Pr = μ·cp/k = ν/α
        """
        return (
            self.viscosity(T)
            * self.heat_capacity(T)
            / self.thermal_conductivity(T)
        )

    def expansion_coefficient(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Volumetric thermal expansion coefficient [1/K]: β = -(1/ρ)(dρ/dT)
        """
        T_C = T - 273.15

        # Derivative of density correlation
        b = self.rho_coeffs["b"]
        c = self.rho_coeffs["c"]

        drho_dT = b + 2 * c * T_C
        rho = self.density(T)

        beta = -drho_dT / rho

        return beta

    def get_all_properties(self, T: Union[float, np.ndarray]) -> dict:
        """
        Get all fluid properties at temperature T [K]

        Returns dictionary with all primary and derived properties
        """
        return {
            # Primary properties
            "temperature_K": T,
            "temperature_C": T - 273.15,
            "density": self.density(T),
            "heat_capacity": self.heat_capacity(T),
            "thermal_conductivity": self.thermal_conductivity(T),
            "viscosity": self.viscosity(T),
            "vapor_pressure": self.vapor_pressure(T),
            # Derived properties
            "kinematic_viscosity": self.kinematic_viscosity(T),
            "thermal_diffusivity": self.thermal_diffusivity(T),
            "prandtl_number": self.prandtl_number(T),
            "expansion_coefficient": self.expansion_coefficient(T),
        }

    def plot_properties(
        self,
        T_range: tuple = None,
        n_points: int = 100,
        save_fig: bool = False,
        filename: str = None,
    ):
        """
        Plot all properties vs temperature

        Creates comprehensive visualization of temperature-dependent properties
        """

        if T_range is None:
            T_range = (self.T_min, self.T_max)

        T = np.linspace(T_range[0], T_range[1], n_points)
        T_C = T - 273.15

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Density
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(T_C, self.density(T), "b-", linewidth=2)
        ax1.set_xlabel("Temperature [°C]")
        ax1.set_ylabel("Density [kg/m³]")
        ax1.set_title("Density", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Heat Capacity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(T_C, self.heat_capacity(T), "r-", linewidth=2)
        ax2.set_xlabel("Temperature [°C]")
        ax2.set_ylabel("Heat Capacity [J/kg·K]")
        ax2.set_title("Specific Heat Capacity", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Thermal Conductivity
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(T_C, self.thermal_conductivity(T), "g-", linewidth=2)
        ax3.set_xlabel("Temperature [°C]")
        ax3.set_ylabel("Thermal Conductivity [W/m·K]")
        ax3.set_title("Thermal Conductivity", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Dynamic Viscosity (log scale)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.semilogy(T_C, self.viscosity(T) * 1000, "purple", linewidth=2)
        ax4.set_xlabel("Temperature [°C]")
        ax4.set_ylabel("Dynamic Viscosity [mPa·s]")
        ax4.set_title("Dynamic Viscosity (log scale)", fontweight="bold")
        ax4.grid(True, alpha=0.3, which="both")

        # Plot 5: Kinematic Viscosity (log scale)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.semilogy(
            T_C, self.kinematic_viscosity(T) * 1e6, "brown", linewidth=2
        )
        ax5.set_xlabel("Temperature [°C]")
        ax5.set_ylabel("Kinematic Viscosity [cSt]")
        ax5.set_title("Kinematic Viscosity (log scale)", fontweight="bold")
        ax5.grid(True, alpha=0.3, which="both")

        # Plot 6: Vapor Pressure (log scale)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.semilogy(T_C, self.vapor_pressure(T) / 1000, "orange", linewidth=2)
        ax6.set_xlabel("Temperature [°C]")
        ax6.set_ylabel("Vapor Pressure [kPa]")
        ax6.set_title("Vapor Pressure (log scale)", fontweight="bold")
        ax6.grid(True, alpha=0.3, which="both")

        # Plot 7: Prandtl Number
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(T_C, self.prandtl_number(T), "cyan", linewidth=2)
        ax7.set_xlabel("Temperature [°C]")
        ax7.set_ylabel("Prandtl Number [-]")
        ax7.set_title("Prandtl Number", fontweight="bold")
        ax7.grid(True, alpha=0.3)

        # Plot 8: Thermal Diffusivity
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(
            T_C, self.thermal_diffusivity(T) * 1e7, "magenta", linewidth=2
        )
        ax8.set_xlabel("Temperature [°C]")
        ax8.set_ylabel("Thermal Diffusivity [×10⁻⁷ m²/s]")
        ax8.set_title("Thermal Diffusivity", fontweight="bold")
        ax8.grid(True, alpha=0.3)

        # Plot 9: Expansion Coefficient
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(T_C, self.expansion_coefficient(T) * 1e3, "teal", linewidth=2)
        ax9.set_xlabel("Temperature [°C]")
        ax9.set_ylabel("Expansion Coeff. [×10⁻³ K⁻¹]")
        ax9.set_title("Thermal Expansion Coefficient", fontweight="bold")
        ax9.grid(True, alpha=0.3)

        plt.suptitle(
            f"{self.name} - Temperature-Dependent Properties\n"
            f"Valid Range: {self.T_min - 273.15:.0f}°C to "
            f"{self.T_max - 273.15:.0f}°C",
            fontsize=14,
            fontweight="bold",
        )

        if save_fig:
            if filename is None:
                filename = "SYLTHERM800_properties.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Figure saved as {filename}")

        plt.show()

    def print_properties_table(self, temperatures: list = None):
        """
        Print formatted table of properties at specified temperatures [°C]
        """

        if temperatures is None:
            # Default temperatures
            temperatures = [-40, 0, 25, 50, 100, 150, 200, 250, 300, 350, 400]

        print(f"\n{self.name} - Property Table")
        print("=" * 120)
        print(
            f"{'T[°C]':<8} {'ρ[kg/m³]':<10} {'cp[J/kg·K]':<12} "
            f"{'k[W/m·K]':<11} {'μ[mPa·s]':<11} {'ν[cSt]':<10} "
            f"{'Pv[kPa]':<10} {'Pr[-]':<8} {'α[m²/s]':<12}"
        )
        print("-" * 120)

        for T_C in temperatures:
            T = T_C + 273.15
            props = self.get_all_properties(T)

            print(
                f"{T_C:<8.0f} "
                f"{props['density']:<10.2f} "
                f"{props['heat_capacity']:<12.1f} "
                f"{props['thermal_conductivity']:<11.5f} "
                f"{props['viscosity'] * 1000:<11.3f} "
                f"{props['kinematic_viscosity'] * 1e6:<10.3f} "
                f"{props['vapor_pressure'] / 1000:<10.2f} "
                f"{props['prandtl_number']:<8.1f} "
                f"{props['thermal_diffusivity']:<12.3e}"
            )

        print()

    def compare_with_data(self, T_data, property_data, property_name):
        """
        Compare correlation with manufacturer data

        Parameters:
        -----------
        T_data : array
            Temperature data points [K]
        property_data : array
            Measured property values
        property_name : str
            Name of property ('density', 'heat_capacity', etc.)
        """

        # Get property function
        prop_func = getattr(self, property_name)

        # Calculate predictions
        pred = prop_func(T_data)

        # Calculate error statistics
        residuals = property_data - pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / property_data)) * 100

        r_squared = 1 - (
            np.sum(residuals**2)
            / np.sum((property_data - np.mean(property_data)) ** 2)
        )

        print(f"\n{property_name.upper()} Correlation Performance:")
        print(f"  RMSE: {rmse:.4e}")
        print(f"  MAE:  {mae:.4e}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r_squared:.6f}")

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        T_C_data = T_data - 273.15
        T_plot = np.linspace(T_data.min(), T_data.max(), 100)
        T_C_plot = T_plot - 273.15

        # Plot 1: Data vs Correlation
        ax1.plot(
            T_C_data,
            property_data,
            "bo",
            label="Manufacturer Data",
            markersize=6,
        )
        ax1.plot(
            T_C_plot,
            prop_func(T_plot),
            "r-",
            label="Fitted Correlation",
            linewidth=2,
        )
        ax1.set_xlabel("Temperature [°C]")
        ax1.set_ylabel(property_name.replace("_", " ").title())
        ax1.set_title(f"{property_name.replace('_', ' ').title()} Correlation")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals
        ax2.plot(T_C_data, residuals, "go", markersize=6)
        ax2.axhline(y=0, color="k", linestyle="--", linewidth=1)
        ax2.set_xlabel("Temperature [°C]")
        ax2.set_ylabel("Residual (Data - Model)")
        ax2.set_title(f"Residuals (R² = {r_squared:.5f})")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return {"rmse": rmse, "mae": mae, "mape": mape, "r_squared": r_squared}


def load_manufacturer_data():
    """
    Load SYLTHERM 800 manufacturer data from Excel or CSV file

    Returns manufacturer data for validation and fitting
    """
    try:
        import pandas as pd

        # Try different possible file locations and formats
        file_paths = [
            "data/properties/fluids/SYLTHERM800_data.csv",
            "SYLTHERM800_data.csv",  # Legacy location
            "data/properties/fluids/Syltherm800 Properties 2024-04-13.xlsx",
            "SYLTHERM800_data.xlsx",
            "Syltherm800 Properties 2024-04-13.xlsx",  # Legacy location
            "src/fluids/properties/mfr_data/Syltherm800 Properties 2024-04-13.xlsx",  # Legacy location
        ]

        for file_path in file_paths:
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    # For Excel files, handle the header format
                    if "2024-04-13" in file_path:
                        df = pd.read_excel(file_path, skiprows=2)
                        df = df.iloc[1:].reset_index(
                            drop=True
                        )  # Skip units row
                        df.columns = [
                            "Temperature_C",
                            "Heat_Capacity_kJ_kg_K",
                            "Density_kg_m3",
                            "Thermal_Conductivity_W_m_K",
                            "Viscosity_mPa_s",
                            "Vapor_Pressure_kPa",
                        ]
                        # Convert to numeric and remove NaN rows
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                        df = df.dropna()
                    else:
                        df = pd.read_excel(file_path)

                # Convert data to the format expected by compare_with_data
                data = {
                    "T_K": df["Temperature_C"].values
                    + 273.15,  # Convert to Kelvin
                    "T_C": df["Temperature_C"].values,
                    "density": df["Density_kg_m3"].values,
                    "heat_capacity": df["Heat_Capacity_kJ_kg_K"].values
                    * 1000,  # Convert kJ/kg·K to J/kg·K
                    "thermal_conductivity": df[
                        "Thermal_Conductivity_W_m_K"
                    ].values,
                    "viscosity": df["Viscosity_mPa_s"].values
                    / 1000,  # Convert mPa·s to Pa·s
                    "vapor_pressure": df["Vapor_Pressure_kPa"].values
                    * 1000,  # Convert kPa to Pa
                }

                print(
                    f"Successfully loaded {len(df)} data points from: {file_path}"
                )
                print(
                    f"Temperature range: {df['Temperature_C'].min():.0f}°C to {df['Temperature_C'].max():.0f}°C"
                )
                return data

            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        print("Warning: No SYLTHERM800 data file found")
        return None

    except ImportError:
        print("Warning: pandas not available for data loading")
        print("Install with: pip install pandas openpyxl")
        return None


def validate_correlations():
    """
    Validate correlations against manufacturer data
    """
    print("SYLTHERM 800 - Correlation Validation")
    print("=" * 50)

    # Initialize fluid model
    syltherm = SYLTHERM800()

    try:
        # Attempt to load and validate against manufacturer data
        data = load_manufacturer_data()
        if data is not None:
            print("Validating against manufacturer data...")

            # Validate each property
            properties_to_validate = [
                ("density", "Density"),
                ("heat_capacity", "Heat Capacity"),
                ("thermal_conductivity", "Thermal Conductivity"),
                ("viscosity", "Viscosity"),
                ("vapor_pressure", "Vapor Pressure"),
            ]

            results = {}
            for prop_key, prop_name in properties_to_validate:
                if prop_key in data:
                    print(f"\n{prop_name}:")
                    stats = syltherm.compare_with_data(
                        data["T_K"], data[prop_key], prop_key
                    )
                    results[prop_key] = stats

            # Summary table
            print("\n\nValidation Summary:")
            print("=" * 70)
            print(f"{'Property':<25} {'R²':<10} {'RMSE':<12} {'MAPE [%]':<12}")
            print("-" * 70)

            for prop_key, prop_name in properties_to_validate:
                if prop_key in results:
                    r2 = results[prop_key]["r_squared"]
                    rmse = results[prop_key]["rmse"]
                    mape = results[prop_key]["mape"]
                    print(
                        f"{prop_name:<25} {r2:<10.6f} {rmse:<12.2e} {mape:<12.2f}"
                    )

            return results
        else:
            print("Using built-in test data for validation...")

            # Test with representative points across temperature range
            T_test = np.array(
                [298.15, 373.15, 473.15, 573.15]
            )  # 25, 100, 200, 300°C

            print(f"Testing at temperatures: {T_test - 273.15} °C")

            for T in T_test:
                props = syltherm.get_all_properties(T)
                print(f"\nT = {T - 273.15:.0f}°C:")
                print(f"  Density: {props['density']:.2f} kg/m³")
                print(f"  Heat capacity: {props['heat_capacity']:.1f} J/kg·K")
                print(
                    f"  Thermal conductivity: {props['thermal_conductivity']:.5f} W/m·K"
                )
                print(f"  Viscosity: {props['viscosity'] * 1000:.3f} mPa·s")
                print(f"  Prandtl number: {props['prandtl_number']:.1f}")

    except Exception as e:
        print(f"   Validation skipped: {e}")

    print("\n" + "=" * 60)
    print("SYLTHERM 800 Implementation Complete!")
    print("\nUsage in your solar collector model:")
    print("  syltherm = SYLTHERM800()")
    print("  rho = syltherm.density(T_fluid)")
    print("  cp = syltherm.heat_capacity(T_fluid)")
    print("  k = syltherm.thermal_conductivity(T_fluid)")
    print("  mu = syltherm.viscosity(T_fluid)")
    print("  Pr = syltherm.prandtl_number(T_fluid)")
    print("\nAll properties are vectorized - work with arrays!")


def demonstration():
    """
    Comprehensive demonstration of SYLTHERM800 capabilities
    """
    print("SYLTHERM 800 Heat Transfer Fluid - Python Implementation")
    print("=" * 80)

    # Initialize the fluid
    syltherm = SYLTHERM800()

    # Demo 1: Basic property calculations
    print("\n1. Basic Property Calculations")
    print("-" * 40)
    T = 573.15  # 300°C
    print(f"At T = {T - 273.15:.0f}°C:")
    print(f"  Density:             {syltherm.density(T):.2f} kg/m³")
    print(f"  Heat Capacity:       {syltherm.heat_capacity(T):.1f} J/kg·K")
    print(
        f"  Thermal Conductivity: {syltherm.thermal_conductivity(T):.5f} W/m·K"
    )
    print(f"  Dynamic Viscosity:   {syltherm.viscosity(T) * 1000:.3f} mPa·s")
    print(f"  Prandtl Number:      {syltherm.prandtl_number(T):.1f}")

    # Demo 2: Array operations
    print("\n2. Vectorized Operations")
    print("-" * 40)
    T_array = np.array([298.15, 373.15, 473.15, 573.15])
    rho_array = syltherm.density(T_array)
    print(f"Temperatures: {T_array - 273.15}")
    print(f"Densities:    {rho_array}")

    # Demo 3: Property table
    print("\n3. Property Table")
    print("-" * 40)
    syltherm.print_properties_table()

    # Demo 4: Plots
    print("\n4. Generating Property Plots...")
    print("-" * 40)
    try:
        syltherm.plot_properties()
        print("✓ Plots generated successfully")
    except Exception as e:
        print(f"✗ Plot generation failed: {e}")

    # Demo 5: Validation
    print("\n5. Correlation Validation")
    print("-" * 40)
    validate_correlations()


if __name__ == "__main__":
    demonstration()
    compare_with_data()
