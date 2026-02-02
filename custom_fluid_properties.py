from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class FluidProperties:
    """
    Base class for temperature-dependent fluid properties.

    Provides methods for computing thermophysical properties from
    polynomial and exponential correlations.

    Parameters
    ----------
    name : str
        Name of the fluid
    description : str
        Description of the fluid
    T_min : float
        Minimum valid temperature [K]
    T_max : float
        Maximum valid temperature [K]
    T_ref : float
        Reference temperature [K]
    coeffs : dict
        Nested dictionary of correlation coefficients with keys:
        - "density": {"a", "b", "c"} for ρ = a + b*T + c*T² [kg/m³] with T in K
        - "heat_capacity": {"a", "b"} for cp = a + b*T [J/kg·K] with T in K
        - "thermal_conductivity": {"a", "b"} for k = a + b*T [W/m·K] with T in K
        - "viscosity": {"A", "B", "C"} for μ = A*exp(B/T) + C [Pa·s] with T in K
        - "vapor_pressure": {"A", "B", "C"} for log10(P_Pa) = A - B/(C + T_C) with T_C in °C
    """

    def __init__(self, name, description, T_min, T_max, T_ref, coeffs):
        self.name = name
        self.description = description
        self.T_min = T_min
        self.T_max = T_max
        self.T_ref = T_ref
        self.coeffs = coeffs

    def _check_temperature_range(self, T):
        """Check if temperature is in valid range and warn if not"""
        T_array = np.atleast_1d(T)
        if np.any(T_array < self.T_min) or np.any(T_array > self.T_max):
            T_min_C = self.T_min - 273.15
            T_max_C = self.T_max - 273.15
            print(
                f"Warning: Temperature outside valid range "
                f"({T_min_C:.0f}°C to {T_max_C:.0f}°C)"
            )

    def density(self, T):
        """
        Density [kg/m³] as function of temperature [K]

        Correlation: ρ = a + b*T + c*T²

        Parameters
        ----------
        T : float or array
            Temperature in Kelvin

        Returns
        -------
        rho : float or array
            Density in kg/m³
        """
        self._check_temperature_range(T)
        coeffs = self.coeffs["density"]
        rho = coeffs["a"] + coeffs["b"] * T + coeffs["c"] * T**2
        return rho

    def heat_capacity(self, T):
        """
        Specific heat capacity [J/kg·K] as function of temperature [K]

        Correlation: cp = a + b*T
        """
        self._check_temperature_range(T)
        coeffs = self.coeffs["heat_capacity"]
        cp = coeffs["a"] + coeffs["b"] * T
        return cp

    def thermal_conductivity(self, T):
        """
        Thermal conductivity [W/m·K] as function of temperature [K]

        Correlation: k = a + b*T
        """
        self._check_temperature_range(T)
        coeffs = self.coeffs["thermal_conductivity"]
        k = coeffs["a"] + coeffs["b"] * T
        return k

    def viscosity(self, T):
        """
        Dynamic viscosity [Pa·s] as function of temperature [K]

        Correlation: μ = A * exp(B/T) + C (Andrade equation with offset)
        """
        self._check_temperature_range(T)
        coeffs = self.coeffs["viscosity"]
        mu = coeffs["A"] * np.exp(coeffs["B"] / T) + coeffs["C"]
        return mu

    def vapor_pressure(self, T):
        """
        Vapor pressure [Pa] as function of temperature [K]

        Correlation: log10(P_Pa) = A - B/(C + T_C) (Antoine equation)
        where T_C is temperature in Celsius
        """
        self._check_temperature_range(T)
        coeffs = self.coeffs["vapor_pressure"]

        # Antoine equation (T must be converted to Celsius)
        T_C = T - 273.15
        log_P_Pa = coeffs["A"] - coeffs["B"] / (coeffs["C"] + T_C)
        P_Pa = 10**log_P_Pa

        # Handle very low temperatures where pressure is negligible
        P_Pa = np.maximum(P_Pa, 0.01)  # Minimum 0.01 Pa
        return P_Pa

    def kinematic_viscosity(self, T):
        """
        Kinematic viscosity [m²/s]: ν = μ/ρ
        """
        return self.viscosity(T) / self.density(T)

    def thermal_diffusivity(self, T):
        """
        Thermal diffusivity [m²/s]: α = k/(ρ·cp)
        """
        return self.thermal_conductivity(T) / (
            self.density(T) * self.heat_capacity(T)
        )

    def prandtl_number(self, T):
        """
        Prandtl number [-]: Pr = μ·cp/k = ν/α
        """
        return (
            self.viscosity(T)
            * self.heat_capacity(T)
            / self.thermal_conductivity(T)
        )

    def expansion_coefficient(self, T):
        """
        Volumetric thermal expansion coefficient [1/K]: β = -(1/ρ)(dρ/dT)
        """
        # Derivative of quadratic density correlation: dρ/dT = b + 2*c*T
        coeffs = self.coeffs["density"]
        drho_dT = coeffs["b"] + 2 * coeffs["c"] * T
        rho = self.density(T)
        beta = -drho_dT / rho
        return beta

    def get_all_properties(self, T):
        """
        Get all fluid properties at temperature T [K]

        Returns dictionary with all primary and derived properties
        """
        return {
            "temperature_K": T,
            "temperature_C": T - 273.15,
            "density": self.density(T),
            "heat_capacity": self.heat_capacity(T),
            "thermal_conductivity": self.thermal_conductivity(T),
            "viscosity": self.viscosity(T),
            "vapor_pressure": self.vapor_pressure(T),
            "kinematic_viscosity": self.kinematic_viscosity(T),
            "thermal_diffusivity": self.thermal_diffusivity(T),
            "prandtl_number": self.prandtl_number(T),
            "expansion_coefficient": self.expansion_coefficient(T),
        }


class SYLTHERM800(FluidProperties):
    """
    SYLTHERM 800 Heat Transfer Fluid Properties

    Based on manufacturer data from Dow Chemical
    Data source: https://www.dow.com/en-us/pdp.syltherm-800-heat-transfer-fluid.html

    Valid range: 200°C to 400°C (473.15 K to 673.15 K)

    All correlations fitted to manufacturer specification data using
    fit_fluid_property_correlations.ipynb
    """

    def __init__(self):
        coeffs = {
            # Density: ρ = a + b*T [kg/m³] with T in K
            # "density": {"a": 1312.344, "b": -1.12593},
            # Density: ρ = a + b*T + c*T^2 [kg/m³] with T in K
            "density": {"a": 960.73, "b": 0.11489, "c": -0.00108245},
            # Heat capacity: cp = a + b*T [J/kg·K] with T in K
            "heat_capacity": {"a": 1108.027, "b": 1.70714},
            # Thermal conductivity: k = a + b*T [W/m·K] with T in K
            "thermal_conductivity": {"a": 0.19091, "b": -0.000189399},
            # Viscosity: μ = A*exp(B/T) + C [Pa·s] with T in K
            "viscosity": {"A": 3.9406e-5, "B": 1637.0, "C": -2.115e-4},
            # Vapor pressure: log10(P_Pa) = A + B/(T + C) with T in K
            "vapor_pressure": {"A": 7.9938, "B": 964.133, "C": 119.431},
        }

        super().__init__(
            name="SYLTHERM 800",
            description="Silicone-based heat transfer fluid (Dow Chemical)",
            T_min=473.15,  # 200°C
            T_max=673.15,  # 400°C
            T_ref=573.15,  # 300°C (reference)
            coeffs=coeffs,
        )


def plot_properties(
    fluid,
    T_range=None,
    n_points=100,
    save_fig=False,
    filename=None,
):
    """
    Plot all properties vs temperature

    Creates comprehensive visualization of temperature-dependent properties
    """

    if T_range is None:
        T_range = (fluid.T_min, fluid.T_max)

    T = np.linspace(T_range[0], T_range[1], n_points)
    T_C = T - 273.15

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Density
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(T_C, fluid.density(T), "b-", linewidth=2)
    ax1.set_xlabel("Temperature [°C]")
    ax1.set_ylabel("Density [kg/m³]")
    ax1.set_title("Density", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heat Capacity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(T_C, fluid.heat_capacity(T), "r-", linewidth=2)
    ax2.set_xlabel("Temperature [°C]")
    ax2.set_ylabel("Heat Capacity [J/kg·K]")
    ax2.set_title("Specific Heat Capacity", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Thermal Conductivity
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(T_C, fluid.thermal_conductivity(T), "g-", linewidth=2)
    ax3.set_xlabel("Temperature [°C]")
    ax3.set_ylabel("Thermal Conductivity [W/m·K]")
    ax3.set_title("Thermal Conductivity", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Dynamic Viscosity (log scale)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(T_C, fluid.viscosity(T) * 1000, "purple", linewidth=2)
    ax4.set_xlabel("Temperature [°C]")
    ax4.set_ylabel("Dynamic Viscosity [mPa·s]")
    ax4.set_title("Dynamic Viscosity (log scale)", fontweight="bold")
    ax4.grid(True, alpha=0.3, which="both")

    # Plot 5: Kinematic Viscosity (log scale)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogy(T_C, fluid.kinematic_viscosity(T) * 1e6, "brown", linewidth=2)
    ax5.set_xlabel("Temperature [°C]")
    ax5.set_ylabel("Kinematic Viscosity [cSt]")
    ax5.set_title("Kinematic Viscosity (log scale)", fontweight="bold")
    ax5.grid(True, alpha=0.3, which="both")

    # Plot 6: Vapor Pressure (log scale)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(T_C, fluid.vapor_pressure(T) / 1000, "orange", linewidth=2)
    ax6.set_xlabel("Temperature [°C]")
    ax6.set_ylabel("Vapor Pressure [kPa]")
    ax6.set_title("Vapor Pressure (log scale)", fontweight="bold")
    ax6.grid(True, alpha=0.3, which="both")

    # Plot 7: Prandtl Number
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(T_C, fluid.prandtl_number(T), "cyan", linewidth=2)
    ax7.set_xlabel("Temperature [°C]")
    ax7.set_ylabel("Prandtl Number [-]")
    ax7.set_title("Prandtl Number", fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # Plot 8: Thermal Diffusivity
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(T_C, fluid.thermal_diffusivity(T) * 1e7, "magenta", linewidth=2)
    ax8.set_xlabel("Temperature [°C]")
    ax8.set_ylabel("Thermal Diffusivity [×10⁻⁷ m²/s]")
    ax8.set_title("Thermal Diffusivity", fontweight="bold")
    ax8.grid(True, alpha=0.3)

    # Plot 9: Expansion Coefficient
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(T_C, fluid.expansion_coefficient(T) * 1e3, "teal", linewidth=2)
    ax9.set_xlabel("Temperature [°C]")
    ax9.set_ylabel("Expansion Coeff. [×10⁻³ K⁻¹]")
    ax9.set_title("Thermal Expansion Coefficient", fontweight="bold")
    ax9.grid(True, alpha=0.3)

    plt.suptitle(
        f"{fluid.name} - Temperature-Dependent Properties\n"
        f"Valid Range: {fluid.T_min - 273.15:.0f}°C to "
        f"{fluid.T_max - 273.15:.0f}°C",
        fontsize=14,
        fontweight="bold",
    )

    if save_fig:
        if filename is None:
            filename = "SYLTHERM800_properties.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Figure saved as {filename}")

    plt.show()


def print_properties_table(fluid, temperatures=None):
    """
    Print formatted table of properties at specified temperatures [°C]

    Parameters
    ----------
    fluid : FluidProperties
        Fluid properties object
    temperatures : list, optional
        List of temperatures in °C to include in table
    """

    if temperatures is None:
        # Default temperatures
        temperatures = [200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]

    print(f"\n{fluid.name} - Property Table")
    print("=" * 120)
    print(
        f"{'T[°C]':<8} {'ρ[kg/m³]':<10} {'cp[J/kg·K]':<12} "
        f"{'k[W/m·K]':<11} {'μ[mPa·s]':<11} {'ν[cSt]':<10} "
        f"{'Pv[kPa]':<10} {'Pr[-]':<8} {'α[m²/s]':<12}"
    )
    print("-" * 120)

    for T_C in temperatures:
        T = T_C + 273.15
        props = fluid.get_all_properties(T)

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


def make_comparison_plot(
    fluid,
    T_data,
    property_data,
    property_name,
    plot_dir="plots",
    figsize=(6, 3),
    dpi=300,
):
    """
    Make plots to compare the model predictions with manufacturer data.

    Produces two separate figures: one for the model comparison and one
    for the residuals.

    Parameters
    ----------
    fluid : FluidProperties
        Fluid properties object
    T_data : array
        Temperature data points [K]
    property_data : array
        Measured property values
    property_name : str
        Name of property ('density', 'heat_capacity', etc.)
    figsize : tuple
        Figure size for each plot
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Units for each property
    units = {
        "density": "kg/m³",
        "heat_capacity": "J/kg·K",
        "thermal_conductivity": "W/m·K",
        "viscosity": "Pa·s",
        "vapor_pressure": "Pa",
    }
    unit = units.get(property_name, "")

    # Get property function
    prop_func = getattr(fluid, property_name)

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

    print(f"\n{property_name.upper()} Model Performance:")
    print(f"  RMSE: {rmse:.4e}")
    print(f"  MAE:  {mae:.4e}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r_squared:.6f}")

    # Prepare data for plotting
    T_C_data = T_data - 273.15
    T_plot = np.linspace(T_data.min(), T_data.max(), 100)
    T_C_plot = T_plot - 273.15

    # Valid temperature range in Celsius
    T_min_C = fluid.T_min - 273.15
    T_max_C = fluid.T_max - 273.15

    # Property name for labels
    prop_label = property_name.replace("_", " ").title()
    ylabel = f"{prop_label} [{unit}]" if unit else prop_label

    # --- Plot 1: Model comparison ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.axvspan(T_min_C, T_max_C, alpha=0.15, color="green")
    ax1.plot(T_C_data, property_data, "bo", label="data", markersize=6)
    ax1.plot(T_C_plot, prop_func(T_plot), "r-", label="model", linewidth=2)
    ax1.set_xlabel("Temperature [°C]")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{prop_label} Model (RMSE = {rmse:.2e})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add valid range indicator near top of plot
    y_min, y_max = ax1.get_ylim()
    y_pos = y_min + 0.9 * (y_max - y_min)
    ax1.annotate(
        "",
        xy=(T_max_C, y_pos),
        xytext=(T_min_C, y_pos),
        arrowprops={
            "arrowstyle": "<->",
            "color": "green",
            "lw": 1,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    T_mid_C = (T_min_C + T_max_C) / 2
    y_text = y_pos + 0.02 * (y_max - y_min)
    ax1.text(
        T_mid_C,
        y_text,
        "Valid temperature range",
        ha="center",
        va="bottom",
        fontsize=8,
        color="green",
    )

    plt.tight_layout()
    filename = f"{fluid.name}_{property_name}_comparison.png".replace(" ", "_")
    plt.savefig(plot_dir / filename, dpi=dpi)
    plt.show()

    # --- Plot 2: Residuals ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.axvspan(T_min_C, T_max_C, alpha=0.15, color="green")
    ax2.plot(T_C_data, residuals, "go", markersize=6)
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=1)
    ax2.set_xlabel("Temperature [°C]")
    ax2.set_ylabel(f"Residual [{unit}]" if unit else "Residual")
    ax2.set_title(f"{prop_label} Residuals (RMSE = {rmse:.2e})")
    ax2.grid(True, alpha=0.3)

    # Add valid range indicator near top of plot
    y_min, y_max = ax2.get_ylim()
    y_pos = y_min + 0.9 * (y_max - y_min)
    ax2.annotate(
        "",
        xy=(T_max_C, y_pos),
        xytext=(T_min_C, y_pos),
        arrowprops={
            "arrowstyle": "<->",
            "color": "green",
            "lw": 1,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    T_mid_C = (T_min_C + T_max_C) / 2
    y_text = y_pos + 0.02 * (y_max - y_min)
    ax2.text(
        T_mid_C,
        y_text,
        "Valid temperature range",
        ha="center",
        va="bottom",
        fontsize=8,
        color="green",
    )

    plt.tight_layout()
    filename = f"{fluid.name}_{property_name}_residuals.png".replace(" ", "_")
    plt.savefig(plot_dir / filename, dpi=dpi)
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


def validate_correlations(fluid, data):
    """
    Validate correlations against manufacturer data
    """
    print(f"{fluid.name} - Correlation Validation")
    print("=" * 50)

    # Temperature range for plots (with margin beyond valid range)
    T_min_plot = fluid.T_min - 50
    T_max_plot = fluid.T_max + 50

    # Filter data to only include points within the plot range
    T_K = data["T_K"]
    mask = (T_K >= T_min_plot) & (T_K <= T_max_plot)
    filtered_data = {key: val[mask] for key, val in data.items()}

    print("Validating against manufacturer data...")
    print(
        f"Temperature range: {T_min_plot - 273.15:.0f}°C to "
        f"{T_max_plot - 273.15:.0f}°C"
    )
    print(f"Data points in range: {mask.sum()}")

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
        if prop_key in filtered_data:
            print(f"\n{prop_name}:")
            stats = make_comparison_plot(
                fluid, filtered_data["T_K"], filtered_data[prop_key], prop_key
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
            print(f"{prop_name:<25} {r2:<10.6f} {rmse:<12.2e} {mape:<12.2f}")

    return results


def demonstration():
    """
    Comprehensive demonstration of SYLTHERM800 capabilities
    """
    print("SYLTHERM 800 Heat Transfer Fluid - Python Implementation")
    print("=" * 80)

    # Initialize the fluid
    fluid = SYLTHERM800()

    # Demo 1: Basic property calculations
    print("\n1. Basic Property Calculations")
    print("-" * 40)
    T = 573.15  # 300°C
    print(f"At T = {T - 273.15:.0f}°C:")
    print(f"  Density:             {fluid.density(T):.2f} kg/m³")
    print(f"  Heat Capacity:       {fluid.heat_capacity(T):.1f} J/kg·K")
    print(f"  Thermal Conductivity: {fluid.thermal_conductivity(T):.5f} W/m·K")
    print(f"  Dynamic Viscosity:   {fluid.viscosity(T) * 1000:.3f} mPa·s")
    print(f"  Prandtl Number:      {fluid.prandtl_number(T):.1f}")

    # Demo 2: Array operations
    print("\n2. Vectorized Operations")
    print("-" * 40)
    T_array = np.array([298.15, 373.15, 473.15, 573.15])
    rho_array = fluid.density(T_array)
    print(f"Temperatures: {T_array - 273.15}")
    print(f"Densities:    {rho_array}")

    # Demo 3: Property table
    print("\n3. Property Table")
    print("-" * 40)
    print_properties_table(fluid)

    # Demo 4: Plots
    print("\n4. Generating Property Plots...")
    print("-" * 40)
    plot_properties(fluid)

    # Demo 5: Validation
    print("\n5. Correlation Validation")
    print("-" * 40)
    # Attempt to load and validate against manufacturer data
    data = load_manufacturer_data()
    validate_correlations(fluid, data)


if __name__ == "__main__":
    demonstration()
