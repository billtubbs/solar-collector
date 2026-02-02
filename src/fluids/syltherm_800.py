"""
SYLTHERM 800 Heat Transfer Fluid Properties

SYLTHERM 800 is a polydimethylsiloxane (PDMS) based heat transfer fluid
manufactured by Dow Chemical Company.

Chemical Composition:
- Polydimethylsiloxane (PDMS): 98.0-100.0%
- Zirconium octanoate (stabilizer): 0.5-1.5%

Physical Properties:
- Molecular Weight: ~4000 g/mol (estimated for PDMS)
- Density: 935 kg/m³ at 25°C
- Operating Range: -40°C to 400°C
"""

from typing import Dict

import pandas as pd

from .fluid_properties import (
    AntoineCorrelation,
    ExponentialCorrelation,
    FluidProperties,
    PolynomialCorrelation,
)


class Syltherm800(FluidProperties):
    """
    SYLTHERM 800 Heat Transfer Fluid Properties

    A complete fluid property model based on manufacturer data with
    temperature-dependent correlations for all major thermophysical properties.

    Properties modeled:
    - Density [kg/m³]
    - Heat capacity [J/kg·K]
    - Thermal conductivity [W/m·K]
    - Dynamic viscosity [Pa·s]
    - Vapor pressure [Pa]

    Temperature range: 233.15 K to 673.15 K (-40°C to 400°C)
    """

    def __init__(self, fit_coefficients: bool = False):
        """
        Initialize SYLTHERM 800 fluid properties

        Parameters:
        -----------
        fit_coefficients : bool, default False
            If True, fit correlations to manufacturer data on initialization
        """

        # Initialize parent class
        super().__init__(
            name="SYLTHERM 800",
            molecular_weight=4.0,  # kg/kmol (4000 g/mol estimated for PDMS)
        )

        # Operating temperature range
        self.T_min = 233.15  # K (-40°C)
        self.T_max = 673.15  # K (400°C)

        # Physical constants
        self.density_reference = 935.0  # kg/m³ at 25°C
        self.viscosity_reference = 9.16e-3  # Pa·s at 25°C (9.8 cSt × 0.935)

        # Initialize correlations with placeholder coefficients
        self._setup_correlations()

        # Fit to manufacturer data if requested
        if fit_coefficients:
            self.fit_to_manufacturer_data()

    def _setup_correlations(self):
        """
        Set up correlation objects with placeholder coefficients.
        These will be fitted to manufacturer data later.
        """

        # Density correlation: ρ(T) = a₀ + a₁T + a₂T² (DIPPR100)
        # For PDMS: decreases linearly with temperature
        density_coeffs = [1000.0, -0.5, 0.0]  # Placeholder coefficients
        self.set_density(
            PolynomialCorrelation(
                coefficients=density_coeffs,
                T_min=self.T_min,
                T_max=self.T_max,
                name="Density_PDMS",
            )
        )

        # Heat capacity correlation: Cp(T) = a₀ + a₁T + a₂T² (DIPPR100)
        # For PDMS: increases with temperature
        heat_capacity_coeffs = [1500.0, 2.0, 0.0]  # Placeholder coefficients
        self.set_heat_capacity(
            PolynomialCorrelation(
                coefficients=heat_capacity_coeffs,
                T_min=self.T_min,
                T_max=self.T_max,
                name="HeatCapacity_PDMS",
            )
        )

        # Thermal conductivity correlation: k(T) = a₀ + a₁T (DIPPR100)
        # For PDMS: slightly decreases with temperature
        thermal_conductivity_coeffs = [
            0.15,
            -0.0001,
        ]  # Placeholder coefficients
        self.set_thermal_conductivity(
            PolynomialCorrelation(
                coefficients=thermal_conductivity_coeffs,
                T_min=self.T_min,
                T_max=self.T_max,
                name="ThermalConductivity_PDMS",
            )
        )

        # Viscosity correlation: μ(T) = A * exp(B/T) + C (DIPPR101)
        # Andrade equation - exponential decrease with temperature
        viscosity_params = {
            "A": 1e-6,  # Pre-exponential factor [Pa·s]
            "B": 2000.0,  # Activation energy parameter [K]
            "C": 0.0,  # Offset [Pa·s]
        }
        self.set_viscosity(
            ExponentialCorrelation(
                A=viscosity_params["A"],
                B=viscosity_params["B"],
                C=viscosity_params["C"],
                T_min=self.T_min,
                T_max=self.T_max,
                name="Viscosity_Andrade",
            )
        )

        # Vapor pressure correlation:
        #   log₁₀(P[Pa]) = A + B/(T[K] + C) (DIPPR101)
        # Antoine equation for PDMS
        vapor_pressure_params = {
            "A": 8.0,  # Antoine constant
            "B": -2500.0,  # Antoine constant [K]
            "C": -50.0,  # Antoine constant [K]
        }
        self.set_vapor_pressure(
            AntoineCorrelation(
                A=vapor_pressure_params["A"],
                B=vapor_pressure_params["B"],
                C=vapor_pressure_params["C"],
                T_min=self.T_min,
                T_max=self.T_max,
                name="VaporPressure_Antoine",
            )
        )

    def load_manufacturer_data(self, file_path: str = None) -> Dict:
        """
        Load manufacturer data from CSV file

        Parameters:
        -----------
        file_path : str, optional
            Path to manufacturer data CSV file
            If None, uses default location

        Returns:
        --------
        dict
            Manufacturer data with keys: T_K, T_C, density, heat_capacity,
            thermal_conductivity, viscosity, vapor_pressure
        """
        if file_path is None:
            # Default file locations
            default_paths = [
                "data/properties/fluids/SYLTHERM800_data.csv",
                "SYLTHERM800_data.csv",
                "../data/properties/fluids/SYLTHERM800_data.csv",
            ]

            for path in default_paths:
                try:
                    df = pd.read_csv(path)
                    file_path = path
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError(
                    "SYLTHERM 800 manufacturer data file not found. "
                    "Tried: " + ", ".join(default_paths)
                )
        else:
            df = pd.read_csv(file_path)

        # Convert data to standard units
        data = {
            "T_K": df["Temperature_C"].values + 273.15,  # Convert to Kelvin
            "T_C": df["Temperature_C"].values,
            "density": df["Density_kg_m3"].values,  # Already kg/m³
            "heat_capacity": df["Heat_Capacity_kJ_kg_K"].values
            * 1000,  # Convert to J/kg·K
            "thermal_conductivity": df[
                "Thermal_Conductivity_W_m_K"
            ].values,  # Already W/m·K
            "viscosity": df["Viscosity_mPa_s"].values
            / 1000,  # Convert to Pa·s
            "vapor_pressure": df["Vapor_Pressure_kPa"].values
            * 1000,  # Convert to Pa
        }

        print(
            f"Loaded {len(data['T_K'])} manufacturer data points "
            f"from: {file_path}"
        )
        print(
            f"Temperature range: {data['T_C'].min():.0f}°C to "
            f"{data['T_C'].max():.0f}°C"
        )

        return data

    def fit_to_manufacturer_data(self, data: Dict = None) -> Dict:
        """
        Fit correlation coefficients to manufacturer data

        Parameters:
        -----------
        data : dict, optional
            Manufacturer data. If None, loads from default file

        Returns:
        --------
        dict
            Fitting statistics for each property
        """
        if data is None:
            data = self.load_manufacturer_data()

        fitting_results = {}

        # TODO: Implement actual fitting algorithms
        # This is a placeholder for the fitting methods we'll develop

        print("Fitting correlations to manufacturer data...")
        print("NOTE: Actual fitting algorithms to be implemented")

        return fitting_results

    def validate_correlations(self, data: Dict = None) -> Dict:
        """
        Validate fitted correlations against manufacturer data

        Parameters:
        -----------
        data : dict, optional
            Manufacturer data for validation

        Returns:
        --------
        dict
            Validation statistics (RMSE, MAE, MAPE, R²) for each property
        """
        if data is None:
            data = self.load_manufacturer_data()

        # Use the compare_with_data method from parent class
        property_data = {
            "density": data["density"],
            "heat_capacity": data["heat_capacity"],
            "thermal_conductivity": data["thermal_conductivity"],
            "viscosity": data["viscosity"],
            "vapor_pressure": data["vapor_pressure"],
        }

        return self.compare_with_data(
            T_data=data["T_K"],
            property_data=property_data,
            save_fig=True,
            filename="SYLTHERM800_correlation_validation.png",
        )

    def get_reference_properties(self) -> Dict:
        """
        Get reference properties at standard conditions (25°C, 1 atm)

        Returns:
        --------
        dict
            Reference property values
        """
        T_ref = 298.15  # 25°C in Kelvin

        return {
            "temperature_C": 25.0,
            "temperature_K": T_ref,
            "density": self.density(T_ref),
            "heat_capacity": self.heat_capacity(T_ref),
            "thermal_conductivity": self.thermal_conductivity(T_ref),
            "viscosity": self.viscosity(T_ref),
            "kinematic_viscosity": self.kinematic_viscosity(T_ref),
            "vapor_pressure": self.vapor_pressure(T_ref),
            "prandtl_number": self.prandtl_number(T_ref),
            "thermal_diffusivity": self.thermal_diffusivity(T_ref),
        }

    def summary(self):
        """Print summary of SYLTHERM 800 properties and correlations"""
        print(f"\\n{self.name} Heat Transfer Fluid")
        print("=" * 60)
        print("Chemical: Polydimethylsiloxane (PDMS)")
        print(f"Molecular Weight: {self.molecular_weight:.0f} kg/kmol")
        print(
            f"Operating Range: {self.T_min - 273.15:.0f}°C to "
            f"{self.T_max - 273.15:.0f}°C"
        )

        print("\\nProperty Correlations:")
        for prop_name, correlation in self.correlations.items():
            print(f"  {prop_name}: {correlation.name}")

        print("\\nReference Properties (25°C):")
        ref_props = self.get_reference_properties()

        # Common format string for consistent alignment
        prop_format = "  {name:>21s}: {value} {units}"

        print(
            prop_format.format(
                name="Density",
                value=f"{ref_props['density']:.1f}",
                units="kg/m³",
            )
        )
        print(
            prop_format.format(
                name="Heat Capacity",
                value=f"{ref_props['heat_capacity']:.0f}",
                units="J/kg·K",
            )
        )
        print(
            prop_format.format(
                name="Thermal Conductivity",
                value=f"{ref_props['thermal_conductivity']:.5f}",
                units="W/m·K",
            )
        )
        print(
            prop_format.format(
                name="Dynamic Viscosity",
                value=f"{ref_props['viscosity'] * 1000:.2f}",
                units="mPa·s",
            )
        )
        print(
            prop_format.format(
                name="Kinematic Viscosity",
                value=f"{ref_props['kinematic_viscosity'] * 1e6:.2f}",
                units="cSt",
            )
        )
        print(
            prop_format.format(
                name="Vapor Pressure",
                value=f"{ref_props['vapor_pressure']:.2f}",
                units="Pa",
            )
        )
        print(
            prop_format.format(
                name="Prandtl Number",
                value=f"{ref_props['prandtl_number']:.1f}",
                units="",
            )
        )


def create_syltherm800(fit_data: bool = True) -> Syltherm800:
    """
    Convenience function to create and initialize SYLTHERM 800 fluid

    Parameters:
    -----------
    fit_data : bool, default True
        Whether to fit correlations to manufacturer data

    Returns:
    --------
    Syltherm800
        Initialized SYLTHERM 800 fluid object
    """
    return Syltherm800(fit_coefficients=fit_data)
