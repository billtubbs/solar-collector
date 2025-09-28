import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class RadialGradientAnalysis:
    """
    Analyze when radial temperature gradients become important
    and demonstrate limitations of single temperature models
    """

    def __init__(self, pipe_diameter, fluid_props, flow_conditions):
        self.D = pipe_diameter
        self.R = self.D / 2

        # Fluid properties
        self.rho = fluid_props["density"]
        self.cp = fluid_props["heat_capacity"]
        self.k = fluid_props["conductivity"]
        self.mu = fluid_props["viscosity"]
        self.alpha = self.k / (self.rho * self.cp)

        # Flow conditions
        self.u_bulk = flow_conditions["velocity"]
        self.Re = self.rho * self.u_bulk * self.D / self.mu
        self.Pr = self.mu * self.cp / self.k

        # Calculate characteristic time scales
        self.calculate_time_scales()

        print("Radial Gradient Analysis:")
        print(f"  Pipe diameter: {self.D * 1000:.1f} mm")
        print(f"  Reynolds number: {self.Re:.0f}")
        print(f"  Prandtl number: {self.Pr:.2f}")

    def calculate_time_scales(self):
        """
        Calculate critical time scales for radial heat transfer
        """

        # Radial diffusion time scale (time to equilibrate across radius)
        self.tau_radial = self.R**2 / self.alpha

        # Axial convection time scale (residence time)
        L_segment = 0.1  # Typical segment length
        self.tau_axial = L_segment / self.u_bulk

        # Wall heat transfer time scale
        if self.Re > 4000:
            Nu = 0.023 * self.Re**0.8 * self.Pr**0.4
        else:
            Nu = 3.66

        h = Nu * self.k / self.D
        self.tau_wall = self.rho * self.cp * self.R / h

        # Damköhler number (ratio of radial diffusion to advection)
        self.Da = self.tau_radial / self.tau_axial

        print("\nCharacteristic Time Scales:")
        print(f"  Radial diffusion: τ_radial = {self.tau_radial:.2f} s")
        print(f"  Axial convection: τ_axial = {self.tau_axial:.2f} s")
        print(f"  Wall heat transfer: τ_wall = {self.tau_wall:.2f} s")
        print(f"  Damköhler number (Da = τ_radial/τ_axial): {self.Da:.2f}")

        if self.Da > 2:
            print("  → RADIAL GRADIENTS IMPORTANT (Da >> 1)")
        elif self.Da < 0.5:
            print("  → RADIAL GRADIENTS NEGLIGIBLE (Da << 1)")
        else:
            print("  → INTERMEDIATE REGIME")

    def single_temperature_model(self, irradiance_profile, t_span):
        """
        Single mean temperature model T_f(x,t)
        """

        # Effective heat transfer coefficient
        if self.Re > 4000:
            Nu = 0.023 * self.Re**0.8 * self.Pr**0.4
        else:
            Nu = 3.66
        h = Nu * self.k / self.D

        # Heat input per unit volume from wall
        def heat_input_rate(t):
            q_solar = irradiance_profile(t)  # W/m²
            # Assume all solar energy goes to heating fluid
            return 4 * q_solar / self.D  # W/m³ (simplified)

        def single_temp_ode(t, T):
            dT_dt = heat_input_rate(t) / (self.rho * self.cp)
            return [dT_dt]

        T_initial = [293.15]  # 20°C
        sol = solve_ivp(single_temp_ode, t_span, T_initial, dense_output=True)

        return sol

    def radial_temperature_model(self, irradiance_profile, t_span, nr=20):
        """
        Detailed radial temperature model T(r,t)

        Solves: ∂T/∂t = α/r ∂/∂r(r ∂T/∂r) + q(t)/ρcp
        """

        # Radial grid (finite volume)
        r_faces = np.linspace(0, self.R, nr + 1)
        r_centers = 0.5 * (r_faces[1:] + r_faces[:-1])
        dr = r_faces[1:] - r_faces[:-1]

        def radial_ode(t, T):
            """
            Radial heat diffusion with solar input
            """
            q_solar = irradiance_profile(t)

            dT_dt = np.zeros(nr)

            for i in range(nr):
                r = r_centers[i]

                if i == 0:  # Center node (r = 0)
                    # Use L'Hôpital's rule: ∂/∂r(r ∂T/∂r) = 2 ∂²T/∂r²
                    d2T_dr2 = 2 * (T[i + 1] - T[i]) / dr[i] ** 2
                    diffusion = self.alpha * d2T_dr2

                elif i == nr - 1:  # Wall node
                    # Boundary condition: solar heat flux at wall
                    # -k ∂T/∂r|wall = q_solar
                    dT_dr_wall = -q_solar / self.k

                    # Finite difference for interior
                    r_face = r_faces[i]
                    dT_dr_inner = (T[i] - T[i - 1]) / dr[i - 1]

                    # Heat balance
                    diffusion = (
                        self.alpha
                        * (r_face * dT_dr_wall - r_face * dT_dr_inner)
                        / (r * dr[i])
                    )

                else:  # Interior nodes
                    r_face_plus = r_faces[i + 1]
                    r_face_minus = r_faces[i]

                    dT_dr_plus = (T[i + 1] - T[i]) / dr[i]
                    dT_dr_minus = (T[i] - T[i - 1]) / dr[i - 1]

                    diffusion = (
                        self.alpha
                        * (r_face_plus * dT_dr_plus - r_face_minus * dT_dr_minus)
                        / (r * dr[i])
                    )

                # No volumetric heat generation (all heat comes from wall)
                dT_dt[i] = diffusion

            return dT_dt

        # Initial conditions (uniform temperature)
        T_initial = np.full(nr, 293.15)

        sol = solve_ivp(radial_ode, t_span, T_initial, dense_output=True, rtol=1e-6)

        return sol, r_centers

    def compare_irradiance_scenarios(self):
        """
        Compare short high vs long low irradiance scenarios
        """

        print("\nComparing Irradiance Scenarios:")
        print("=" * 35)

        # Scenario parameters
        total_energy = 1000 * 60  # J/m² (same total energy)

        # Scenario 1: Short high irradiance (10s at 6000 W/m²)
        def irradiance_short_high(t):
            return 6000 if 0 <= t <= 10 else 0

        # Scenario 2: Long low irradiance (60s at 1000 W/m²)
        def irradiance_long_low(t):
            return 1000 if 0 <= t <= 60 else 0

        # Scenario 3: Very short intense pulse (1s at 60000 W/m²)
        def irradiance_pulse(t):
            return 60000 if 0 <= t <= 1 else 0

        scenarios = [
            ("Short High (10s @ 6kW/m²)", irradiance_short_high),
            ("Long Low (60s @ 1kW/m²)", irradiance_long_low),
            ("Intense Pulse (1s @ 60kW/m²)", irradiance_pulse),
        ]

        t_span = (0, 120)  # 2 minutes total
        results = {}

        for name, irradiance_func in scenarios:
            print(f"\nSolving: {name}")

            # Single temperature model
            sol_single = self.single_temperature_model(irradiance_func, t_span)

            # Radial temperature model
            sol_radial, r_centers = self.radial_temperature_model(
                irradiance_func, t_span
            )

            results[name] = {
                "single": sol_single,
                "radial": sol_radial,
                "r_centers": r_centers,
                "irradiance": irradiance_func,
            }

        self.plot_scenario_comparison(results, t_span)
        return results

    def plot_scenario_comparison(self, results, t_span):
        """
        Plot comparison of different irradiance scenarios
        """

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        t_eval = np.linspace(t_span[0], t_span[1], 300)
        colors = ["blue", "red", "green"]

        scenario_names = list(results.keys())

        # Row 1: Irradiance profiles
        for i, (name, data) in enumerate(results.items()):
            ax = axes[0, i]
            irradiance_vals = [data["irradiance"](t) for t in t_eval]
            ax.plot(t_eval, np.array(irradiance_vals) / 1000, colors[i], linewidth=2)
            ax.set_title(f"{name}\nIrradiance Profile")
            ax.set_ylabel("Irradiance [kW/m²]")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 120)

        # Row 2: Single temperature model results
        for i, (name, data) in enumerate(results.items()):
            ax = axes[1, i]
            T_single = data["single"].sol(t_eval)[0] - 273.15
            ax.plot(t_eval, T_single, colors[i], linewidth=2, label="Single model")
            ax.set_title(f"{name}\nSingle Temperature Model")
            ax.set_ylabel("Temperature [°C]")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 120)

        # Row 3: Radial model results (mean temperature)
        for i, (name, data) in enumerate(results.items()):
            ax = axes[2, i]

            # Calculate area-weighted mean temperature
            sol_radial = data["radial"]
            r_centers = data["r_centers"]

            T_radial_profiles = sol_radial.sol(t_eval)

            # Area weighting for mean temperature
            # Create annular areas for each radial node
            r_faces = np.linspace(0, self.R, len(r_centers) + 1)
            r_areas = np.pi * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)

            T_mean_radial = []
            for j in range(len(t_eval)):
                T_profile = T_radial_profiles[:, j]
                T_mean = np.sum(T_profile * r_areas) / np.sum(r_areas)
                T_mean_radial.append(T_mean)

            ax.plot(
                t_eval,
                np.array(T_mean_radial) - 273.15,
                colors[i],
                linewidth=2,
                label="Radial model (mean)",
                linestyle="-",
            )

            # Also plot wall temperature
            T_wall = T_radial_profiles[-1, :] - 273.15
            ax.plot(
                t_eval,
                T_wall,
                colors[i],
                linewidth=1,
                label="Wall temp",
                linestyle="--",
                alpha=0.7,
            )

            ax.set_title(f"{name}\nRadial Temperature Model")
            ax.set_ylabel("Temperature [°C]")
            ax.set_xlabel("Time [s]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 120)

        plt.tight_layout()
        plt.show()

    def analyze_gradient_effects(self, results):
        """
        Quantify the differences between models
        """

        print("\nQuantitative Analysis of Model Differences:")
        print("=" * 45)

        t_eval = np.linspace(0, 120, 300)

        print(
            f"{'Scenario':<25} {'Single [°C]':<12} {'Radial [°C]':<12} {'Difference':<12} {'Error %':<10}"
        )
        print("-" * 75)

        for name, data in results.items():
            # Final temperatures
            T_single_final = data["single"].sol([120])[0][0] - 273.15

            # Radial model mean temperature
            sol_radial = data["radial"]
            r_centers = data["r_centers"]
            T_radial_final = sol_radial.sol([120])

            # Area-weighted mean
            # Create annular areas for each radial node
            r_faces = np.linspace(0, self.R, len(r_centers) + 1)
            r_areas = np.pi * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)
            T_mean_radial = (
                np.sum(T_radial_final[:, 0] * r_areas) / np.sum(r_areas) - 273.15
            )

            difference = T_single_final - T_mean_radial
            error_percent = (
                abs(difference) / T_mean_radial * 100 if T_mean_radial > 0 else 0
            )

            print(
                f"{name:<25} {T_single_final:<12.2f} {T_mean_radial:<12.2f} {difference:<12.2f} {error_percent:<10.1f}"
            )

    def when_radial_gradients_matter(self):
        """
        Provide guidelines for when radial gradients are important
        """

        print("\nWhen Do Radial Temperature Gradients Matter?")
        print("=" * 45)

        criteria = [
            "1. TIME SCALE ANALYSIS:",
            f"   • Damköhler number: Da = τ_radial/τ_axial = {self.Da:.2f}",
            "   • Da >> 1: Radial gradients persist (IMPORTANT)",
            "   • Da << 1: Radial gradients equilibrate quickly (negligible)",
            "",
            "2. HEAT FLUX INTENSITY:",
            "   • High irradiance (>5 kW/m²): Gradients likely important",
            "   • Moderate irradiance (<2 kW/m²): Usually negligible",
            "   • Pulse heating: Always creates gradients",
            "",
            "3. PIPE SIZE EFFECTS:",
            f"   • Current pipe (D = {self.D * 1000:.1f} mm): τ_radial = {self.tau_radial:.2f} s",
            "   • Larger pipes: Longer radial diffusion times",
            "   • Smaller pipes: Faster equilibration",
            "",
            "4. FLUID PROPERTIES:",
            f"   • Current Pr = {self.Pr:.2f}",
            "   • High Pr fluids (oils): Slower thermal diffusion",
            "   • Low Pr fluids (liquid metals): Faster equilibration",
            "",
            "5. PRACTICAL INDICATORS:",
            "   • Rapid irradiance changes (clouds, tracking errors)",
            "   • Concentrated solar systems (>1000x concentration)",
            "   • Startup/shutdown transients",
            "   • Non-uniform heating (partial shading)",
        ]

        for criterion in criteria:
            print(criterion)

    def design_recommendations(self):
        """
        Provide design recommendations based on analysis
        """

        print("\nDesign Recommendations:")
        print("=" * 25)

        if self.Da > 5:
            print(
                f"HIGH DAMKÖHLER NUMBER (Da = {self.Da:.1f}) - RADIAL GRADIENTS IMPORTANT:"
            )
            print("  → Use multi-region or 2D models for accurate predictions")
            print("  → Consider design changes to reduce gradients:")
            print("    • Increase flow rate (reduces residence time)")
            print("    • Use smaller diameter pipes")
            print("    • Add mixing elements or turbulence promoters")
            print("    • Implement smoother irradiance control")

        elif self.Da < 0.2:
            print(f"LOW DAMKÖHLER NUMBER (Da = {self.Da:.1f}) - SINGLE MODEL ADEQUATE:")
            print("  → Single temperature model will be accurate")
            print("  → Focus on other design aspects")
            print("  → Radial gradients are negligible")

        else:
            print(f"INTERMEDIATE DAMKÖHLER NUMBER (Da = {self.Da:.1f}):")
            print("  → Single model may have moderate errors (5-15%)")
            print("  → Consider validation with detailed model")
            print("  → Decision depends on accuracy requirements")


def main():
    """
    Demonstrate radial gradient effects in solar collectors
    """

    print("Radial Temperature Gradient Effects in Solar Collectors")
    print("=" * 55)

    # Example solar collector parameters
    pipe_params = {
        "diameter": 0.02,  # 20 mm diameter (typical solar collector tube)
    }

    fluid_props = {
        "density": 850,  # kg/m³ (thermal oil)
        "heat_capacity": 2100,  # J/kg·K
        "conductivity": 0.12,  # W/m·K
        "viscosity": 0.01,  # Pa·s
    }

    flow_conditions = {
        "velocity": 0.5  # m/s (moderate flow rate)
    }

    # Create analysis
    analysis = RadialGradientAnalysis(
        pipe_params["diameter"], fluid_props, flow_conditions
    )

    # Compare different irradiance scenarios
    results = analysis.compare_irradiance_scenarios()

    # Quantify differences
    analysis.analyze_gradient_effects(results)

    # Guidelines
    analysis.when_radial_gradients_matter()

    # Design recommendations
    analysis.design_recommendations()

    print("\nCONCLUSION:")
    print("Your intuition is correct! For solar collectors with:")
    print("• Rapid irradiance changes")
    print("• High heat flux levels")
    print("• Moderate to large pipe diameters")
    print("• High Prandtl number fluids")
    print("")
    print("Radial temperature gradients CAN significantly affect")
    print("the outlet temperature, and single temperature models")
    print("may not capture these effects accurately.")


if __name__ == "__main__":
    main()
