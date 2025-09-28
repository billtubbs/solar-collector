import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class SimpleMixingHeatModel:
    """
    Single mean temperature model for fluid mixing and heat transfer in pipes

    Uses effective transport coefficients to account for turbulent mixing
    and radial heat transfer without resolving radial profiles
    """

    def __init__(self, pipe_diameter, pipe_length, fluid_props, flow_conditions):
        self.D = pipe_diameter
        self.L = pipe_length
        self.R = self.D / 2

        # Fluid properties
        self.rho = fluid_props["density"]  # kg/m³
        self.cp = fluid_props["heat_capacity"]  # J/kg·K
        self.k = fluid_props["conductivity"]  # W/m·K
        self.mu = fluid_props["viscosity"]  # Pa·s
        self.alpha = self.k / (self.rho * self.cp)  # Thermal diffusivity

        # Flow conditions
        self.Re = flow_conditions["reynolds"]
        self.Pr = self.mu * self.cp / self.k

        print("Simple Mixing-Heat Model Initialized:")
        print(f"  Pipe: D = {self.D:.3f} m, L = {self.L:.1f} m")
        print(f"  Re = {self.Re:.0f}, Pr = {self.Pr:.2f}")

    def effective_axial_diffusivity(self):
        """
        Calculate effective axial diffusivity for longitudinal mixing

        Combines molecular and turbulent contributions:
        D_eff = α + D_turbulent
        """

        # Molecular thermal diffusivity
        D_molecular = self.alpha

        # Turbulent thermal diffusivity
        if self.Re > 4000:  # Turbulent flow
            # Taylor-Aris dispersion for heat transfer
            # D_turb ≈ 0.1 * U * R for turbulent pipe flow
            U_bulk = self.Re * self.mu / (self.rho * self.D)
            D_turbulent = 0.1 * U_bulk * self.R / self.Pr**0.5
        else:  # Laminar flow
            # Classical Taylor dispersion
            U_bulk = self.Re * self.mu / (self.rho * self.D)
            D_turbulent = U_bulk**2 * self.R**2 / (48 * self.alpha)

        D_eff = D_molecular + D_turbulent

        print(f"  Axial diffusivity: D_eff = {D_eff:.2e} m²/s")
        print(f"    Molecular: {D_molecular:.2e} m²/s")
        print(f"    Turbulent: {D_turbulent:.2e} m²/s")

        return D_eff

    def wall_heat_transfer_coefficient(self):
        """
        Calculate wall heat transfer coefficient using Nusselt correlations
        """

        if self.Re > 4000:  # Turbulent flow
            if self.Re < 5e6 and 0.5 < self.Pr < 2000:
                # Gnielinski correlation
                f = (0.79 * np.log(self.Re) - 1.64) ** (-2)
                Nu = (
                    (f / 8)
                    * (self.Re - 1000)
                    * self.Pr
                    / (1 + 12.7 * (f / 8) ** 0.5 * (self.Pr ** (2 / 3) - 1))
                )
            else:
                # Dittus-Boelter correlation
                Nu = 0.023 * self.Re**0.8 * self.Pr**0.4
        else:  # Laminar flow
            # Constant Nusselt number for constant wall temperature
            Nu = 3.66

        h = Nu * self.k / self.D

        print(f"  Heat transfer: Nu = {Nu:.1f}, h = {h:.0f} W/m²·K")

        return h, Nu

    def single_temperature_model(self, x_grid, t_span, boundary_conditions):
        """
        Solve single mean temperature T_f(x,t) model

        Governing equation:
        ∂T_f/∂t + u ∂T_f/∂x = D_eff ∂²T_f/∂x² + (4h/ρcp D)(T_wall - T_f)
        """

        # Get transport properties
        D_eff = self.effective_axial_diffusivity()
        h, Nu = self.wall_heat_transfer_coefficient()

        # Model parameters
        u = self.Re * self.mu / (self.rho * self.D)  # Bulk velocity
        heat_transfer_coeff = 4 * h / (self.rho * self.cp * self.D)

        nx = len(x_grid)
        dx = x_grid[1] - x_grid[0]

        def temperature_ode(t, T_f):
            """ODE system for temperature evolution"""
            dT_dt = np.zeros(nx)

            T_wall = boundary_conditions["wall_temp"](t)
            T_inlet = boundary_conditions["inlet_temp"](t)

            for i in range(nx):
                # Convective term
                if i == 0:  # Inlet boundary
                    convective = u * (T_inlet - T_f[i]) / dx
                else:
                    convective = -u * (T_f[i] - T_f[i - 1]) / dx

                # Diffusive term (central difference)
                if i == 0:  # Inlet boundary
                    diffusive = D_eff * (T_f[i + 1] - T_f[i]) / dx**2
                elif i == nx - 1:  # Outlet boundary
                    diffusive = D_eff * (T_f[i - 1] - T_f[i]) / dx**2
                else:
                    diffusive = D_eff * (T_f[i + 1] - 2 * T_f[i] + T_f[i - 1]) / dx**2

                # Wall heat transfer
                wall_heat = heat_transfer_coeff * (T_wall - T_f[i])

                dT_dt[i] = convective + diffusive + wall_heat

            return dT_dt

        # Initial conditions
        T_initial = np.full(nx, boundary_conditions["initial_temp"])

        # Solve ODE
        sol = solve_ivp(
            temperature_ode, t_span, T_initial, dense_output=True, rtol=1e-6
        )

        return sol, {"D_eff": D_eff, "h": h, "Nu": Nu, "u": u}

    def plot_single_model_results(
        self, x_grid, sol, params, title="Single Temperature Model"
    ):
        """Plot results from single temperature model"""

        t_eval = np.linspace(sol.t[0], sol.t[-1], 50)
        T_profiles = sol.sol(t_eval)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Temperature profiles at different times
        time_fractions = [0, 0.25, 0.5, 0.75, 1.0]
        for frac in time_fractions:
            idx = int(frac * (len(t_eval) - 1))
            ax1.plot(
                x_grid,
                T_profiles[:, idx] - 273.15,
                linewidth=2,
                label=f"t = {t_eval[idx]:.1f}s",
            )

        ax1.set_xlabel("Axial Position [m]")
        ax1.set_ylabel("Temperature [°C]")
        ax1.set_title("Temperature Profiles Along Pipe")
        ax1.legend()
        ax1.grid(True)

        # Outlet temperature vs time
        ax2.plot(t_eval, T_profiles[-1, :] - 273.15, "r-", linewidth=2)
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Outlet Temperature [°C]")
        ax2.set_title("Outlet Temperature Evolution")
        ax2.grid(True)

        # Temperature contour
        X, T = np.meshgrid(x_grid, t_eval)
        contour = ax3.contourf(X, T, T_profiles.T - 273.15, levels=20, cmap="hot")
        ax3.set_xlabel("Axial Position [m]")
        ax3.set_ylabel("Time [s]")
        ax3.set_title("Temperature Evolution (x-t)")
        plt.colorbar(contour, ax=ax3, label="Temperature [°C]")

        # Model parameters display
        param_text = (
            f"Model Parameters:\n"
            f"D_eff = {params['D_eff']:.2e} m²/s\n"
            f"h = {params['h']:.0f} W/m²·K\n"
            f"Nu = {params['Nu']:.1f}\n"
            f"u = {params['u']:.2f} m/s"
        )

        ax4.text(
            0.1,
            0.5,
            param_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgray"),
        )
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title("Model Parameters")
        ax4.axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class MultiRegionMixingModel:
    """
    Multi-region model dividing fluid into discrete radial zones

    Regions:
    1. Core region (turbulent mixing dominant)
    2. Buffer region (intermediate)
    3. Near-wall region (viscous effects)
    """

    def __init__(
        self, pipe_diameter, pipe_length, fluid_props, flow_conditions, n_regions=3
    ):
        self.D = pipe_diameter
        self.L = pipe_length
        self.R = self.D / 2
        self.n_regions = n_regions

        # Fluid properties (same as simple model)
        self.rho = fluid_props["density"]
        self.cp = fluid_props["heat_capacity"]
        self.k = fluid_props["conductivity"]
        self.mu = fluid_props["viscosity"]

        self.Re = flow_conditions["reynolds"]
        self.Pr = self.mu * self.cp / self.k

        # Define radial regions
        self.define_regions()

        print("Multi-Region Model Initialized:")
        print(f"  Regions: {self.n_regions}")
        print(f"  Region boundaries: {[f'{r:.3f}' for r in self.region_boundaries]}")

    def define_regions(self):
        """Define radial regions based on turbulent flow structure"""

        # Calculate friction velocity for region definition
        f = 0.316 * self.Re ** (-0.25) if self.Re < 1e5 else 0.184 * self.Re ** (-0.2)
        u_bulk = self.Re * self.mu / (self.rho * self.D)
        u_tau = u_bulk * np.sqrt(f / 8)
        nu = self.mu / self.rho

        # y⁺ boundaries for flow regions
        y_plus_viscous = 5  # Viscous sublayer
        y_plus_buffer = 30  # Buffer layer

        # Convert to physical distances
        y_viscous = y_plus_viscous * nu / u_tau
        y_buffer = y_plus_buffer * nu / u_tau

        if self.n_regions == 2:
            # Core + Near-wall
            self.region_boundaries = [0, self.R - y_buffer, self.R]
            self.region_names = ["Core", "Near-wall"]

        elif self.n_regions == 3:
            # Core + Buffer + Viscous
            self.region_boundaries = [0, self.R - y_buffer, self.R - y_viscous, self.R]
            self.region_names = ["Core", "Buffer", "Viscous"]

        else:
            # Equal volume regions
            self.region_boundaries = np.linspace(0, self.R, self.n_regions + 1)
            self.region_names = [f"Region_{i + 1}" for i in range(self.n_regions)]

        # Calculate region properties
        self.calculate_region_properties()

    def calculate_region_properties(self):
        """Calculate volume, velocity, and mixing properties for each region"""

        self.region_volumes = []
        self.region_velocities = []
        self.mixing_coefficients = []

        for i in range(self.n_regions):
            r_inner = self.region_boundaries[i]
            r_outer = self.region_boundaries[i + 1]

            # Volume (per unit length)
            volume = np.pi * (r_outer**2 - r_inner**2)
            self.region_volumes.append(volume)

            # Average velocity (using power law profile)
            n = 7  # Power law exponent
            if r_inner == 0:  # Core region includes centerline
                r_avg = r_outer / 2
            else:
                r_avg = (r_inner + r_outer) / 2

            u_bulk = self.Re * self.mu / (self.rho * self.D)
            U_max = u_bulk * (2 * n**2) / ((n + 1) * (2 * n + 1))
            u_region = U_max * (1 - r_avg / self.R) ** (1 / n)
            self.region_velocities.append(u_region)

            # Mixing coefficient (empirical)
            if i == 0:  # Core region - high mixing
                mix_coeff = 0.1 * u_bulk * self.R
            elif i == self.n_regions - 1:  # Wall region - low mixing
                mix_coeff = 0.01 * u_bulk * self.R
            else:  # Intermediate regions
                mix_coeff = 0.05 * u_bulk * self.R

            self.mixing_coefficients.append(mix_coeff)

    def multi_region_model(self, x_grid, t_span, boundary_conditions):
        """
        Solve multi-region temperature model

        Each region has its own temperature T_i(x,t)
        Includes inter-region mixing and wall heat transfer
        """

        nx = len(x_grid)
        dx = x_grid[1] - x_grid[0]

        def temperature_ode(t, T_all):
            """ODE system for multi-region temperatures"""

            # Reshape to [n_regions, nx] array
            T_regions = T_all.reshape(self.n_regions, nx)
            dT_dt = np.zeros((self.n_regions, nx))

            T_wall = boundary_conditions["wall_temp"](t)
            T_inlet = boundary_conditions["inlet_temp"](t)

            for region in range(self.n_regions):
                u = self.region_velocities[region]
                mix_coeff = self.mixing_coefficients[region]

                for i in range(nx):
                    # Axial convection
                    if i == 0:  # Inlet
                        convective = u * (T_inlet - T_regions[region, i]) / dx
                    else:
                        convective = (
                            -u * (T_regions[region, i] - T_regions[region, i - 1]) / dx
                        )

                    # Axial mixing
                    if i == 0:
                        mixing = (
                            mix_coeff
                            * (T_regions[region, i + 1] - T_regions[region, i])
                            / dx**2
                        )
                    elif i == nx - 1:
                        mixing = (
                            mix_coeff
                            * (T_regions[region, i - 1] - T_regions[region, i])
                            / dx**2
                        )
                    else:
                        mixing = (
                            mix_coeff
                            * (
                                T_regions[region, i + 1]
                                - 2 * T_regions[region, i]
                                + T_regions[region, i - 1]
                            )
                            / dx**2
                        )

                    # Inter-region heat exchange
                    inter_region_exchange = 0
                    if region > 0:  # Exchange with inner region
                        exchange_coeff = 1000  # W/m³·K (empirical)
                        inter_region_exchange += exchange_coeff * (
                            T_regions[region - 1, i] - T_regions[region, i]
                        )

                    if region < self.n_regions - 1:  # Exchange with outer region
                        exchange_coeff = 1000
                        inter_region_exchange += exchange_coeff * (
                            T_regions[region + 1, i] - T_regions[region, i]
                        )

                    # Wall heat transfer (only for outermost region)
                    wall_heat = 0
                    if region == self.n_regions - 1:  # Outermost region
                        h = 1000  # W/m²·K (simplified)
                        wall_area_per_volume = 4 / self.D  # 1/m
                        wall_heat = (
                            h * wall_area_per_volume * (T_wall - T_regions[region, i])
                        )

                    dT_dt[region, i] = (
                        convective
                        + mixing
                        + inter_region_exchange / self.rho / self.cp
                        + wall_heat / self.rho / self.cp
                    )

            return dT_dt.flatten()

        # Initial conditions
        T_initial = np.full(self.n_regions * nx, boundary_conditions["initial_temp"])

        # Solve ODE
        sol = solve_ivp(
            temperature_ode, t_span, T_initial, dense_output=True, rtol=1e-6
        )

        return sol

    def plot_multi_region_results(self, x_grid, sol, title="Multi-Region Model"):
        """Plot results from multi-region model"""

        t_eval = np.linspace(sol.t[0], sol.t[-1], 50)
        T_all = sol.sol(t_eval)

        # Reshape to [n_regions, nx, nt]
        nx = len(x_grid)
        T_regions = T_all.reshape(self.n_regions, nx, len(t_eval))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Temperature profiles for each region (final time)
        colors = ["blue", "green", "red", "orange", "purple"]
        for region in range(self.n_regions):
            ax1.plot(
                x_grid,
                T_regions[region, :, -1] - 273.15,
                color=colors[region],
                linewidth=2,
                label=f"{self.region_names[region]}",
            )

        ax1.set_xlabel("Axial Position [m]")
        ax1.set_ylabel("Temperature [°C]")
        ax1.set_title("Final Temperature Profiles by Region")
        ax1.legend()
        ax1.grid(True)

        # Mixed-mean temperature evolution
        # Volume-weighted average
        T_mixed = np.zeros(len(t_eval))
        for t_idx in range(len(t_eval)):
            weighted_sum = 0
            total_volume = 0
            for region in range(self.n_regions):
                weighted_sum += (
                    np.mean(T_regions[region, :, t_idx]) * self.region_volumes[region]
                )
                total_volume += self.region_volumes[region]
            T_mixed[t_idx] = weighted_sum / total_volume

        ax2.plot(t_eval, T_mixed - 273.15, "k-", linewidth=2, label="Mixed-mean")
        ax2.plot(
            t_eval,
            T_regions[-1, -1, :] - 273.15,
            "r--",
            linewidth=2,
            label="Outlet (outer region)",
        )
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Temperature [°C]")
        ax2.set_title("Temperature Evolution")
        ax2.legend()
        ax2.grid(True)

        # Radial temperature distribution at outlet (final time)
        r_centers = []
        T_radial = []
        for region in range(self.n_regions):
            r_inner = self.region_boundaries[region]
            r_outer = self.region_boundaries[region + 1]
            r_center = (r_inner + r_outer) / 2
            r_centers.append(r_center * 1000)  # Convert to mm
            T_radial.append(T_regions[region, -1, -1] - 273.15)

        ax3.plot(r_centers, T_radial, "bo-", linewidth=2, markersize=8)
        ax3.set_xlabel("Radial Position [mm]")
        ax3.set_ylabel("Temperature [°C]")
        ax3.set_title("Radial Temperature at Outlet")
        ax3.grid(True)

        # Region properties table
        region_data = []
        for i in range(self.n_regions):
            region_data.append(
                [
                    self.region_names[i],
                    f"{self.region_velocities[i]:.2f}",
                    f"{self.region_volumes[i]:.2e}",
                    f"{self.mixing_coefficients[i]:.2e}",
                ]
            )

        table_text = "Region Properties:\n"
        table_text += (
            f"{'Name':<10} {'Vel[m/s]':<8} {'Vol[m²]':<10} {'Mix[m²/s]':<10}\n"
        )
        table_text += "-" * 45 + "\n"
        for row in region_data:
            table_text += f"{row[0]:<10} {row[1]:<8} {row[2]:<10} {row[3]:<10}\n"

        ax4.text(
            0.1,
            0.5,
            table_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray"),
        )
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title("Model Properties")
        ax4.axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def compare_modeling_approaches():
    """
    Compare simple single-temperature vs multi-region approaches
    """

    print("Comparing Modeling Approaches")
    print("=" * 35)

    # Common parameters
    pipe_params = {
        "diameter": 0.1,  # 10 cm diameter
        "length": 5.0,  # 5 m length
    }

    fluid_props = {
        "density": 1000,  # kg/m³
        "heat_capacity": 4180,  # J/kg·K
        "conductivity": 0.6,  # W/m·K
        "viscosity": 1e-3,  # Pa·s
    }

    flow_conditions = {
        "reynolds": 30000  # Turbulent flow
    }

    # Boundary conditions
    def wall_temp(t):
        return 373.15  # 100°C constant wall

    def inlet_temp(t):
        return 293.15 + 10 * (1 - np.exp(-t / 10))  # Step change

    boundary_conditions = {
        "wall_temp": wall_temp,
        "inlet_temp": inlet_temp,
        "initial_temp": 293.15,  # 20°C initial
    }

    # Create models
    simple_model = SimpleMixingHeatModel(
        pipe_params["diameter"], pipe_params["length"], fluid_props, flow_conditions
    )

    multi_model = MultiRegionMixingModel(
        pipe_params["diameter"],
        pipe_params["length"],
        fluid_props,
        flow_conditions,
        n_regions=3,
    )

    # Simulation parameters
    x_grid = np.linspace(0, pipe_params["length"], 21)
    t_span = (0, 60)  # 60 seconds

    print("\nRunning simulations...")

    # Solve both models
    sol_simple, params_simple = simple_model.single_temperature_model(
        x_grid, t_span, boundary_conditions
    )

    sol_multi = multi_model.multi_region_model(x_grid, t_span, boundary_conditions)

    # Plot results
    simple_model.plot_single_model_results(
        x_grid, sol_simple, params_simple, "Single Temperature Model"
    )

    multi_model.plot_multi_region_results(x_grid, sol_multi, "Multi-Region Model")

    return simple_model, multi_model, sol_simple, sol_multi


def model_selection_guidelines():
    """
    Provide guidelines for choosing between modeling approaches
    """

    print("\nModel Selection Guidelines")
    print("=" * 30)

    comparison = {
        "Aspect": [
            "Computational Cost",
            "Implementation Complexity",
            "Radial Resolution",
            "Mixing Accuracy",
            "Wall Heat Transfer",
            "Parameter Tuning",
            "Physical Insight",
            "Engineering Design",
            "Research Applications",
        ],
        "Single Temperature": [
            "Low",
            "Simple",
            "None (bulk average)",
            "Good (via D_eff)",
            "Good (via h)",
            "Minimal",
            "Limited",
            "Excellent",
            "Limited",
        ],
        "Multi-Region": [
            "Medium-High",
            "Complex",
            "Good (discrete zones)",
            "Very Good",
            "Very Good",
            "Extensive",
            "Good",
            "Good",
            "Excellent",
        ],
    }

    print(f"{'Aspect':<25} {'Single Temp':<15} {'Multi-Region':<15}")
    print("-" * 55)

    for i, aspect in enumerate(comparison["Aspect"]):
        print(
            f"{aspect:<25} {comparison['Single Temperature'][i]:<15} {comparison['Multi-Region'][i]:<15}"
        )

    print("\nRecommendations:")
    print("  • Use SINGLE TEMPERATURE when:")
    print("    - Engineering design calculations")
    print("    - Process control applications")
    print("    - Computational efficiency is critical")
    print("    - Bulk properties are sufficient")
    print("")
    print("  • Use MULTI-REGION when:")
    print("    - Radial gradients are important")
    print("    - Research requiring detailed physics")
    print("    - Validation against CFD/experiments")
    print("    - Complex heat transfer phenomena")


def main():
    """
    Demonstrate both modeling approaches
    """

    print("Fluid Mixing and Heat Transfer Models")
    print("=" * 40)

    # Compare approaches
    simple_model, multi_model, sol_simple, sol_multi = compare_modeling_approaches()

    # Selection guidelines
    model_selection_guidelines()

    print("\nConclusion:")
    print("Both approaches are valid - choice depends on your specific needs!")
    print(
        "Single temperature model is usually sufficient for engineering applications."
    )


if __name__ == "__main__":
    main()
