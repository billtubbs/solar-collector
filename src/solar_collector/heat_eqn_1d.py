import casadi as cas
import numpy as np
import matplotlib.pyplot as plt


# Define parameters
N = 50  # Number of spatial grid points
alpha = 1.0  # Thermal diffusivity
dx = 1.0 / (N - 1)  # Spatial step size

# Define symbolic variables for the state (temperature at each grid point)
u = cas.SX.sym("u", N)

# Define the right-hand side of the ODE system
dudt = []
for i in range(N):
    if i == 0:  # Left boundary condition (e.g., Dirichlet)
        dudt.append(0.0)  # Assume u[0] is fixed
    elif i == N - 1:  # Right boundary condition (e.g., Dirichlet)
        dudt.append(0.0)  # Assume u[N-1] is fixed
    else:  # Interior points
        dudt.append(alpha * (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx**2))

# Create a CasADi function for the ODE system
f = cas.Function("f", [u], [cas.vertcat(*dudt)])

# Set up the integrator
ode = {"x": u, "ode": f(u)}
opts = {"tf": 1.0}
F = cas.integrator("F", "cvodes", ode, opts)

# Initial condition - using numpy array for consistency
u0 = np.array([np.sin(np.pi * i * dx) for i in range(N)])

print(f"\nInitial condition shape: {u0.shape}")
print(f"Initial condition range: [{u0.min():.4f}, {u0.max():.4f}]")

# Integrate the system
print("\nIntegrating...")
sol = F(x0=u0)
u_final = sol["xf"]

print("Integration completed!")
print(f"Final solution shape: {np.array(u_final).shape}")
print(
    f"Final solution range: "
    f"[{float(cas.mmin(u_final)):.6f}, "
    f"{float(cas.mmax(u_final)):.6f}]"
)

# Convert to numpy for plotting
u0_np = np.array(u0)
u_final_np = np.array(u_final).flatten()
x_grid = np.linspace(0, 1, N)

# Plot results
plt.figure(figsize=(12, 8))

# Plot 1: Initial vs Final
plt.subplot(2, 2, 1)
plt.plot(x_grid, u0_np, "b-", linewidth=2, label="Initial (t=0)")
plt.plot(x_grid, u_final_np, "r-", linewidth=2, label="Final (t=1)")
plt.xlabel("Position x")
plt.ylabel("Temperature u")
plt.title("Heat Equation Solution: Initial vs Final")
plt.legend()
plt.grid(True)

# Plot 2: Log scale to see decay
plt.subplot(2, 2, 2)
plt.semilogy(x_grid, np.abs(u0_np) + 1e-12, "b-", linewidth=2, label="Initial")
plt.semilogy(x_grid, np.abs(u_final_np) + 1e-12, "r-", linewidth=2, label="Final")
plt.xlabel("Position x")
plt.ylabel("|Temperature| (log scale)")
plt.title("Temperature Decay (Log Scale)")
plt.legend()
plt.grid(True)

# Plot 3: Energy decay over time (multiple time steps)
time_points = [0.0, 0.1, 0.2, 0.5, 1.0]
plt.subplot(2, 2, 3)

for t in time_points:
    if t == 0:
        u_t = u0_np
    else:
        # Create integrator for time t
        opts_t = {"tf": t}
        F_t = cas.integrator("F_t", "cvodes", ode, opts_t)
        sol_t = F_t(x0=u0)
        u_t = np.array(sol_t["xf"]).flatten()

    plt.plot(x_grid, u_t, linewidth=2, label=f"t={t}")

plt.xlabel("Position x")
plt.ylabel("Temperature u")
plt.title("Temperature Evolution Over Time")
plt.legend()
plt.grid(True)

# Plot 4: Total energy vs time
plt.subplot(2, 2, 4)
energy_times = []
total_energies = []

for t in np.linspace(0, 1, 21):
    if t == 0:
        u_t = u0_np
    else:
        # Use new integrator syntax
        F_t = cas.integrator("F_t", "cvodes", ode, 0.0, t, {})
        sol_t = F_t(x0=u0)
        u_t = np.array(sol_t["xf"]).flatten()

    # Calculate total energy (integral of u^2)
    energy = np.trapezoid(u_t**2, x_grid)
    energy_times.append(t)
    total_energies.append(energy)

plt.plot(energy_times, total_energies, "g-", linewidth=2, marker="o")
plt.xlabel("Time t")
plt.ylabel("Total Energy ∫u² dx")
plt.title("Energy Decay Over Time")
plt.grid(True)

plt.tight_layout()
plt.show()
