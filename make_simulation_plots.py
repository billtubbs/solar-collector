#!/usr/bin/env python
"""Generate plots from simulation results.

This script creates plots from saved simulation results.

Usage:
    python make_simulation_plots.py <experiment_name>

Example:
    python make_simulation_plots.py steps

This will:
1. Look for simulations/<experiment_name>/results/*.csv
2. Generate plots for each simulation
3. Save plots to simulations/<experiment_name>/plots/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from solar_collector.solar_collector_dae_pyo_two_temp import (
    plot_time_series,
    plot_temperature_field,
    plot_spatial_profiles,
    ZERO_C,
)
from solar_collector.config import PLOT_COLORS


def load_simulation_results(results_dir: Path, sim_name: str) -> dict:
    """
    Load simulation results from CSV files.

    Parameters
    ----------
    results_dir : Path
        Directory containing result files.
    sim_name : str
        Simulation name (file prefix).

    Returns
    -------
    dict
        Dictionary with loaded data arrays.
    """
    # Load time series data
    ts_file = results_dir / f"{sim_name}_timeseries.csv"
    ts_df = pd.read_csv(ts_file)

    # Load temperature fields
    T_f_file = results_dir / f"{sim_name}_T_f.csv"
    T_f_df = pd.read_csv(T_f_file, index_col=0)

    T_p_file = results_dir / f"{sim_name}_T_p.csv"
    T_p_df = pd.read_csv(T_p_file, index_col=0)

    # Load metadata
    meta_file = results_dir / f"{sim_name}_metadata.yaml"
    with open(meta_file, "r") as f:
        metadata = yaml.safe_load(f)

    # Extract arrays
    t_vals = ts_df["time_s"].values
    x_vals = np.array([float(col) for col in T_f_df.columns])

    return {
        "t_vals": t_vals,
        "x_vals": x_vals,
        "T_f_C": T_f_df.values,  # Already in Celsius
        "T_p_C": T_p_df.values,
        "velocity_inlet": ts_df["velocity_inlet_m_s"].values,
        "velocity_outlet": ts_df["velocity_outlet_m_s"].values,
        "irradiance": ts_df["irradiance_W_m2"].values,
        "T_inlet_K": ts_df["T_inlet_K"].values,
        "T_f_outlet_C": ts_df["T_f_outlet_C"].values,
        "T_p_outlet_C": ts_df["T_p_outlet_C"].values,
        "metadata": metadata,
    }


def make_time_series_plot(
    data: dict,
    sim_name: str,
    colors: dict = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Create time series plot from simulation data.

    Parameters
    ----------
    data : dict
        Simulation data from load_simulation_results().
    sim_name : str
        Simulation name for title.
    colors : dict, optional
        Color scheme.
    figsize : tuple, default=(10, 8)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    if colors is None:
        colors = PLOT_COLORS

    t_vals = data["t_vals"]
    T_inlet_C = data["T_inlet_K"] - ZERO_C

    # Inlet wall temperature (first spatial point)
    T_p_inlet_C = data["T_p_C"][:, 0]

    # Create position-based colors for velocity (yellowish-green to dark green)
    # Using 'YlGn' colormap: 0.3 = light yellow-green (inlet), 0.8 = dark green (outlet)
    cmap = plt.colormaps["YlGn"]
    v_inlet_color = cmap(0.3)  # Light yellowish-green for inlet (x=0)
    v_outlet_color = cmap(0.8)  # Dark green for outlet (x=L)

    time_series_data = [
        {
            "lines": [
                {
                    "y": data["velocity_inlet"],
                    "label": "Inlet (x=0)",
                    "color": v_inlet_color,
                },
                {
                    "y": data["velocity_outlet"],
                    "label": "Outlet (x=L)",
                    "color": v_outlet_color,
                },
            ],
            "ylabel": "Velocity [m/s]",
            "title": "Fluid Velocity",
        },
        {
            "y": data["irradiance"],
            "ylabel": "Irradiance [W/m²]",
            "title": "Solar Irradiance (DNI)",
            "color": colors.get("q_solar_conc"),
        },
        {
            "lines": [
                {
                    "y": T_inlet_C,
                    "label": "Oil",
                    "color": colors.get("T_f"),
                },
                {
                    "y": T_p_inlet_C,
                    "label": "Wall",
                    "color": colors.get("T_p"),
                },
            ],
            "ylabel": "Inlet Temp [°C]",
            "title": "Collector Inlet Temperatures",
        },
        {
            "lines": [
                {
                    "y": data["T_f_outlet_C"],
                    "label": "Oil",
                    "color": colors.get("T_f"),
                },
                {
                    "y": data["T_p_outlet_C"],
                    "label": "Wall",
                    "color": colors.get("T_p"),
                },
            ],
            "ylabel": "Outlet Temp [°C]",
            "title": "Collector Outlet Temperatures",
        },
    ]

    fig, _ = plot_time_series(
        t_vals,
        time_series_data,
        title=sim_name,
        colors=colors,
        figsize=figsize,
    )

    return fig


def make_temperature_field_plot(
    data: dict,
    variable: str,
    sim_name: str,
    temp_range: tuple = (0, 400),
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create temperature field contour plot.

    Parameters
    ----------
    data : dict
        Simulation data from load_simulation_results().
    variable : str
        'T_f' for fluid or 'T_p' for pipe wall.
    sim_name : str
        Simulation name for title.
    temp_range : tuple, default=(0, 400)
        Temperature range for colorbar [°C].
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    t_vals = data["t_vals"]
    x_vals = data["x_vals"]
    L = data["metadata"]["L"]

    if variable == "T_f":
        temp_vals = data["T_f_C"]
        title = f"{sim_name} - Oil Temperature Field"
    elif variable == "T_p":
        temp_vals = data["T_p_C"]
        title = f"{sim_name} - Pipe Wall Temperature Field"
    else:
        raise ValueError(f"Unknown variable: {variable}")

    fig, _, _ = plot_temperature_field(
        t_vals,
        x_vals,
        temp_vals,
        title=title,
        temp_range=temp_range,
        collector_length=L,
        figsize=figsize,
    )

    return fig


def find_simulations(results_dir: Path) -> list:
    """
    Find all simulation names in a results directory.

    Returns list of simulation names (without file extensions).
    """
    # Look for metadata files to identify simulations
    meta_files = list(results_dir.glob("*_metadata.yaml"))
    sim_names = [f.stem.replace("_metadata", "") for f in meta_files]
    return sorted(sim_names)


def make_initial_final_plot(
    data: dict,
    sim_name: str,
    colors: dict = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create plot showing initial and final temperature profiles.

    Parameters
    ----------
    data : dict
        Simulation data from load_simulation_results().
    sim_name : str
        Simulation name for title.
    colors : dict, optional
        Color scheme.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    if colors is None:
        colors = PLOT_COLORS

    x_vals = data["x_vals"]
    L = data["metadata"]["L"]

    # Limit x_vals to collector length L
    x_mask = x_vals <= L * 1.01
    x_plot = x_vals[x_mask]

    # Extract initial and final temperatures (already in Celsius)
    T_f_initial = data["T_f_C"][0, x_mask]
    T_p_initial = data["T_p_C"][0, x_mask]
    T_f_final = data["T_f_C"][-1, x_mask]
    T_p_final = data["T_p_C"][-1, x_mask]

    t_final = data["t_vals"][-1]

    # Build data series for plot_spatial_profiles
    profile_data = [
        {
            "lines": [
                {
                    "y": T_f_initial,
                    "label": "Oil (T_f)",
                    "color": colors.get("T_f"),
                },
                {
                    "y": T_p_initial,
                    "label": "Wall (T_p)",
                    "color": colors.get("T_p"),
                },
            ],
            "ylabel": "Temperature [°C]",
            "title": "Initial Temperatures (t = 0)",
        },
        {
            "lines": [
                {
                    "y": T_f_final,
                    "label": "Oil (T_f)",
                    "color": colors.get("T_f"),
                },
                {
                    "y": T_p_final,
                    "label": "Wall (T_p)",
                    "color": colors.get("T_p"),
                },
            ],
            "ylabel": "Temperature [°C]",
            "title": f"Final Temperatures (t = {t_final:.0f} s)",
        },
    ]

    fig, _ = plot_spatial_profiles(
        x_plot,
        profile_data,
        title=sim_name,
        colors=colors,
        figsize=figsize,
        sharey=True,
    )

    return fig


def make_all_plots(
    results_dir: Path,
    plots_dir: Path,
    sim_name: str,
    dpi: int = 300,
):
    """
    Generate all plots for a simulation and save to files.

    Parameters
    ----------
    results_dir : Path
        Directory containing result CSV files.
    plots_dir : Path
        Directory to save plot images.
    sim_name : str
        Simulation name.
    dpi : int, default=300
        Resolution for saved images.
    """
    print("  Loading data...")
    data = load_simulation_results(results_dir, sim_name)

    # Determine appropriate temperature range from data
    T_min = min(data["T_f_C"].min(), data["T_p_C"].min())
    T_max = max(data["T_f_C"].max(), data["T_p_C"].max())
    # Round to nice values
    T_min_range = max(0, np.floor(T_min / 50) * 50 - 50)
    T_max_range = np.ceil(T_max / 50) * 50 + 50
    temp_range = (T_min_range, T_max_range)

    # Create plots directory
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Time series plot
    print("  Creating time series plot...")
    fig1 = make_time_series_plot(data, sim_name)
    fig1.savefig(plots_dir / f"{sim_name}_timeseries.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig1)

    # Initial/final temperature profiles
    print("  Creating initial/final temperature plot...")
    fig2 = make_initial_final_plot(data, sim_name)
    fig2.savefig(
        plots_dir / f"{sim_name}_initial_final.png", dpi=dpi, bbox_inches="tight"
    )
    plt.close(fig2)

    # Fluid temperature field
    print("  Creating fluid temperature field plot...")
    fig3 = make_temperature_field_plot(
        data, "T_f", sim_name, temp_range=temp_range
    )
    fig3.savefig(plots_dir / f"{sim_name}_T_f_field.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig3)

    # Pipe wall temperature field
    print("  Creating pipe wall temperature field plot...")
    fig4 = make_temperature_field_plot(
        data, "T_p", sim_name, temp_range=temp_range
    )
    fig4.savefig(plots_dir / f"{sim_name}_T_p_field.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig4)

    print(f"  Saved 4 plots to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from simulation results",
        epilog="""
Examples:
  python make_simulation_plots.py steps                    # Plot all simulations
  python make_simulation_plots.py steps --sim steps_01    # Plot only steps_01
  python make_simulation_plots.py steps --sim steps_01 steps_02  # Plot multiple
  python make_simulation_plots.py steps --sim "steps_0*"  # Plot matching pattern
  python make_simulation_plots.py steps --list            # List available simulations
  python make_simulation_plots.py steps --dpi 150         # Lower resolution
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment_name",
        help="Name of experiment (directory in simulations/)",
    )
    parser.add_argument(
        "--sim",
        nargs="*",
        dest="simulations",
        metavar="NAME",
        help="Process specific simulation(s) by name. "
             "Supports glob patterns (e.g., 'steps_*'). "
             "If not specified, processes all simulations.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available simulations and exit",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for saved images (default: 300)",
    )

    args = parser.parse_args()

    # Find directories
    base_dir = Path(__file__).parent / "simulations" / args.experiment_name
    results_dir = base_dir / "results"
    plots_dir = base_dir / "plots"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Find all available simulations
    all_sim_names = find_simulations(results_dir)

    if not all_sim_names:
        print(f"Error: No simulation results found in {results_dir}")
        sys.exit(1)

    # Handle --list option
    if args.list:
        print(f"Available simulations in '{args.experiment_name}':")
        for sim_name in all_sim_names:
            print(f"  {sim_name}")
        sys.exit(0)

    # Select simulations based on --sim argument
    if args.simulations:
        import fnmatch
        sim_names = []
        for pattern in args.simulations:
            # Check if pattern contains glob characters
            if "*" in pattern or "?" in pattern or "[" in pattern:
                # Use glob pattern matching
                matched = [s for s in all_sim_names if fnmatch.fnmatch(s, pattern)]
                if not matched:
                    print(f"Warning: No simulations match pattern '{pattern}'")
                sim_names.extend(matched)
            else:
                # Exact name match
                if pattern in all_sim_names:
                    sim_names.append(pattern)
                else:
                    print(f"Error: Simulation not found: {pattern}")
                    sys.exit(1)
        # Remove duplicates and sort
        sim_names = sorted(set(sim_names))
    else:
        sim_names = all_sim_names

    if not sim_names:
        print(f"Error: No simulation results found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(sim_names)} simulation(s)")
    print(f"Plots will be saved to: {plots_dir}")
    print()

    # Generate plots for each simulation
    for sim_name in sim_names:
        print(f"Processing: {sim_name}")
        try:
            make_all_plots(results_dir, plots_dir, sim_name, dpi=args.dpi)
        except Exception as e:
            print(f"  Error: {e}")
            raise

    print()
    print("All plots generated!")


if __name__ == "__main__":
    main()
