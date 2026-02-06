#!/usr/bin/env python
"""Generate comparison plots from multiple simulation results.

This script creates plots that overlay results from multiple simulations
for easy comparison.

Usage:
    python make_comparison_plots.py <experiment_name> --sim <sim1> <sim2> [sim3]

Example:
    python make_comparison_plots.py steps --sim steps_01 steps_02_steady_init

This will:
1. Load results from simulations/<experiment_name>/results/
2. Generate comparison plots with all simulations overlaid
3. Save plots to simulations/<experiment_name>/plots/
"""

import argparse
import fnmatch
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from solar_collector.config import PLOT_COLORS
from solar_collector.solar_collector_dae_pyo_two_temp import (
    ZERO_C,
    plot_time_series_comparison,
)


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
    with open(meta_file, "r", encoding="utf-8") as f:
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


def build_time_series_data(data: dict, colors: dict = None) -> list:
    """
    Build time series data structure for plotting.

    Parameters
    ----------
    data : dict
        Simulation data from load_simulation_results().
    colors : dict, optional
        Color scheme.

    Returns
    -------
    list
        List of data series dicts for plot_time_series functions.
    """
    if colors is None:
        colors = PLOT_COLORS

    T_inlet_C = data["T_inlet_K"] - ZERO_C
    T_p_inlet_C = data["T_p_C"][:, 0]

    # Create position-based colors for velocity (yellowish-green to dark green)
    import matplotlib.pyplot as plt

    cmap = plt.colormaps["YlGn"]
    v_inlet_color = cmap(0.3)  # Light yellowish-green for inlet (x=0)
    v_outlet_color = cmap(0.8)  # Dark green for outlet (x=L)

    return [
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


def find_simulations(results_dir: Path) -> list:
    """
    Find all simulation names in a results directory.

    Returns list of simulation names (without file extensions).
    """
    # Look for metadata files to identify simulations
    meta_files = list(results_dir.glob("*_metadata.yaml"))
    sim_names = [f.stem.replace("_metadata", "") for f in meta_files]
    return sorted(sim_names)


def make_comparison_plot(
    results_dir: Path,
    sim_names: list,
    title: str = "Simulation Comparison",
    colors: dict = None,
    figsize: tuple = (10, 10),
    min_yrange: float = 0.22,
) -> plt.Figure:
    """
    Create comparison plot with multiple simulations overlaid.

    Parameters
    ----------
    results_dir : Path
        Directory containing result files.
    sim_names : list
        List of simulation names to compare.
    title : str, default="Simulation Comparison"
        Plot title.
    colors : dict, optional
        Color scheme.
    figsize : tuple, default=(10, 10)
        Figure size.
    min_yrange : float, default=0.22
        Minimum y-axis range to prevent tiny axis scales.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    if colors is None:
        colors = PLOT_COLORS

    # Load data and build series for each simulation
    simulations = []
    for sim_name in sim_names:
        print(f"  Loading {sim_name}...")
        data = load_simulation_results(results_dir, sim_name)
        series = build_time_series_data(data, colors)
        simulations.append((data["t_vals"], series, sim_name))

    # Create comparison plot
    fig, _ = plot_time_series_comparison(
        simulations,
        title=title,
        colors=colors,
        figsize=figsize,
        min_yrange=min_yrange,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from multiple simulation results",
        epilog="""
Examples:
  python make_comparison_plots.py steps --sim steps_01 steps_02
  python make_comparison_plots.py steps --sim "steps_*"
  python make_comparison_plots.py steps --list
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
        help="Simulations to compare (2-3 required). "
        "Supports glob patterns (e.g., 'steps_*').",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available simulations and exit",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title (default: 'Comparison: sim1 vs sim2 ...')",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for saved images (default: 300)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output filename (default: auto-generated from sim names)",
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

    # Require --sim argument for comparison
    if not args.simulations:
        print(
            "Error: --sim argument required. Specify 2-3 simulations to compare."
        )
        print(f"Available simulations: {', '.join(all_sim_names)}")
        sys.exit(1)

    # Select simulations based on --sim argument
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

    # Validate number of simulations
    if len(sim_names) < 2:
        print("Error: Need at least 2 simulations to compare.")
        sys.exit(1)
    if len(sim_names) > 3:
        print(f"Error: Too many simulations ({len(sim_names)}). Maximum is 3.")
        print("Select specific simulations or use a more specific pattern.")
        sys.exit(1)

    # Generate title
    if args.title:
        title = args.title
    else:
        title = "Comparison: " + " vs ".join(sim_names)

    # Generate output filename
    if args.output:
        output_name = args.output
    else:
        output_name = "compare_" + "_".join(sim_names)

    print(f"Comparing {len(sim_names)} simulations: {', '.join(sim_names)}")
    print(f"Output will be saved to: {plots_dir / output_name}.png")
    print()

    # Create plots directory
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison plot
    print("Creating comparison plot...")
    fig = make_comparison_plot(
        results_dir,
        sim_names,
        title=title,
    )

    # Save plot
    output_path = plots_dir / f"{output_name}.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
