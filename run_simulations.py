#!/usr/bin/env python
"""Run simulations from YAML specification files.

This script runs solar collector simulations defined by YAML spec files.

Usage:
    python run_simulations.py <experiment_name>

Example:
    python run_simulations.py steps

This will:
1. Look for simulations/<experiment_name>/sim_specs/*.yaml
2. Run each simulation defined in the YAML files
3. Save results to simulations/<experiment_name>/results/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from simulate.inputs import (
    ConstantInput,
    InterpolatedInput,
    RampInput,
    SinusoidalInput,
    StepInput,
)
from solar_collector.fluid_properties import SYLTHERM800
from solar_collector.solar_collector_dae_pyo_two_temp import (
    ZERO_C,
    create_collector_model,
    extract_model_data,
    run_simulation,
)

# Registry of available input classes
INPUT_CLASSES = {
    "ConstantInput": ConstantInput,
    "StepInput": StepInput,
    "RampInput": RampInput,
    "InterpolatedInput": InterpolatedInput,
    "SinusoidalInput": SinusoidalInput,
}

# Mapping from YAML input names to model function names
INPUT_NAME_MAP = {
    "mass_flow_rate": "mass_flow_rate_func",
    "T_inlet": "T_inlet_func",
    "fluid_inlet_temperature": "T_inlet_func",
    "irradiance": "irradiance_func",
    "solar_irradiance": "irradiance_func",
}


def parse_input_spec(input_spec: dict) -> callable:
    """
    Parse an input specification from YAML and return a callable.

    Parameters
    ----------
    input_spec : dict
        Input specification from YAML, e.g.:
        {'ConstantInput': {'value': 0.5}}
        or
        {'StepInput': {'initial_value': 0, 'steps': [...]}}

    Returns
    -------
    callable
        An input function that takes time t and returns a value.
    """
    # Get the input class name (should be the only key)
    class_name = list(input_spec.keys())[0]
    params = input_spec[class_name]

    if class_name not in INPUT_CLASSES:
        raise ValueError(f"Unknown input class: {class_name}")

    input_class = INPUT_CLASSES[class_name]

    # Handle special case for StepInput with steps format
    if class_name == "StepInput" and "steps" in params:
        # Convert steps format to times/values arrays
        steps = params["steps"]
        times = [step["time"] for step in steps]
        values = [step["value"] for step in steps]

        # Add initial value if provided
        initial_value = params.get("initial_value", 0.0)
        values = [initial_value] + values

        return StepInput(times=times, values=values)

    # Standard case: pass params directly to class
    return input_class(**params)


def load_sim_spec(yaml_path: Path) -> dict:
    """Load and validate a simulation specification from YAML."""
    with open(yaml_path, "r") as f:
        spec = yaml.safe_load(f)

    # Basic validation
    required_sections = [
        "system",
        "simulation",
        "initial_conditions",
        "inputs",
    ]

    for section in required_sections:
        if section not in spec:
            raise ValueError(f"Missing required section: {section}")

    return spec


def create_input_functions(inputs_spec: dict) -> dict:
    """
    Create input functions from YAML specification.

    Returns dict with keys matching model parameter names.
    """
    input_funcs = {}

    for yaml_name, input_spec in inputs_spec.items():
        # Map YAML name to model parameter name
        model_name = INPUT_NAME_MAP.get(yaml_name, yaml_name + "_func")

        # Parse the input specification
        input_func = parse_input_spec(input_spec)

        input_funcs[model_name] = input_func

    return input_funcs


def run_single_simulation(spec: dict, fluid_props) -> dict:
    """
    Run a single simulation from a specification.

    Returns dict with model, results, and extracted data.
    """
    # Extract parameters
    system_spec = spec.get("system", {})
    model_spec = system_spec.get("model", {})
    sim_spec = spec.get("simulation", {})
    ic_spec = spec.get("initial_conditions", {})
    inputs_spec = spec.get("inputs", {})

    # Get model creation parameters
    create_params = model_spec.get(
        "create_params", model_spec.get("params", {})
    )

    # Handle t_final which might be in different places
    t_final = create_params.pop("t_final", sim_spec.pop("t_final", 300.0))

    # Create input functions
    input_funcs = create_input_functions(inputs_spec)

    # Get initial mass flow rate for h_int calculation when using constant
    # properties
    mass_flow_rate_func = input_funcs.get("mass_flow_rate_func")
    initial_mass_flow_rate = (
        mass_flow_rate_func(0.0) if mass_flow_rate_func else None
    )

    # Create the model
    print("Creating model...")
    model = create_collector_model(
        fluid_props,
        t_final=t_final,
        initial_mass_flow_rate=initial_mass_flow_rate,
        **create_params,
    )

    # Get simulation parameters
    n_x = sim_spec.get("n_x", 110)
    n_t = sim_spec.get("n_t", 50)
    solver_spec = sim_spec.get("solver", {})
    max_iter = solver_spec.get("max_iter", 1000)
    tol = solver_spec.get("tol", 1e-6)
    print_level = solver_spec.get("print_level", 0)
    tee = solver_spec.get("tee", False)

    # Get initial conditions
    # Check if initial_steady_state is requested
    initial_steady_state = ic_spec.get("initial_steady_state", False)

    if initial_steady_state:
        # Use steady-state initialization - T_f_initial and T_p_initial
        # will be computed from input functions at t=0
        T_f_initial = None
        T_p_initial = None
    else:
        T_f_initial = ic_spec["T_f_initial"]
        T_p_initial = ic_spec["T_p_initial"]

    # Run simulation
    print("Running simulation...")
    results = run_simulation(
        model,
        mass_flow_rate_func=input_funcs.get("mass_flow_rate_func"),
        irradiance_func=input_funcs.get("irradiance_func"),
        T_inlet_func=input_funcs.get("T_inlet_func"),
        T_f_initial=T_f_initial,
        T_p_initial=T_p_initial,
        initial_steady_state=initial_steady_state,
        n_x=n_x,
        n_t=n_t,
        max_iter=max_iter,
        tol=tol,
        print_level=print_level,
        tee=tee,
    )

    print(f"Solver status: {results.solver.termination_condition}")

    # Extract data
    data = extract_model_data(model)

    return {
        "model": model,
        "results": results,
        "data": data,
        "spec": spec,
    }


def save_results(sim_result: dict, output_dir: Path, sim_name: str):
    """Save simulation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = sim_result["data"]

    # Save temperature fields as CSV
    t_vals = data["t_vals"]
    x_vals = data["x_vals"]

    # Fluid temperature
    T_f_df = pd.DataFrame(
        data["T_f"] - ZERO_C,
        index=pd.Index(t_vals, name="time_s"),
        columns=pd.Index(x_vals, name="position_m"),
    )
    T_f_df.to_csv(output_dir / f"{sim_name}_T_f.csv")

    # Pipe wall temperature
    T_p_df = pd.DataFrame(
        data["T_p"] - ZERO_C,
        index=pd.Index(t_vals, name="time_s"),
        columns=pd.Index(x_vals, name="position_m"),
    )
    T_p_df.to_csv(output_dir / f"{sim_name}_T_p.csv")

    # Time series data
    # Use outlet_idx to get temperatures at collector outlet (x=L), not
    # extended domain
    outlet_idx = data["outlet_idx"]
    ts_data = {
        "time_s": t_vals,
        "m_dot_kg_s": data["m_dot"],
        "velocity_inlet_m_s": data["v"],
        "velocity_outlet_m_s": data["v_outlet"],
        "irradiance_W_m2": data["I"],
        "T_inlet_K": data["T_inlet"],
        "T_f_outlet_C": data["T_f"][:, outlet_idx] - ZERO_C,
        "T_p_outlet_C": data["T_p"][:, outlet_idx] - ZERO_C,
    }

    # Add temperature-dependent properties at outlet if present
    if "rho_f_outlet" in data:
        ts_data["rho_f_outlet_kg_m3"] = data["rho_f_outlet"]
    if "eta_f_outlet" in data:
        ts_data["eta_f_outlet_Pa_s"] = data["eta_f_outlet"]
    if "k_f_outlet" in data:
        ts_data["k_f_outlet_W_mK"] = data["k_f_outlet"]
    if "cp_f_outlet" in data:
        ts_data["cp_f_outlet_J_kgK"] = data["cp_f_outlet"]
    if "h_int_outlet" in data:
        ts_data["h_int_outlet_W_m2K"] = data["h_int_outlet"]

    ts_df = pd.DataFrame(ts_data)
    ts_df.to_csv(output_dir / f"{sim_name}_timeseries.csv", index=False)

    # Save metadata
    metadata = {
        "simulation_name": sim_name,
        "timestamp": datetime.now().isoformat(),
        "solver_status": str(
            sim_result["results"].solver.termination_condition
        ),
        "n_x": len(x_vals),
        "n_t": len(t_vals),
        "t_final": float(t_vals[-1]),
        "L": data["L"],
    }
    with open(output_dir / f"{sim_name}_metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run solar collector simulations from YAML specs",
        epilog="""
Examples:
  python run_simulations.py steps                    # Run all simulations in steps/
  python run_simulations.py steps --sim steps_01    # Run only steps_01
  python run_simulations.py steps --sim steps_01 steps_02  # Run multiple
  python run_simulations.py steps --sim "steps_0*"  # Run matching pattern
  python run_simulations.py steps --list            # List available simulations
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment_name",
        help="Name of experiment (directory in simulations/)",
    )
    parser.add_argument(
        "--sim",
        "--spec",
        nargs="*",
        dest="simulations",
        metavar="NAME",
        help="Run specific simulation(s) by name (without .yaml). "
        "Supports glob patterns (e.g., 'steps_*'). "
        "If not specified, runs all simulations.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available simulations and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse specs but don't run simulations",
    )

    args = parser.parse_args()

    # Find experiment directory
    base_dir = Path(__file__).parent / "simulations" / args.experiment_name
    spec_dir = base_dir / "sim_specs"
    results_dir = base_dir / "results"

    if not spec_dir.exists():
        print(f"Error: Spec directory not found: {spec_dir}")
        sys.exit(1)

    # Find all available YAML files
    all_yaml_files = sorted(spec_dir.glob("*.yaml"))

    if not all_yaml_files:
        print(f"Error: No YAML files found in {spec_dir}")
        sys.exit(1)

    # Handle --list option
    if args.list:
        print(f"Available simulations in '{args.experiment_name}':")
        for yaml_file in all_yaml_files:
            print(f"  {yaml_file.stem}")
        sys.exit(0)

    # Select YAML files based on --sim argument
    if args.simulations:
        yaml_files = []
        for pattern in args.simulations:
            # Check if pattern contains glob characters
            if "*" in pattern or "?" in pattern or "[" in pattern:
                # Use glob pattern matching
                matched = list(spec_dir.glob(f"{pattern}.yaml"))
                if not matched:
                    print(f"Warning: No files match pattern '{pattern}'")
                yaml_files.extend(matched)
            else:
                # Exact name match
                yaml_path = spec_dir / f"{pattern}.yaml"
                if yaml_path.exists():
                    yaml_files.append(yaml_path)
                else:
                    print(f"Error: Spec file not found: {yaml_path}")
                    sys.exit(1)
        # Remove duplicates and sort
        yaml_files = sorted(set(yaml_files))
    else:
        yaml_files = all_yaml_files

    if not yaml_files:
        print(f"Error: No YAML files found in {spec_dir}")
        sys.exit(1)

    print(f"Found {len(yaml_files)} simulation spec(s)")
    print(f"Results will be saved to: {results_dir}")
    print()

    # Create fluid properties (shared across all simulations)
    fluid_props = SYLTHERM800()

    # Run each simulation
    for yaml_file in yaml_files:
        sim_name = yaml_file.stem
        print(f"{'=' * 60}")
        print(f"Processing: {sim_name}")
        print(f"{'=' * 60}")

        # Load spec
        spec = load_sim_spec(yaml_file)

        if args.dry_run:
            print("Spec loaded successfully (dry run)")
            print(f"  System: {spec.get('system', {}).get('name', 'unnamed')}")
            print(f"  Inputs: {list(spec.get('inputs', {}).keys())}")
            continue

        # Run simulation
        try:
            result = run_single_simulation(spec, fluid_props)

            # Save results
            save_results(result, results_dir, sim_name)

        except Exception as e:
            print(f"Error running simulation {sim_name}: {e}")
            raise

        print()

    print("All simulations completed!")


if __name__ == "__main__":
    main()
