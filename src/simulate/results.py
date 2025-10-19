"""Simulation results storage and analysis.

This module provides the SimulationResult class for storing and
analyzing simulation data.
"""

from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Container for simulation results.

    Stores time series data from simulation including states, outputs,
    and inputs as pandas DataFrames with time as the index.

    Parameters
    ----------
    time : ndarray
        Time points, shape (n_steps,)
    states : DataFrame, optional
        State trajectories with time index, shape (n_steps, n_states)
    outputs : DataFrame, optional
        Output trajectories with time index, shape (n_steps, n_outputs)
    inputs : DataFrame, optional
        Input trajectories with time index, shape (n_steps, n_inputs)
    config : SimulationConfig, optional
        Configuration used for this simulation

    Examples
    --------
    >>> result = SimulationResult(
    ...     time=np.linspace(0, 10, 100),
    ...     states=pd.DataFrame({'x1': ..., 'x2': ...}),
    ...     outputs=pd.DataFrame({'y1': ...})
    ... )
    >>> result.plot()
    >>> df = result.to_dataframe()
    """

    time: np.ndarray
    states: Optional[pd.DataFrame] = None
    outputs: Optional[pd.DataFrame] = None
    inputs: Optional[pd.DataFrame] = None
    config: Optional[Any] = None

    def __post_init__(self):
        """Validate result dimensions and ensure time indices are set."""
        n_steps = len(self.time)

        # Validate and ensure time index for states
        if self.states is not None:
            if len(self.states) != n_steps:
                raise ValueError(
                    f"States length {len(self.states)} != time length {n_steps}"
                )
            if not isinstance(self.states.index, pd.Index) or not np.array_equal(
                self.states.index.values, self.time
            ):
                self.states.index = pd.Index(self.time, name="time")

        # Validate and ensure time index for outputs
        if self.outputs is not None:
            if len(self.outputs) != n_steps:
                raise ValueError(
                    f"Outputs length {len(self.outputs)} != time length {n_steps}"
                )
            if not isinstance(self.outputs.index, pd.Index) or not np.array_equal(
                self.outputs.index.values, self.time
            ):
                self.outputs.index = pd.Index(self.time, name="time")

        # Validate and ensure time index for inputs
        if self.inputs is not None:
            if len(self.inputs) != n_steps:
                raise ValueError(
                    f"Inputs length {len(self.inputs)} != time length {n_steps}"
                )
            if not isinstance(self.inputs.index, pd.Index) or not np.array_equal(
                self.inputs.index.values, self.time
            ):
                self.inputs.index = pd.Index(self.time, name="time")

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.time)

    @property
    def n_states(self) -> Optional[int]:
        """Number of states (None if states not saved)."""
        if self.states is None:
            return None
        return len(self.states.columns)

    @property
    def n_outputs(self) -> Optional[int]:
        """Number of outputs (None if outputs not saved)."""
        if self.outputs is None:
            return None
        return len(self.outputs.columns)

    @property
    def n_inputs(self) -> Optional[int]:
        """Number of inputs (None if inputs not saved)."""
        if self.inputs is None:
            return None
        return len(self.inputs.columns)

    @property
    def dt(self) -> float:
        """Average time step size."""
        return np.mean(np.diff(self.time))

    def plot(self, figsize=(10, 8), **kwargs):
        """Plot simulation results.

        Creates a multi-panel figure with states, outputs, and inputs.
        Uses the DataFrame column names as labels.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height), by default (10, 8)
        **kwargs
            Additional arguments passed to plt.plot()

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : ndarray
            Array of axes objects

        Examples
        --------
        >>> fig, axes = result.plot()
        >>> plt.show()
        """
        # Determine number of subplots needed
        n_plots = sum(
            [
                self.states is not None,
                self.outputs is not None,
                self.inputs is not None,
            ]
        )

        if n_plots == 0:
            raise ValueError(
                "No data to plot (states, outputs, inputs all None)"
            )

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        plot_idx = 0

        # Plot states
        if self.states is not None:
            ax = axes[plot_idx]
            for col in self.states.columns:
                ax.plot(self.time, self.states[col], label=col, **kwargs)
            ax.set_ylabel("States")
            if len(self.states.columns) > 1:
                ax.legend()
            ax.set_title("State Trajectories")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Plot outputs
        if self.outputs is not None:
            ax = axes[plot_idx]
            for col in self.outputs.columns:
                ax.plot(self.time, self.outputs[col], label=col, **kwargs)
            ax.set_ylabel("Outputs")
            if len(self.outputs.columns) > 1:
                ax.legend()
            ax.set_title("Output Trajectories")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Plot inputs
        if self.inputs is not None:
            ax = axes[plot_idx]
            for col in self.inputs.columns:
                ax.plot(self.time, self.inputs[col], label=col, **kwargs)
            ax.set_ylabel("Inputs")
            ax.set_xlabel("Time")
            ax.set_title("Input Signals")
            if len(self.inputs.columns) > 1:
                ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Set x-label on bottom plot
        if plot_idx > 0:
            axes[plot_idx - 1].set_xlabel("Time")

        plt.tight_layout()
        return fig, axes

    def to_dataframe(self):
        """Convert results to a single pandas DataFrame.

        Concatenates inputs, states, and outputs into one DataFrame
        with time as a column (not index), in the order: time, inputs, states, outputs.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with columns: time, inputs, states, outputs

        Examples
        --------
        >>> df = result.to_dataframe()
        >>> df.head()
        >>> df.plot(x='time', y=['x1', 'x2'])
        """
        dfs = []

        # Add inputs first
        if self.inputs is not None:
            dfs.append(self.inputs)

        # Then states
        if self.states is not None:
            dfs.append(self.states)

        # Then outputs
        if self.outputs is not None:
            dfs.append(self.outputs)

        if not dfs:
            # Return DataFrame with just time column
            return pd.DataFrame({"time": self.time})

        # Concatenate all DataFrames
        result_df = pd.concat(dfs, axis=1)

        # Reset index to make time a column instead of index
        result_df = result_df.reset_index()

        return result_df

    def save(self, filename: str):
        """Save results to file.

        Supports .npz (NumPy), .csv (via pandas), and .mat (MATLAB) formats.

        Parameters
        ----------
        filename : str
            Output filename with extension

        Examples
        --------
        >>> result.save('simulation_results.npz')
        >>> result.save('simulation_results.csv')
        """
        import os

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".npz":
            # Save as NumPy compressed archive
            save_dict = {"time": self.time}
            if self.states is not None:
                save_dict["states"] = self.states.to_numpy()
                save_dict["state_columns"] = self.states.columns.tolist()
            if self.outputs is not None:
                save_dict["outputs"] = self.outputs.to_numpy()
                save_dict["output_columns"] = self.outputs.columns.tolist()
            if self.inputs is not None:
                save_dict["inputs"] = self.inputs.to_numpy()
                save_dict["input_columns"] = self.inputs.columns.tolist()
            np.savez_compressed(filename, **save_dict)

        elif ext == ".csv":
            # Save as CSV via pandas
            df = self.to_dataframe()
            df.to_csv(filename, index=False)

        elif ext == ".mat":
            # Save as MATLAB file
            from scipy.io import savemat

            save_dict = {"time": self.time}
            if self.states is not None:
                save_dict["states"] = self.states.to_numpy()
            if self.outputs is not None:
                save_dict["outputs"] = self.outputs.to_numpy()
            if self.inputs is not None:
                save_dict["inputs"] = self.inputs.to_numpy()
            savemat(filename, save_dict)

        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Use .npz, .csv, or .mat"
            )

    @classmethod
    def load(cls, filename: str) -> "SimulationResult":
        """Load results from file.

        Parameters
        ----------
        filename : str
            Input filename (.npz or .mat format)

        Returns
        -------
        result : SimulationResult
            Loaded simulation results

        Examples
        --------
        >>> result = SimulationResult.load('simulation_results.npz')
        """
        import os

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".npz":
            data = np.load(filename, allow_pickle=True)
            time = data["time"]

            states = None
            if "states" in data:
                state_cols = (
                    data["state_columns"].tolist()
                    if "state_columns" in data
                    else [f"x{i+1}" for i in range(data["states"].shape[1])]
                )
                states = pd.DataFrame(
                    data["states"], index=time, columns=state_cols
                )
                states.index.name = "time"

            outputs = None
            if "outputs" in data:
                output_cols = (
                    data["output_columns"].tolist()
                    if "output_columns" in data
                    else [f"y{i+1}" for i in range(data["outputs"].shape[1])]
                )
                outputs = pd.DataFrame(
                    data["outputs"], index=time, columns=output_cols
                )
                outputs.index.name = "time"

            inputs = None
            if "inputs" in data:
                input_cols = (
                    data["input_columns"].tolist()
                    if "input_columns" in data
                    else [f"u{i+1}" for i in range(data["inputs"].shape[1])]
                )
                inputs = pd.DataFrame(
                    data["inputs"], index=time, columns=input_cols
                )
                inputs.index.name = "time"

            return cls(
                time=time, states=states, outputs=outputs, inputs=inputs
            )

        elif ext == ".mat":
            from scipy.io import loadmat

            data = loadmat(filename)
            time = data["time"].flatten()

            states = None
            if "states" in data:
                n_states = data["states"].shape[1]
                states = pd.DataFrame(
                    data["states"],
                    index=time,
                    columns=[f"x{i+1}" for i in range(n_states)],
                )
                states.index.name = "time"

            outputs = None
            if "outputs" in data:
                n_outputs = data["outputs"].shape[1]
                outputs = pd.DataFrame(
                    data["outputs"],
                    index=time,
                    columns=[f"y{i+1}" for i in range(n_outputs)],
                )
                outputs.index.name = "time"

            inputs = None
            if "inputs" in data:
                n_inputs = data["inputs"].shape[1]
                inputs = pd.DataFrame(
                    data["inputs"],
                    index=time,
                    columns=[f"u{i+1}" for i in range(n_inputs)],
                )
                inputs.index.name = "time"

            return cls(
                time=time, states=states, outputs=outputs, inputs=inputs
            )

        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Use .npz or .mat"
            )

    def __repr__(self):
        parts = [f"SimulationResult(n_steps={self.n_steps}"]
        if self.n_states is not None:
            parts.append(f"n_states={self.n_states}")
        if self.n_outputs is not None:
            parts.append(f"n_outputs={self.n_outputs}")
        if self.n_inputs is not None:
            parts.append(f"n_inputs={self.n_inputs}")
        return ", ".join(parts) + ")"
