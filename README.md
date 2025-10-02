# Solar Collector

Dynamic simulation of a parabolic mirror solar collector project using Python optimization libraries.

## Description

This project provides dynamic simulation capabilities for parabolic mirror solar collector systems. It uses advanced optimization libraries including CasADi and Pyomo with the IPOPT solver for efficient numerical computations.

## Installation

### Dependencies

Requires IPOPT.  See installation instructions [here](https://coin-or.github.io/Ipopt/INSTALL.html).

Or install the Anaconda distribution manually:

```bash
conda install -c conda-forge ipopt
```


### Create Python Environment

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### Install Python Dependencies

Install the project and its dependencies:

```bash
# Upgrade pip
pip install --upgrade pip

# Install project in editable mode
pip install -e .
```

## License

This project is licensed under the MIT License.

## Author

Bill Tubbs
