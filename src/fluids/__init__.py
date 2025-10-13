"""
Fluids Module

A comprehensive module for modeling fluid properties and heat transfer fluids
used in solar collector systems.

This module provides:
- Base classes for property correlations (polynomial, exponential, Antoine, etc.)
- FluidProperties base class for complete fluid modeling
- Specific fluid implementations (SYLTHERM 800, etc.)
- Data fitting and validation capabilities
- Integration with manufacturer property data

Classes:
--------
PropertyCorrelation : ABC
    Abstract base class for property correlations
PolynomialCorrelation : PropertyCorrelation
    Polynomial correlation: f(T) = a₀ + a₁T + a₂T² + ...
ExponentialCorrelation : PropertyCorrelation
    Exponential correlation: f(T) = A*exp(B/T) + C
AntoineCorrelation : PropertyCorrelation
    Antoine equation: log₁₀(P) = A + B/(T + C)
PowerLawCorrelation : PropertyCorrelation
    Power law correlation: f(T) = A*T^B
TableCorrelation : PropertyCorrelation
    Tabulated data with interpolation
CustomCorrelation : PropertyCorrelation
    User-defined correlation function
FluidProperties : Base class for complete fluid modeling
Syltherm800 : FluidProperties
    SYLTHERM 800 polydimethylsiloxane heat transfer fluid

Functions:
----------
create_syltherm800 : Create and initialize SYLTHERM 800 fluid
"""

from .fluid_properties import (
    AntoineCorrelation,
    CustomCorrelation,
    ExponentialCorrelation,
    FluidProperties,
    # Correlation types
    PolynomialCorrelation,
    PowerLawCorrelation,
    # Base classes
    PropertyCorrelation,
    TableCorrelation,
)
from .syltherm_800 import (
    Syltherm800,
    create_syltherm800,
)

# Module metadata
__version__ = "0.1.0"
__author__ = "Solar Collector Project"

# Define what gets imported with 'from fluids import *'
__all__ = [
    # Base classes
    "PropertyCorrelation",
    "FluidProperties",
    # Correlation types
    "PolynomialCorrelation",
    "ExponentialCorrelation",
    "AntoineCorrelation",
    "PowerLawCorrelation",
    "TableCorrelation",
    "CustomCorrelation",
    # Specific fluids
    "Syltherm800",
    # Convenience functions
    "create_syltherm800",
]
