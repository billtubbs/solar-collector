"""Tests for SYLTHERM 800 fluid properties."""

import pytest

from fluids.syltherm_800 import Syltherm800


def test_syltherm800_instantiation():
    """Test instantiation of Syltherm800 with placeholder coefficients."""
    syltherm = Syltherm800(fit_coefficients=False)

    assert syltherm.name == "SYLTHERM 800"
    assert syltherm.T_min == 233.15  # K (-40°C)
    assert syltherm.T_max == 673.15  # K (400°C)
    assert syltherm.molecular_weight == 4.0  # kg/kmol


def test_syltherm800_summary(capsys):
    """Test that summary method runs without error and produces output."""
    syltherm = Syltherm800(fit_coefficients=False)

    syltherm.summary()

    captured = capsys.readouterr()
    assert "SYLTHERM 800 Heat Transfer Fluid" in captured.out
    assert "Polydimethylsiloxane (PDMS)" in captured.out


def test_load_manufacturer_data():
    """Test loading manufacturer data if available."""
    syltherm = Syltherm800()

    try:
        data = syltherm.load_manufacturer_data()
        assert isinstance(data, dict)
        assert "T_K" in data
        assert "density" in data
        assert len(data["T_K"]) > 0
    except FileNotFoundError:
        pytest.skip("Manufacturer data file not found")


def test_validate_correlations():
    """Test validation of correlations against manufacturer data if available."""
    syltherm = Syltherm800(fit_coefficients=False)

    try:
        data = syltherm.load_manufacturer_data()
        results = syltherm.validate_correlations(data)
        assert isinstance(results, dict)
        # Placeholder: actual validation would check specific metrics
    except FileNotFoundError:
        pytest.skip("Manufacturer data file not found")


def test_get_reference_properties():
    """Test getting reference properties at 25°C."""
    syltherm = Syltherm800()

    ref_props = syltherm.get_reference_properties()

    assert ref_props["temperature_C"] == 25.0
    assert ref_props["temperature_K"] == 298.15
    assert "density" in ref_props
    assert "heat_capacity" in ref_props
    assert "thermal_conductivity" in ref_props
    assert "viscosity" in ref_props
