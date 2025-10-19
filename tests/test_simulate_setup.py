"""Tests for simulate.setup module."""

import pint

from simulate.setup import read_param_values, read_param_values_pint


def test_read_param_values_basic():
    """Test basic flattening with two levels (units as strings)."""
    params = {
        "fluid": {
            "thermal_diffusivity": {"value": 0.25, "units": "m**2/s"},
            "density": {"value": 800, "units": "kg/m**3"},
        },
        "collector": {"diameter": {"value": 0.07, "units": "m"}},
    }

    result = read_param_values(params)

    # Check keys
    assert "fluid_thermal_diffusivity" in result
    assert "fluid_density" in result
    assert "collector_diameter" in result

    # Check structure and values (units should be strings)
    assert result["fluid_thermal_diffusivity"]["value"] == 0.25
    assert result["fluid_thermal_diffusivity"]["units"] == "m**2/s"
    assert result["fluid_density"]["value"] == 800
    assert result["fluid_density"]["units"] == "kg/m**3"
    assert result["collector_diameter"]["value"] == 0.07
    assert result["collector_diameter"]["units"] == "m"


def test_read_param_values_pint_basic():
    """Test basic flattening with two levels and pint units."""
    ureg = pint.UnitRegistry()
    params = {
        "fluid": {
            "thermal_diffusivity": {"value": 0.25, "units": "m**2/s"},
            "density": {"value": 800, "units": "kg/m**3"},
        },
        "collector": {"diameter": {"value": 0.07, "units": "m"}},
    }

    result = read_param_values_pint(params, ureg)

    # Check keys
    assert "fluid_thermal_diffusivity" in result
    assert "fluid_density" in result
    assert "collector_diameter" in result

    # Check structure and values (units should be pint objects)
    assert result["fluid_thermal_diffusivity"]["value"] == 0.25
    assert result["fluid_thermal_diffusivity"]["units"] == ureg("m**2/s").units
    assert result["fluid_density"]["value"] == 800
    assert result["fluid_density"]["units"] == ureg("kg/m**3").units
    assert result["collector_diameter"]["value"] == 0.07
    assert result["collector_diameter"]["units"] == ureg("m").units


def test_read_param_values_three_levels():
    """Test flattening with three nested levels."""
    params = {
        "system": {
            "fluid": {
                "thermal_diffusivity": {"value": 0.25, "units": "m**2/s"}
            }
        }
    }

    result = read_param_values(params)

    assert "system_fluid_thermal_diffusivity" in result
    assert result["system_fluid_thermal_diffusivity"]["value"] == 0.25
    assert result["system_fluid_thermal_diffusivity"]["units"] == "m**2/s"


def test_read_param_values_mixed_types():
    """Test flattening with mixed value types (with and without units)."""
    params = {
        "fluid": {
            "temperature": {"value": 270.0, "units": "degC"},
            "density": {"value": 800, "units": "kg/m**3"},
        },
        "discretization": {
            "n_x": 50,  # No 'value' key, just a plain integer
            "n_t": 100,
        },
    }

    result = read_param_values(params)

    # Parameters with units (should be strings)
    assert result["fluid_temperature"]["value"] == 270.0
    assert result["fluid_temperature"]["units"] == "degC"
    assert result["fluid_density"]["value"] == 800
    assert result["fluid_density"]["units"] == "kg/m**3"

    # Parameters without 'value'/'units' structure have units=None
    assert result["discretization_n_x"]["value"] == 50
    assert result["discretization_n_x"]["units"] is None
    assert result["discretization_n_t"]["value"] == 100
    assert result["discretization_n_t"]["units"] is None


def test_read_param_values_custom_separator():
    """Test flattening with custom separator."""
    params = {"fluid": {"density": {"value": 800, "units": "kg/m**3"}}}

    result = read_param_values(params, sep=".")

    assert "fluid.density" in result
    assert result["fluid.density"]["value"] == 800
    assert result["fluid.density"]["units"] == "kg/m**3"


def test_read_param_values_empty_dict():
    """Test flattening empty dictionary."""
    result = read_param_values({})
    assert result == {}


def test_read_param_values_no_nesting():
    """Test flattening with no nesting (single level)."""
    params = {
        "temperature": {"value": 20.0, "units": "degC"},
        "pressure": {"value": 101325, "units": "Pa"},
    }

    result = read_param_values(params)

    assert result["temperature"]["value"] == 20.0
    assert result["temperature"]["units"] == "degC"
    assert result["pressure"]["value"] == 101325
    assert result["pressure"]["units"] == "Pa"


def test_read_param_values_yaml_structure():
    """Test with structure from steps_01.yaml (units as strings)."""
    # This matches the structure under system.model.params in steps_01.yaml
    params = {
        "fluid": {
            "thermal_diffusivity": {"value": 0.25, "units": "m**2/s"},
            "density": {"value": 800, "units": "kg/m**3"},
            "specific_heat_capacity": {"value": 2000.0, "units": "J/(kg*K)"},
            "heat_transfer_coeff_ext": {"value": 10.0, "units": "W/(m**2*K)"},
        },
        "collector": {
            "diameter": {"value": 0.07, "units": "m"},
            "length": {"value": 100.0, "units": "m"},
        },
        "ambient": {
            "T": {"value": 20.0, "units": "degC"},
            "heat_transfer_coeff": {"value": 10.0, "units": "W/(m**2*K)"},
        },
    }

    result = read_param_values(params)

    # Check all keys exist
    expected_keys = [
        "fluid_thermal_diffusivity",
        "fluid_density",
        "fluid_specific_heat_capacity",
        "fluid_heat_transfer_coeff_ext",
        "collector_diameter",
        "collector_length",
        "ambient_T",
        "ambient_heat_transfer_coeff",
    ]
    for key in expected_keys:
        assert key in result

    # Spot check some values and units (units should be strings)
    assert result["fluid_thermal_diffusivity"]["value"] == 0.25
    assert result["fluid_thermal_diffusivity"]["units"] == "m**2/s"
    assert result["fluid_density"]["value"] == 800
    assert result["fluid_density"]["units"] == "kg/m**3"
    assert result["collector_length"]["value"] == 100.0
    assert result["collector_length"]["units"] == "m"
    assert result["ambient_T"]["value"] == 20.0
    assert result["ambient_T"]["units"] == "degC"


def test_read_param_values_no_units():
    """Test parameters with value but no units."""
    params = {
        "settings": {"tolerance": {"value": 1e-6}, "max_iter": {"value": 1000}}
    }

    result = read_param_values(params)

    # Should not have 'units' key when no units specified
    assert result["settings_tolerance"]["value"] == 1e-6
    assert "units" not in result["settings_tolerance"]
    assert result["settings_max_iter"]["value"] == 1000
    assert "units" not in result["settings_max_iter"]


def test_read_param_values_additional_fields():
    """Test that additional fields like 'name' and 'desc' are preserved."""
    params = {
        "fluid": {
            "density": {
                "value": 800,
                "units": "kg/m**3",
                "name": "Fluid Density",
                "desc": "The density of the fluid at normal operating conditions",
            },
            "temperature": {
                "value": 270.0,
                "units": "degC",
                "name": "Fluid Temperature",
            },
        },
        "collector": {
            "length": {
                "value": 100.0,
                "units": "m",
                "desc": "Total length of collector pipe",
            }
        },
    }

    result = read_param_values(params)

    # Check that value and units are correct (units should be strings)
    assert result["fluid_density"]["value"] == 800
    assert result["fluid_density"]["units"] == "kg/m**3"

    # Check that additional fields are preserved
    assert result["fluid_density"]["name"] == "Fluid Density"
    assert (
        result["fluid_density"]["desc"]
        == "The density of the fluid at normal operating conditions"
    )

    # Check parameter with only name field
    assert result["fluid_temperature"]["name"] == "Fluid Temperature"
    assert "desc" not in result["fluid_temperature"]

    # Check parameter with only desc field
    assert (
        result["collector_length"]["desc"] == "Total length of collector pipe"
    )
    assert "name" not in result["collector_length"]
