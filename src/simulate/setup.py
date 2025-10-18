"""Setup utilities for simulation configurations."""

import pint


def read_param_values(params_dict, parent_key="", sep="_"):
    """
    Flatten nested parameter dictionary by concatenating keys.

    Returns a dictionary where each parameter is a dictionary with a
    'value' and any other attributes such as 'units'.

    Parameters
    ----------
    params_dict : dict
        Nested dictionary of parameters to flatten. Leaf nodes should have
        'value' and optionally 'units' keys (as strings).
    parent_key : str, optional
        Prefix for keys (used in recursion), by default ''
    sep : str, optional
        Separator between nested keys, by default '_'

    Returns
    -------
    dict
        Flat dictionary with concatenated keys. Each value is a dict with
        'value' and any additional fields (e.g., 'units', 'name', 'desc')
        from the original parameter dict.

    Examples
    --------
    >>> params = {
    ...     'fluid': {
    ...         'thermal_diffusivity': {'value': 0.25, 'units': 'm**2/s'},
    ...         'density': {
    ...             'value': 800,
    ...             'units': 'kg/m**3',
    ...             'name': 'Fluid Density',
    ...             'desc': 'The density of the fluid'
    ...         }
    ...     },
    ...     'collector': {
    ...         'diameter': {'value': 0.07, 'units': 'm'}
    ...     }
    ... }
    >>> result = read_param_values(params)
    >>> result['fluid_thermal_diffusivity']
    {'value': 0.25, 'units': 'm**2/s'}
    >>> result['fluid_density']
    {'value': 800, 'units': 'kg/m**3', 'name': 'Fluid Density',
     'desc': 'The density of the fluid'}

    Notes
    -----
    - Extracts all fields from leaf dictionaries (those with 'value' key)
    - Any additional fields (e.g., 'name', 'desc') are preserved in output
    - For non-dict values, stores as dict with value and units=None
    - Handles arbitrary nesting depth

    See Also
    --------
    read_param_values_pint : Converts units to pint unit objects
    """
    items = []

    for key, value in params_dict.items():
        # Construct new key by concatenating parent and current key
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            # Check if this is a leaf node with 'value' key
            if "value" in value:
                # Extract value and units, plus any additional fields
                param_dict = {"value": value["value"]}

                # Copy any additional fields (name, desc, etc.)
                for field_key, field_value in value.items():
                    if field_key != "value":
                        param_dict[field_key] = field_value

                items.append((new_key, param_dict))
            else:
                # Recursively flatten nested dictionaries
                items.extend(
                    read_param_values(
                        value, parent_key=new_key, sep=sep
                    ).items()
                )
        else:
            # Non-dict values: store value with no units
            items.append((new_key, {"value": value, "units": None}))

    return dict(items)


def read_param_values_pint(params_dict, ureg=None, parent_key="", sep="_"):
    """
    Flatten nested parameter dictionary and convert units to pint objects.

    Flattens a nested dictionary of parameters by concatenating keys with a
    separator. Converts unit strings to pint unit objects using the specified
    or default unit registry.

    Parameters
    ----------
    params_dict : dict
        Nested dictionary of parameters to flatten. Leaf nodes should have
        'value' and optionally 'units' keys (as strings).
    ureg : pint.UnitRegistry, optional
        Unit registry to use for creating unit objects. If None, a new
        registry is created.
    parent_key : str, optional
        Prefix for keys (used in recursion), by default ''
    sep : str, optional
        Separator between nested keys, by default '_'

    Returns
    -------
    dict
        Flat dictionary with concatenated keys. Each value is a dict with
        'value' and 'units' keys, plus any additional fields from the
        original parameter dict. 'units' is a pint Unit object if units
        were specified in the input, otherwise None.

    Examples
    --------
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> params = {
    ...     'fluid': {
    ...         'thermal_diffusivity': {'value': 0.25, 'units': 'm**2/s'},
    ...         'density': {
    ...             'value': 800,
    ...             'units': 'kg/m**3',
    ...             'name': 'Fluid Density',
    ...             'desc': 'The density of the fluid'
    ...         }
    ...     },
    ...     'collector': {
    ...         'diameter': {'value': 0.07, 'units': 'm'}
    ...     }
    ... }
    >>> result = read_param_values_pint(params, ureg)
    >>> result['fluid_thermal_diffusivity']
    {'value': 0.25, 'units': <Unit('meter ** 2 / second')>}
    >>> result['fluid_density']
    {'value': 800, 'units': <Unit('kilogram / meter ** 3')>,
     'name': 'Fluid Density', 'desc': 'The density of the fluid'}

    Notes
    -----
    - Converts 'units' field from string to pint Unit object
    - If no units specified in input, 'units' will be None
    - Any additional fields (e.g., 'name', 'desc') are preserved in output
    - For non-dict values (no 'value' key), stores as value with units=None
    - Handles arbitrary nesting depth
    - Internally calls read_param_values() then converts units

    See Also
    --------
    read_param_values : Flatten without converting units to pint objects
    """
    if ureg is None:
        ureg = pint.UnitRegistry()
    params_flat = read_param_values(
        params_dict, parent_key=parent_key, sep=sep
    )

    for value in params_flat.values():
        # Convert units if present
        value["units"] = (
            ureg(value["units"]).units if "units" in value else None
        )

    return params_flat
