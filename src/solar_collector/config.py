VAR_INFO = {
    "T_f": {
        "long_name": "Oil Temperature",
        "units": "K",
        "description": "Temperature of the oil inside the absorber pipe",
    },
    "T_p": {
        "long_name": "Absorber Pipe Wall Temperature",
        "units": "K",
        "description": "Temperature of the absorber pipe wall",
    },
    "F": {
        "long_name": "Oil Flow Rate",
        "units": "m^3/s",
        "description": "Volumetric flowrate of the oil through the collector",
    },
    "v": {
        "long_name": "Oil Flow Velocity",
        "units": "m/s",
        "description": "Velocity of the oil inside the absorber pipe",
    },
    "T_inlet": {
        "long_name": "Oil Inlet Temperature",
        "units": "K",
        "description": "Temperature of the oil at inlet to the absorber pipe",
    },
    "q_solar_conc": {
        "long_name": "Concentrated Solar Heat Flux",
        "units": "W/m^2",
        "description": "Heat delivered by solar collector to absorber pipe wall",
    },
    "q_solar": {
        "long_name": "Solar Heat Flux",
        "units": "W/m^2",
        "description": "Heat delivered by the sunlight per unit surface area.",
    },
}

PLOT_COLORS = {
    "T_f": "tab:blue",
    "T_p": "tab:orange",
    "F": "tab:green",
    "v": "tab:green",
    "q_solar_conc": "tab:purple",
    "q_solar": "tab:brown",
}
