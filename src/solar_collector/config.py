VAR_INFO = {
    "T_f": {
        "long_name": "Collector Oil Temperature",
        "units": "K",
        "description": "Temperature of the oil inside the pipe",
    },
    "T_p": {
        "long_name": "Collector Pipe Wall Temperature",
        "units": "K",
        "description": "Temperature of the pipe wall",
    },
    "F": {
        "long_name": "Collector Oil Flow Rate",
        "units": "m^3/s",
        "description": "Volumetric flowrate of the oil through the collector",
    },
    "v": {
        "long_name": "Collector Oil Flow Velocity",
        "units": "m/s",
        "description": "Velocity of the oil inside the collector pipe",
    },
    "T_inlet": {
        "long_name": "Collector Oil Inlet Temperature",
        "units": "K",
        "description": "Temperature of the oil at inlet to the collector pipe",
    },
    "q_solar_conc": {
        "long_name": "Concentrated Solar Heat Flux",
        "units": "W/m^2",
        "description": "Heat delivered by solar collector to pipe wall surface",
    },
    "q_solar": {
        "long_name": "Solar Heat Flux",
        "units": "W/m^2",
        "description": "Heat delivered by the sunlight per unit surface area.",
    },
}

PLOT_COLORS = {
    'T_f': 'tab:blue',
    'T_p': 'tab:orange',
    'F': 'tab:green',
    'v': 'tab:green',
    'q_solar_conc': 'tab:purple',
    'q_solar': 'tab:brown'
}