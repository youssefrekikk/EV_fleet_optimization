# Most popular EVs in US market with METRIC specs
EV_MODELS = {
    'tesla_model_3': {
        'battery_capacity': 75,     # kWh
        'wltp_range': 491,# km (more realistic than EPA)
        'efficiency': 15.0,# kWh/100km
        'max_charging_speed': 250, # kW
        'weight': 1836,# kg
        'drag_coefficient': 0.23,
        'frontal_area': 2.22,# m²
        'market_share': 0.21,
        "nominal_voltage": 339,
        "internal_resistance_ohms": 0.056,
        "heat_pump": True
    },
    'tesla_model_y': {
        'battery_capacity': 75,
        'wltp_range': 455,
        'efficiency': 16.5,
        'max_charging_speed': 250,
        'weight': 2003,
        'drag_coefficient': 0.23,
        'frontal_area': 2.53,
        'market_share': 0.18,
        "nominal_voltage": 340,
        "internal_resistance_ohms": 0.05,
        "heat_pump": True
    },
    'nissan_leaf': {
        'battery_capacity': 62,
        'wltp_range': 385,
        'efficiency': 16.1,
        'max_charging_speed': 100,
        'weight': 1605,
        'drag_coefficient': 0.28,
        'frontal_area': 2.27,
        'market_share': 0.10,
        "nominal_voltage": 360,
        "internal_resistance_ohms": 0.05,
        "heat_pump": True
    },
    'chevy_bolt': {
        'battery_capacity': 65,
        'wltp_range': 417,
        'efficiency': 15.6,
        'max_charging_speed': 55,
        'weight': 1616,
        'drag_coefficient': 0.32,
        'frontal_area': 2.39,
        'market_share': 0.09,
        "nominal_voltage": 360,
        "internal_resistance_ohms": 0.05,
        "heat_pump": True
    },
    'ford_mustang_mach_e': {
        'battery_capacity': 88,
        'wltp_range': 502,
        'efficiency': 17.5,
        'max_charging_speed': 150,
        'weight': 2232,
        'drag_coefficient': 0.29,
        'frontal_area': 2.64,
        'market_share': 0.08,
        "nominal_voltage": 355,
        "internal_resistance_ohms": 0.05,
        "heat_pump": True
    },
    'hyundai_ioniq_5': {
        'battery_capacity': 77.4,
        'wltp_range': 481,
        'efficiency': 16.1,
        'max_charging_speed': 235,
        'weight': 2268,
        'drag_coefficient': 0.288,
        'frontal_area': 2.85,
        'market_share': 0.07,
        "nominal_voltage": 400,
        "internal_resistance_ohms": 0.04,
        "heat_pump": True
    },
    'volkswagen_id4': {
        'battery_capacity': 77,
        'wltp_range': 520,
        'efficiency': 14.8,
        'max_charging_speed': 125,
        'weight': 2124,
        'drag_coefficient': 0.28,
        'frontal_area': 2.57,
        'market_share': 0.07,
        "nominal_voltage": 400,
        "internal_resistance_ohms": 0.04,
        "heat_pump": True
    },
    'renault_zoe': {
        'battery_capacity': 52,
        'wltp_range': 395,
        'efficiency': 13.2,
        'max_charging_speed': 50,
        'weight': 1468,
        'drag_coefficient': 0.29,
        'frontal_area': 2.0,
        'market_share': 0.05,
        "nominal_voltage": 400,
        "internal_resistance_ohms": 0.04,
        "heat_pump": True
    },
    'kia_ev6': {
        'battery_capacity': 84,
        'wltp_range': 581,
        'efficiency': 15.9,
        'max_charging_speed': 263,
        'weight': 1825,
        'drag_coefficient': 0.28,
        'frontal_area': 2.85,
        'market_share': 0.05,
        "nominal_voltage": 400,
        "internal_resistance_ohms": 0.04,
        "heat_pump": True
    },
    'nio_et5': {
        'battery_capacity': 100,
        'wltp_range': 550,
        'efficiency': 18.0,
        'max_charging_speed': 240,
        'weight': 2145,
        'drag_coefficient': 0.22,
        'frontal_area': 2.7,
        'market_share': 0.03,
        "nominal_voltage": 400,
        "internal_resistance_ohms": 0.03,
        "heat_pump": True
    },
    'audi_e_tron_gt': {
        'battery_capacity': 105,
        'wltp_range': 622,
        'efficiency': 17.8,
        'max_charging_speed': 320,
        'weight': 2355,
        'drag_coefficient': 0.24,
        'frontal_area': 2.60,
        'market_share': 0.03,
        "nominal_voltage": 400,
        "internal_resistance_ohms": 0.04,
        "heat_pump": True
    },
    'audi_q8_e_tron': {
        'battery_capacity': 106,
        'wltp_range': 582,
        'efficiency': 20.0,
        'max_charging_speed': 170,
        'weight': 2600,
        'drag_coefficient': 0.28,
        'frontal_area': 2.9,
        'market_share': 0.02,
        "nominal_voltage": 396,
        "internal_resistance_ohms": 0.04,
        "heat_pump": True
    },
    'audi_a6_e_tron': {
        'battery_capacity': 100,
        'wltp_range': 666,
        'efficiency': 15.0,
        'max_charging_speed': 270,
        'weight': 2250,
        'drag_coefficient': 0.25,
        'frontal_area': 2.6,
        'market_share': 0.02,
        "nominal_voltage": 400,
        "internal_resistance_ohms": 0.04,
        "heat_pump": True
    }
}

