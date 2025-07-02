# Most popular EVs in US market with METRIC specs
EV_MODELS = {
    'tesla_model_3': {
        'battery_capacity': 75,      # kWh
        'wltp_range': 491,          # km (more realistic than EPA)
        'efficiency': 15.0,         # kWh/100km
        'max_charging_speed': 250,   # kW
        'weight': 1836,             # kg
        'drag_coefficient': 0.23,
        'frontal_area': 2.22,       # m²
        'market_share': 0.25
    },
    'tesla_model_y': {
        'battery_capacity': 75,
        'wltp_range': 455,          # km
        'efficiency': 16.5,         # kWh/100km
        'max_charging_speed': 250,
        'weight': 2003,             # kg
        'drag_coefficient': 0.23,
        'frontal_area': 2.53,       # m²
        'market_share': 0.20
    },
    'nissan_leaf': {
        'battery_capacity': 62,
        'wltp_range': 385,          # km
        'efficiency': 16.1,         # kWh/100km
        'max_charging_speed': 100,
        'weight': 1605,             # kg
        'drag_coefficient': 0.28,
        'frontal_area': 2.27,       # m²
        'market_share': 0.15
    },
    'chevy_bolt': {
        'battery_capacity': 65,
        'wltp_range': 417,          # km
        'efficiency': 15.6,         # kWh/100km
        'max_charging_speed': 55,
        'weight': 1616,             # kg
        'drag_coefficient': 0.32,
        'frontal_area': 2.39,       # m²
        'market_share': 0.12
    },
    'ford_mustang_mach_e': {
        'battery_capacity': 88,
        'wltp_range': 502,          # km
        'efficiency': 17.5,         # kWh/100km
        'max_charging_speed': 150,
        'weight': 2232,             # kg
        'drag_coefficient': 0.29,
        'frontal_area': 2.64,       # m²
        'market_share': 0.10
    },
    'hyundai_ioniq_5': {
        'battery_capacity': 77.4,
        'wltp_range': 481,          # km
        'efficiency': 16.1,         # kWh/100km
        'max_charging_speed': 235,
        'weight': 2268,             # kg
        'drag_coefficient': 0.288,
        'frontal_area': 2.85,       # m²
        'market_share': 0.08
    },
    'volkswagen_id4': {
        'battery_capacity': 77,
        'wltp_range': 520,          # km
        'efficiency': 14.8,         # kWh/100km
        'max_charging_speed': 125,
        'weight': 2124,             # kg
        'drag_coefficient': 0.28,
        'frontal_area': 2.57,       # m²
        'market_share': 0.10
    }
}
