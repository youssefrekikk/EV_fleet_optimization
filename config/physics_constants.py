# Physical constants for energy consumption calculation
PHYSICS_CONSTANTS = {
    'air_density': 1.225,               # kg/m³ at sea level
    'rolling_resistance': 0.008,        # Updated: typical for modern EV tires
    'gravity': 9.81,                    # m/s²
    'regen_efficiency': 0.8,            # 70% energy recovery on braking
    'motor_efficiency': 0.9,            # 90% motor efficiency
    'battery_efficiency': 0.95,         # 95% battery efficiency
    'hvac_base_power': 2,            # kW base HVAC consumption
    'auxiliary_power': 0.25,            # kW for lights, electronics, etc.
    'auxiliary_usage_factor': 0.5,      # 50% usage factor for auxiliary systems
    # New advanced physics constants
    'gas_constant': 8.314,             # J/(mol·K) - universal gas constant
    'activation_energy': 20000,        # J/mol - typical for Li-ion batteries
    'reference_temp_k': 298.15,        # K (25°C) - reference temperature
    'temp_capacity_alpha': 0.5,        # Temperature capacity coefficient
    'inverter_efficiency': 0.95,       # Inverter efficiency
    'transmission_efficiency': 0.98,   # Transmission efficiency
    
    # HVAC model parameters
    'hvac_cop_heat_pump': 3.0,         # Coefficient of performance for heat pump
    'hvac_cop_resistive': 1.0,         # Resistive heating COP
    'cabin_thermal_mass': 50000,       # J/K (approximate cabin thermal mass)
    'cabin_heat_loss_coeff': 100,      # W/K (cabin heat loss coefficient)
    'target_cabin_temp': 21.0,         # °C target cabin temperature
    
    # Rolling resistance factors
    'rolling_resistance_speed_factor': 0.15,    # 10% increase per 100 km/h
    'rolling_resistance_cold_factor': 1.15,    # 15% increase in cold (<0°C)
    'rolling_resistance_hot_factor': 1.05,     # 5% increase in heat (>35°C)
    'rolling_resistance_rain_factor': 1.2,     # 20% increase in rain
    
    # Air density factors
    'humidity_density_factor': 0.378,          # Humidity correction factor
    
    # Regenerative braking parameters
    'regen_speed_threshold': 15.0,             # m/s - full efficiency above this speed
    'regen_hard_braking_threshold': 3.0,       # m/s² - hard braking threshold
    'regen_moderate_braking_threshold': 1.5,   # m/s² - moderate braking threshold
    'regen_hard_braking_efficiency': 0.6,      # 50% efficiency for hard braking
    'regen_moderate_braking_efficiency': 0.85,  # 80% efficiency for moderate braking
    
    # Minimum consumption parameters
    'min_consumption_per_km': 0.02,            # kWh/km minimum consumption
    'min_driving_time_hours': 0.5,             # Minimum assumed driving time
}

# Temperature effects on battery efficiency (enhanced with more data points)
TEMPERATURE_EFFICIENCY = {
    -20: 0.55,  # 55% efficiency at -20°C (extreme cold)
    -15: 0.60,  # 60% efficiency at -15°C
    -10: 0.65,  # 65% efficiency at -10°C
    -5: 0.70,   # 70% efficiency at -5°C
    0: 0.75,    # 75% efficiency at 0°C
    5: 0.82,    # 82% efficiency at 5°C
    10: 0.88,   # 88% efficiency at 10°C
    15: 0.95,   # 95% efficiency at 15°C
    20: 1.0,    # 100% efficiency at 20°C (optimal)
    25: 1.0,    # 100% efficiency at 25°C (optimal)
    30: 0.97,   # 97% efficiency at 30°C
    35: 0.93,   # 93% efficiency at 35°C
    40: 0.88,   # 88% efficiency at 40°C
    45: 0.85    # 85% efficiency at 45°C (extreme heat)
}

# Add this at the end of the file:

# Vehicle-specific battery parameters (extracted from ev_models.py)
BATTERY_PARAMETERS = {
    'tesla_model_3': {
        'internal_resistance': 0.056,    # Ohms
        'nominal_voltage': 339,          # Volts
        'has_heat_pump': True
    },
    'tesla_model_y': {
        'internal_resistance': 0.05,     # Ohms
        'nominal_voltage': 340,          # Volts
        'has_heat_pump': True
    },
    'nissan_leaf': {
        'internal_resistance': 0.05,     # Ohms
        'nominal_voltage': 360,          # Volts
        'has_heat_pump': True
    },
    'chevy_bolt': {
        'internal_resistance': 0.05,     # Ohms
        'nominal_voltage': 360,          # Volts
        'has_heat_pump': True
    },
    'ford_mustang_mach_e': {
        'internal_resistance': 0.05,     # Ohms
        'nominal_voltage': 355,          # Volts
        'has_heat_pump': True
    },
    'hyundai_ioniq_5': {
        'internal_resistance': 0.04,     # Ohms
        'nominal_voltage': 400,          # Volts
        'has_heat_pump': True
    },
    'volkswagen_id4': {
        'internal_resistance': 0.04,     # Ohms
        'nominal_voltage': 400,          # Volts
        'has_heat_pump': True
    },
    'renault_zoe': {
        'internal_resistance': 0.04,     # Ohms
        'nominal_voltage': 400,          # Volts
        'has_heat_pump': True
    },
    'kia_ev6': {
        'internal_resistance': 0.04,     # Ohms
        'nominal_voltage': 400,          # Volts
        'has_heat_pump': True
    },
    'nio_et5': {
        'internal_resistance': 0.03,     # Ohms
        'nominal_voltage': 400,          # Volts
        'has_heat_pump': True
    },
    'audi_e_tron_gt': {
        'internal_resistance': 0.04,     # Ohms
        'nominal_voltage': 400,          # Volts
        'has_heat_pump': True
    },
    'audi_q8_e_tron': {
        'internal_resistance': 0.04,     # Ohms
        'nominal_voltage': 396,          # Volts
        'has_heat_pump': True
    },
    'audi_a6_e_tron': {
        'internal_resistance': 0.04,     # Ohms
        'nominal_voltage': 400,          # Volts
        'has_heat_pump': True
    }
}


