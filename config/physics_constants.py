# Physical constants for energy consumption calculation
PHYSICS_CONSTANTS = {
    'air_density': 1.225,               # kg/m³ at sea level
    'rolling_resistance': 0.01,         # typical for EV tires
    'gravity': 9.81,                    # m/s²
    'regen_efficiency': 0.7,            # 70% energy recovery on braking
    'motor_efficiency': 0.9,            # 90% motor efficiency
    'battery_efficiency': 0.95,         # 95% battery efficiency
    'hvac_base_power': 2.0,            # kW base HVAC consumption
    'auxiliary_power': 0.5,            # kW for lights, electronics, etc.
}

# Temperature effects on battery efficiency
TEMPERATURE_EFFICIENCY = {
    -10: 0.65,  # 65% efficiency at -10°C
    0: 0.75,    # 75% efficiency at 0°C
    10: 0.85,   # 85% efficiency at 10°C
    20: 1.0,    # 100% efficiency at 20°C (optimal)
    25: 1.0,    # 100% efficiency at 25°C
    30: 0.95,   # 95% efficiency at 30°C
    35: 0.90,   # 90% efficiency at 35°C
    40: 0.85    # 85% efficiency at 40°C
}
