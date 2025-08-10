# AdvancedEVEnergyModel – Physics-based Consumption

Purpose: compute realistic EV energy consumption for a GPS trace with weather and vehicle context, with optional per-segment breakdowns.

File: `src/data_generation/advanced_energy_model.py`

## Inputs

- gps_trace: list of dict rows with at least:
  - `timestamp` (ISO 8601), `latitude`, `longitude`
  - Optional: `speed_kmh`, `elevation_m`, `heading`
- vehicle: dict
  - `model` (key in `config.ev_config.EV_MODELS`)
  - `battery_capacity` (kWh)
  - Optional inferred via `BATTERY_PARAMETERS`: `base_internal_resistance`, `nominal_voltage`, `has_heat_pump`
- weather: dict
  - `temperature` (°C), `wind_speed_kmh`, `humidity` (0–1), `is_raining` (bool)

## Outputs

```json
{
  "total_consumption_kwh": float,
  "total_distance_km": float,
  "efficiency_kwh_per_100km": float,
  "temperature_celsius": float,
  "temperature_efficiency_factor": float,
  "battery_internal_resistance_ohm": float,
  "consumption_breakdown": {
    "rolling_resistance": kWh,
    "aerodynamic_drag": kWh,
    "elevation_change": kWh,
    "acceleration": kWh,
    "regenerative_braking": kWh,
    "hvac": kWh,
    "auxiliary": kWh,
    "battery_thermal_loss": kWh
  },
  "weather_conditions": {…},
  "model_version": "advanced_physics_v1.0"
}
```

If `return_segments=True`, a second list is returned with segment-level rows including start/end coordinates/times, speeds, elevations, distance, energy_kwh, breakdown columns, and weather snapshot.

## Model overview

Per segment between consecutive GPS points:

- Distance m via geodesic
- Time delta from timestamps (fallback from speed/distance)
- Dynamics: speeds (m/s), average speed, acceleration, elevation change
- Forces:
  - Rolling: \( F_r = C_r m g \cos(\theta) \) with grade clipped to ±15% and weather/speed multipliers
  - Aerodynamics: \( F_d = \tfrac{1}{2} \rho C_d A v_{rel}^2 \) with RMS of road speed and wind
  - Grade: \( F_g = m g \sin(\theta) \)
- Energies (J): multiply forces by distance; kinetic energy change split into acceleration vs. regenerative recovery (efficiency depends on deceleration magnitude and speed)
- HVAC load (W): function of |ambient − target|, heating vs cooling mode, heat pump usage, vehicle size
- Auxiliary load (W): from `PHYSICS_CONSTANTS['auxiliary_power']` and usage factor
- Battery thermal loss (W): \( I^2 R \) with current estimated from power and nominal voltage
- Drivetrain losses: divide mechanical energy by product of motor, inverter, transmission efficiencies
- Temperature factor: scale total by `TEMPERATURE_EFFICIENCY` interpolation (cold ⇒ higher kWh)

Short segments (<30 m) use a simplified rolling + minimal auxiliary estimate for robustness.

## Equations and constants

- Rolling resistance coefficient with modifiers
  - Base: `PHYSICS_CONSTANTS['rolling_resistance']`
  - Speed factor: `1 + (speed_kmh / 100) * rolling_resistance_speed_factor`
  - Temperature factor: cold/hot multipliers at thresholds 0°C and 35°C
  - Rain factor: ×`rolling_resistance_rain_factor` when `is_raining`

- Air density correction
  - `rho = AIR_DENSITY * (273.15 / temp_K) * (1 - humidity_density_factor * humidity)`
  - Relative wind speed `sqrt(v^2 + w^2)` (RMS) used when wind direction unknown

- Regenerative braking efficiency
  - Base `regen_efficiency` capped by speed and deceleration thresholds: `regen_*_braking_*` constants

- Drivetrain efficiency
  - `motor_efficiency * inverter_efficiency * transmission_efficiency`

- HVAC power
  - Minimal fan load within ±3°C of target cabin temp
  - Heating vs. cooling base loads and slopes, scaled by vehicle mass and heat pump availability

- Battery parameters vs. temperature
  - `R(T) = R0 * exp(Ea/R * (1/T - 1/Tref))`
  - `C_eff = C_nom * (T/Tref)^alpha`

- Unit conversions
  - J → kWh via `/ 3.6e6`; speeds in m/s; distances in meters

Key defaults in `config/physics_constants.py`:
- Rolling resistance: 0.008; air density: 1.225; gravity: 9.81
- Efficiencies: motor 0.9, inverter 0.95, transmission 0.98
- Regen thresholds: speed 15 m/s; deceleration tiers 1.5, 3.0 m/s²
- HVAC: target 21°C; base/responses differ by heating vs cooling and heat pump presence

## Temperature and battery parameters

- Internal resistance temperature dependence: \( R(T) = R_0 e^{E_a / R (1/T - 1/T_{ref})} \)
- Effective capacity factor: \( (T/T_{ref})^{\alpha} \)
- `temperature_efficiency_factor` from lookup with linear interpolation

## Safeguards

- Minimum positive consumption per segment (≥ 1 Wh)
- Trip-level efficiency floors/caps: 8–60 kWh/100 km
- Nonzero minimal result for traces with <2 points (noise-injected realistic stub)

Edge cases handled:
- Non-monotonic or invalid timestamps → estimate `time_diff_s` from distance and avg speed
- Zero/negative time deltas → clamp with default 30 km/h assumption
- Missing elevations/speeds → defaults applied; elevation grade clipped to ±15%

## Configuration references

- `config/physics_constants.py`: constants, HVAC parameters, regen, rolling resistance factors
- `config/ev_config.py` → `EV_MODELS`: mass, Cd, frontal area; `BATTERY_PARAMETERS` for battery electricals

## Detailed logging

- Controlled via `config.logging_config.DETAILED_LOGGING_COMPONENTS['energy_calculation']`
- Writes per-segment forces and energy terms when enabled, using `src/utils/logger.log_detailed`

## Example

```python
from src.data_generation.advanced_energy_model import AdvancedEVEnergyModel

model = AdvancedEVEnergyModel()
result, segments = model.calculate_energy_consumption(
    gps_trace=[...],
    vehicle={"model": "tesla_model_3", "battery_capacity": 75},
    weather={"temperature": 18, "wind_speed_kmh": 10, "is_raining": False, "humidity": 0.6},
    return_segments=True,
)
```


