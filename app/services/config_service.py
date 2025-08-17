from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field, conint, confloat, validator


UI_OVERRIDES_PATH = Path("config/ui_overrides.yaml")


class FleetConfigSchema(BaseModel):
    fleet_size: conint(ge=1, le=500) = Field(10)
    simulation_days: conint(ge=1, le=60) = Field(7)
    region: str = Field("bay_area")


class ChargingConfigSchema(BaseModel):
    enable_home_charging: bool = True
    home_charging_availability: confloat(ge=0, le=1) = 0.8
    home_charging_power: confloat(ge=1, le=350) = 7.4
    public_fast_charging_power: confloat(ge=20, le=350) = 150
    charging_efficiency: confloat(ge=0.6, le=1.0) = 0.9


class PhysicsConfigSchema(BaseModel):
    # Core physics constants (exactly matching PHYSICS_CONSTANTS)
    air_density: confloat(ge=0.5, le=2.0) = 1.225
    rolling_resistance: confloat(ge=0.005, le=0.02) = 0.008
    gravity: confloat(ge=9.0, le=10.0) = 9.81
    regen_efficiency: confloat(ge=0.5, le=0.95) = 0.8
    motor_efficiency: confloat(ge=0.7, le=0.98) = 0.9
    battery_efficiency: confloat(ge=0.8, le=0.99) = 0.95
    hvac_base_power: confloat(ge=0.5, le=5.0) = 2.0
    auxiliary_power: confloat(ge=0.1, le=1.0) = 0.25
    auxiliary_usage_factor: confloat(ge=0.1, le=1.0) = 0.5
    
    # Advanced physics constants
    gas_constant: confloat(ge=8.0, le=9.0) = 8.314
    activation_energy: confloat(ge=15000.0, le=25000.0) = 20000.0
    reference_temp_k: confloat(ge=270.0, le=310.0) = 298.15
    temp_capacity_alpha: confloat(ge=0.1, le=1.0) = 0.5
    inverter_efficiency: confloat(ge=0.9, le=0.99) = 0.95
    transmission_efficiency: confloat(ge=0.95, le=0.99) = 0.98
    
    # HVAC model parameters
    hvac_cop_heat_pump: confloat(ge=2.0, le=4.0) = 3.0
    hvac_cop_resistive: confloat(ge=0.8, le=1.2) = 1.0
    cabin_thermal_mass: confloat(ge=30000.0, le=70000.0) = 50000.0
    cabin_heat_loss_coeff: confloat(ge=50.0, le=150.0) = 100.0
    target_cabin_temp: confloat(ge=18.0, le=25.0) = 21.0
    
    # Rolling resistance factors
    rolling_resistance_speed_factor: confloat(ge=0.1, le=0.2) = 0.15
    rolling_resistance_cold_factor: confloat(ge=1.1, le=1.2) = 1.15
    rolling_resistance_hot_factor: confloat(ge=1.0, le=1.1) = 1.05
    rolling_resistance_rain_factor: confloat(ge=1.15, le=1.25) = 1.2
    
    # Air density factors
    humidity_density_factor: confloat(ge=0.3, le=0.5) = 0.378
    
    # Regenerative braking parameters
    regen_speed_threshold: confloat(ge=5.0, le=25.0) = 15.0
    regen_hard_braking_threshold: confloat(ge=1.0, le=5.0) = 3.0
    regen_moderate_braking_threshold: confloat(ge=0.5, le=3.0) = 1.5
    regen_hard_braking_efficiency: confloat(ge=0.3, le=0.8) = 0.6
    regen_moderate_braking_efficiency: confloat(ge=0.7, le=0.95) = 0.85
    
    # Minimum consumption parameters
    min_consumption_per_km: confloat(ge=0.01, le=0.05) = 0.02
    min_driving_time_hours: confloat(ge=0.1, le=1.0) = 0.5


class DriverProfilesConfigSchema(BaseModel):
    # Driver profile proportions (must sum to 1.0)
    commuter_proportion: confloat(ge=0.0, le=1.0) = 0.35
    rideshare_proportion: confloat(ge=0.0, le=1.0) = 0.25
    delivery_proportion: confloat(ge=0.0, le=1.0) = 0.20
    casual_proportion: confloat(ge=0.0, le=1.0) = 0.20
    
    # Commuter profile parameters
    commuter_daily_km_min: conint(ge=30, le=200) = 65
    commuter_daily_km_max: conint(ge=50, le=300) = 130
    commuter_trips_per_day_min: conint(ge=1, le=10) = 2
    commuter_trips_per_day_max: conint(ge=2, le=15) = 4
    commuter_avg_speed_city: conint(ge=15, le=50) = 25
    commuter_avg_speed_highway: conint(ge=60, le=120) = 90
    commuter_home_charging_prob: confloat(ge=0.0, le=1.0) = 0.9
    commuter_charging_threshold: confloat(ge=0.1, le=0.8) = 0.3
    commuter_weekend_factor: confloat(ge=0.1, le=2.0) = 0.3
    
    # Rideshare profile parameters
    rideshare_daily_km_min: conint(ge=100, le=500) = 160
    rideshare_daily_km_max: conint(ge=200, le=800) = 320
    rideshare_trips_per_day_min: conint(ge=10, le=50) = 15
    rideshare_trips_per_day_max: conint(ge=20, le=80) = 25
    rideshare_avg_speed_city: conint(ge=15, le=50) = 22
    rideshare_avg_speed_highway: conint(ge=60, le=120) = 85
    rideshare_home_charging_prob: confloat(ge=0.0, le=1.0) = 0.6
    rideshare_charging_threshold: confloat(ge=0.1, le=0.8) = 0.2
    rideshare_weekend_factor: confloat(ge=0.1, le=2.0) = 1.2
    
    # Delivery profile parameters
    delivery_daily_km_min: conint(ge=150, le=800) = 240
    delivery_daily_km_max: conint(ge=300, le=1000) = 480
    delivery_trips_per_day_min: conint(ge=15, le=80) = 20
    delivery_trips_per_day_max: conint(ge=25, le=100) = 40
    delivery_avg_speed_city: conint(ge=15, le=50) = 20
    delivery_avg_speed_highway: conint(ge=60, le=120) = 80
    delivery_home_charging_prob: confloat(ge=0.0, le=1.0) = 0.3
    delivery_charging_threshold: confloat(ge=0.1, le=0.8) = 0.25
    delivery_weekend_factor: confloat(ge=0.1, le=2.0) = 0.8
    
    # Casual profile parameters
    casual_daily_km_min: conint(ge=10, le=150) = 30
    casual_daily_km_max: conint(ge=20, le=300) = 100
    casual_trips_per_day_min: conint(ge=1, le=8) = 1
    casual_trips_per_day_max: conint(ge=2, le=12) = 3
    casual_avg_speed_city: conint(ge=15, le=50) = 30
    casual_avg_speed_highway: conint(ge=60, le=120) = 95
    casual_home_charging_prob: confloat(ge=0.0, le=1.0) = 0.95
    casual_charging_threshold: confloat(ge=0.1, le=0.8) = 0.4
    casual_weekend_factor: confloat(ge=0.1, le=2.0) = 1.5





class UIOverrides(BaseModel):
    fleet: FleetConfigSchema = FleetConfigSchema()
    charging: ChargingConfigSchema = ChargingConfigSchema()
    physics: PhysicsConfigSchema = PhysicsConfigSchema()
    driver_profiles: DriverProfilesConfigSchema = DriverProfilesConfigSchema()
    # Optional distributions to override EV models and driver profiles
    ev_models_market_share: Dict[str, float] | None = None
    ev_models_parameters: Dict[str, Dict[str, Any]] | None = None
    driver_profiles_proportion: Dict[str, float] | None = None


def load_overrides() -> UIOverrides:
    if UI_OVERRIDES_PATH.exists():
        data = yaml.safe_load(UI_OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
        return UIOverrides(**data)
    return UIOverrides()


def save_overrides(overrides: UIOverrides) -> None:
    UI_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_OVERRIDES_PATH.write_text(yaml.safe_dump(overrides.dict(), sort_keys=False), encoding="utf-8")


def merged_runtime_config() -> Dict[str, Any]:
    # Import python configs lazily to avoid circular imports with Streamlit reloads
    from config.ev_config import (
        FLEET_CONFIG as FLEET_DEFAULT,
        CHARGING_CONFIG as CHARGING_DEFAULT,
    )
    from config.physics_constants import PHYSICS_CONSTANTS

    ui = load_overrides()

    fleet = {**FLEET_DEFAULT, **ui.fleet.dict()}
    charging = {**CHARGING_DEFAULT, **ui.charging.dict()}
    physics = {**PHYSICS_CONSTANTS, **ui.physics.dict()}

    # Use driver profile proportions directly
    driver_proportions = {k.replace('_proportion', ''): float(v) for k, v in ui.driver_profiles.dict().items()}

    # Apply optional distributions to runtime (not mutating code files)
    ev_shares = (ui.ev_models_market_share or {}).copy()
    if ev_shares:
        total = sum(ev_shares.values()) or 1.0
        ev_shares = {k: float(v) / total for k, v in ev_shares.items()}
    driver_props = (ui.driver_profiles_proportion or {}).copy()
    if driver_props:
        totalp = sum(driver_props.values()) or 1.0
        driver_props = {k: float(v) / totalp for k, v in driver_props.items()}

    # Apply EV model parameter overrides
    ev_params = (ui.ev_models_parameters or {}).copy()
    
    return {
        "fleet": fleet,
        "charging": charging,
        "physics": physics,
        "driver_profiles_proportion": driver_proportions,
        "ev_models_market_share": ev_shares or None,
        "ev_models_parameters": ev_params or None,
        "driver_profiles_proportion_override": driver_props or None,
    }

