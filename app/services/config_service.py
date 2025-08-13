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


class OptimizationConfigSchema(BaseModel):
    route_optimization_algorithm: str = Field("dijkstra")  # or 'astar'
    gamma_time_weight: confloat(ge=0, le=1) = 0.02
    price_weight_kwh_per_usd: confloat(ge=0, le=10) = 0.0
    battery_buffer_percentage: confloat(ge=0.05, le=0.5) = 0.15
    max_detour_for_charging_km: confloat(ge=0, le=50) = 5.0
    # Fleet evaluation
    fleet_eval_max_days: int | None = None
    fleet_eval_trip_sample_frac: confloat(ge=0, le=1) | None = 0.7
    # SOC routing objective
    soc_objective: str = Field("energy")
    alpha_usd_per_hour: confloat(ge=0, le=200) = 0.0
    beta_kwh_to_usd: confloat(ge=0, le=5) = 0.0
    planning_mode: str = Field("myopic")
    reserve_soc: confloat(ge=0.0, le=0.95) = 0.15
    reserve_kwh: confloat(ge=0.0, le=200.0) = 0.0
    horizon_trips: conint(ge=0, le=10) = 1
    horizon_hours: confloat(ge=0.0, le=72.0) = 0.0

    @validator("route_optimization_algorithm")
    def _algo_choice(cls, v: str) -> str:
        if v not in {"dijkstra", "astar"}:
            raise ValueError("route_optimization_algorithm must be 'dijkstra' or 'astar'")
        return v

    @validator("soc_objective")
    def _soc_choice(cls, v: str) -> str:
        if v not in {"energy", "cost", "time", "weighted"}:
            raise ValueError("soc_objective must be one of: energy, cost, time, weighted")
        return v

    @validator("planning_mode")
    def _planning_choice(cls, v: str) -> str:
        if v not in {"myopic", "next_trip", "rolling_horizon"}:
            raise ValueError("planning_mode must be one of: myopic, next_trip, rolling_horizon")
        return v


class UIOverrides(BaseModel):
    fleet: FleetConfigSchema = FleetConfigSchema()
    charging: ChargingConfigSchema = ChargingConfigSchema()
    optimization: OptimizationConfigSchema = OptimizationConfigSchema()
    # Optional distributions to override EV models and driver profiles
    ev_models_market_share: Dict[str, float] | None = None
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
    from config.optimization_config import OPTIMIZATION_CONFIG as OPT_DEFAULT

    ui = load_overrides()

    fleet = {**FLEET_DEFAULT, **ui.fleet.dict()}
    charging = {**CHARGING_DEFAULT, **ui.charging.dict()}
    optimization = {**OPT_DEFAULT, **ui.optimization.dict()}

    # Apply optional distributions to runtime (not mutating code files)
    ev_shares = (ui.ev_models_market_share or {}).copy()
    if ev_shares:
        total = sum(ev_shares.values()) or 1.0
        ev_shares = {k: float(v) / total for k, v in ev_shares.items()}
    driver_props = (ui.driver_profiles_proportion or {}).copy()
    if driver_props:
        totalp = sum(driver_props.values()) or 1.0
        driver_props = {k: float(v) / totalp for k, v in driver_props.items()}

    return {
        "fleet": fleet,
        "charging": charging,
        "optimization": optimization,
        "ev_models_market_share": ev_shares or None,
        "driver_profiles_proportion": driver_props or None,
    }

    return {
        "fleet": fleet,
        "charging": charging,
        "optimization": optimization,
    }

