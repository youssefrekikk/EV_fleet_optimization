from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.data_generation.synthetic_ev_generator import SyntheticEVGenerator


def generate_synthetic_data(
    fleet_size: int,
    simulation_days: int,
    start_date: str = "2024-01-01",
    ev_models_market_share: dict | None = None,
    driver_profiles_proportion: dict | None = None,
) -> Dict[str, Any]:
    override: Dict[str, Any] = {
        "fleet": {
            "fleet_size": fleet_size,
            "simulation_days": simulation_days,
            "start_date": start_date,
        }
    }
    if ev_models_market_share:
        override["ev_models_market_share"] = ev_models_market_share
    if driver_profiles_proportion:
        override["driver_profiles_proportion"] = driver_profiles_proportion

    gen = SyntheticEVGenerator(config_override=override)

    datasets = gen.generate_complete_dataset(num_days=simulation_days)
    paths = gen.save_datasets(datasets)

    # Normalize output
    out_dir = Path(paths.get("base_dir", "data/synthetic"))
    fleet_count = len(datasets.get("fleet_info", []))
    route_count = len(datasets.get("routes", []))
    charging_count = len(datasets.get("charging_sessions", []))

    return {
        "output_dir": str(out_dir),
        "fleet_count": fleet_count,
        "route_count": route_count,
        "charging_count": charging_count,
        "files": paths,
    }

