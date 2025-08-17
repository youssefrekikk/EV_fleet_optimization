from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import pandas as pd

from src.models.route_optimization.optimize_fleet import optimize_fleet_day


def run_optimization(
    routes_csv: str,
    fleet_csv: str,
    weather_csv: str,
    date: str,
    algorithm: Optional[str] = None,
    soc_planning: bool = False,
) -> Dict[str, Any]:
    routes_df = pd.read_csv(routes_csv)
    fleet_df = pd.read_csv(fleet_csv)
    weather_df = pd.read_csv(weather_csv)
    dt = datetime.fromisoformat(date)

    # The optimize_fleet_day function expects a prebuilt network and router in some contexts;
    # here we assume its internal fallbacks. We can enhance later to pass a graph.
    try:
        result = optimize_fleet_day(
            G=None,
            routes_df=routes_df,
            fleet_info=fleet_df,
            weather_df=weather_df,
            date=dt,
            algorithm=algorithm,
            soc_planning=soc_planning,
        )
    except asyncio.CancelledError:
        # If the task is cancelled, raise an exception to interrupt the optimization algorithm
        raise Exception("Optimization cancelled")
    
    return result or {}

