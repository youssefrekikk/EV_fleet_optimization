from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def get_price_per_kwh(
    station: Optional[Dict[str, Any]],
    when: datetime,
    charging_config: Dict[str, Any],
    is_home: bool = False,
) -> float:
    """
    Return USD/kWh for a charge starting at 'when', reproducing the exact
    time-of-use logic used by the synthetic data generator.

    Rules (matching synthetic_ev_generator.py):
    - Home: flat base rate from charging_config['base_electricity_cost']
    - Public:
      base = station['estimated_cost_per_kwh'] or station['cost_usd_per_kwh'] (fallback to 0.30)
      If within any peak interval defined in charging_config['peak_hours'],
      multiply by charging_config['peak_pricing_multiplier']
    """
    # Home charging flat rate
    if is_home:
        return float(charging_config.get('base_electricity_cost', 0.15))

    # Public station base price
    base_price = 0.30
    if station is not None:
        if 'estimated_cost_per_kwh' in station and station['estimated_cost_per_kwh'] is not None:
            base_price = float(station['estimated_cost_per_kwh'])
        elif 'cost_usd_per_kwh' in station and station['cost_usd_per_kwh'] is not None:
            base_price = float(station['cost_usd_per_kwh'])

    # Peak pricing
    peak_hours = charging_config.get('peak_hours', [])
    is_peak = False
    try:
        hr = int(when.hour)
        is_peak = any(start <= hr <= end for start, end in peak_hours)
    except Exception:
        is_peak = False

    if is_peak:
        multiplier = float(charging_config.get('peak_pricing_multiplier', 1.5))
        return base_price * multiplier

    return base_price


