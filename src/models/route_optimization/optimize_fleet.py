from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import pandas as pd
import networkx as nx
import sys
import os
# Ensure project root is on sys.path so 'config' and 'src' are importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from config.ev_config import FLEET_CONFIG
from config.optimization_config import OPTIMIZATION_CONFIG
from src.data_generation.road_network_db import NetworkDatabase
from src.models.route_optimization.optimization import (
    SegmentEnergyRouter,
    SOCResourceRouter,
)
from src.data_generation.advanced_energy_model import AdvancedEVEnergyModel
from src.utils.logger import info, warning, debug, error, print_summary


def load_contexts_for_day(date: datetime, fleet_info: pd.DataFrame, weather: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build per-vehicle context dictionaries (vehicle, driver, weather) for a given date.
    Expects fleet_info columns to include: vehicle_id, model, battery_capacity, driver_profile, driving_style, driver_personality
    Weather DataFrame should include a single row per date with temperature, wind_speed_kmh, is_raining.
    """
    contexts: Dict[str, Dict[str, Any]] = {}
    day_wx = weather.loc[weather['date'] == date.strftime('%Y-%m-%d')].to_dict(orient='records')
    wx = day_wx[0] if day_wx else {}
    weather_ctx = {
        'temperature': float(wx.get('temperature', 15.0)),
        'wind_speed_kmh': float(wx.get('wind_speed_kmh', 10.0)),
        'is_raining': bool(wx.get('is_raining', False)),
        'humidity': float(wx.get('humidity', 0.6)) if 'humidity' in wx else 0.6,
    }
    for _, row in fleet_info.iterrows():
        vehicle_id = row['vehicle_id']
        vehicle_ctx = {
            'vehicle_id': vehicle_id,
            'model': row.get('model') or row.get('vehicle_model', 'tesla_model_3'),
            'battery_capacity': float(row.get('battery_capacity', 60.0)),
            'max_charging_speed': float(row.get('max_charging_speed', 100.0)),
            'driver_profile': row.get('driver_profile', 'commuter'),
            'driving_style': row.get('driving_style', 'normal'),
            'driver_personality': row.get('driver_personality', 'optimizer')
        }
        driver_ctx = {
            'driver_profile': vehicle_ctx['driver_profile'],
            'driving_style': vehicle_ctx['driving_style'],
            'driver_personality': vehicle_ctx['driver_personality'],
        }
        contexts[vehicle_id] = {
            'vehicle': vehicle_ctx,
            'driver': driver_ctx,
            'weather': weather_ctx,
        }
    return contexts


def route_and_evaluate(
    G: nx.MultiDiGraph,
    router: SegmentEnergyRouter,
    origin: Any,
    dest: Any,
    vehicle: Dict[str, Any],
    driver: Dict[str, Any],
    weather: Dict[str, Any],
    depart_time: datetime,
    algorithm: str,
) -> Dict[str, Any]:
    # Try configured algorithm; on failure try the other one once before bubbling up
    try:
        path = router.find_energy_optimal_path(
            G, origin, dest, vehicle, driver, weather, depart_time, algorithm=algorithm
        )
    except Exception:
        alt_algo = 'astar' if (algorithm or '').lower() == 'dijkstra' else 'dijkstra'
        path = router.find_energy_optimal_path(
            G, origin, dest, vehicle, driver, weather, depart_time, algorithm=alt_algo
        )
    # ML energy accumulation
    w = router.make_weight_function(G, vehicle, driver, weather, depart_time)
    energy_kwh = 0.0
    travel_time_s = 0.0
    length_m = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edata = (G.edges[u, v, 0] if G.has_edge(u, v, 0) else list(G[u][v].values())[0]).copy()
        edata.setdefault('key', 0)
        energy_kwh += float(w(u, v, edata))
        travel_time_s += float(edata.get('travel_time', 0.0))
        length_m += float(edata.get('length', 0.0))
    # Physics validation
    physics = router.validate_path_with_physics(G, path, vehicle, weather, depart_time)
    return {
        'path': path,
        'ml_energy_kwh': energy_kwh,
        'travel_time_s': travel_time_s,
        'length_m': length_m,
        'physics_kwh': physics['physics_kwh'],
        'distance_km': physics['distance_km'],
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fallback_evaluate_direct(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    vehicle: Dict[str, Any],
    weather: Dict[str, Any],
    depart_time: datetime,
) -> Dict[str, Any]:
    # Straight-line distance with detour factor like generator
    distance_km = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    if distance_km <= 0:
        distance_km = 0.1
    detour_factor = 1.3
    route_km = distance_km * detour_factor
    avg_speed_kmh = 40.0
    total_time_s = (route_km / max(avg_speed_kmh, 1e-3)) * 3600.0
    # Build simple GPS trace with 60 points
    num_points = 60
    gps_trace = []
    current_time = depart_time
    for i in range(num_points):
        t = i / (num_points - 1)
        lat = origin_lat + (dest_lat - origin_lat) * t
        lon = origin_lon + (dest_lon - origin_lon) * t
        gps_trace.append({
            'timestamp': current_time.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'speed_kmh': avg_speed_kmh,
            'elevation_m': 50.0,
            'heading': 0.0,
        })
        if i < num_points - 1:
            current_time += timedelta(seconds=total_time_s / (num_points - 1))

    # Ensure vehicle has battery capacity
    vehicle_for_physics = dict(vehicle)
    if 'battery_capacity' not in vehicle_for_physics:
        vehicle_for_physics['battery_capacity'] = float(vehicle.get('battery_capacity', 60.0))

    physics = AdvancedEVEnergyModel()
    res = physics.calculate_energy_consumption(gps_trace, vehicle_for_physics, weather)
    return {
        'path': [],
        'ml_energy_kwh': None,
        'travel_time_s': total_time_s,
        'length_m': route_km * 1000.0,
        'physics_kwh': float(res.get('total_consumption_kwh', 0.0)),
        'distance_km': float(res.get('total_distance_km', route_km)),
        'fallback_used': True,
    }


def optimize_fleet_day(
    G: nx.MultiDiGraph,
    routes_df: pd.DataFrame,
    fleet_info: pd.DataFrame,
    weather_df: pd.DataFrame,
    date: datetime,
    algorithm: Optional[str] = None,
    soc_planning: bool = False,
) -> Dict[str, Any]:
    """
    Re-run routing for all trips on a given date using ML energy-aware routing (and optional SOC planning),
    then compare against original route metrics.
    Expects routes_df with: vehicle_id, origin_lat, origin_lon, destination_lat, destination_lon, start_time
    """
    contexts = load_contexts_for_day(date, fleet_info, weather_df)
    info("Loading/Binding road network for fleet-day optimization", "optimize_fleet")
    db = NetworkDatabase()
    # Ensure this db instance has the loaded graph bound before nearest-node calls
    G = db.load_or_create_network() if G is None else G
    db.network = G

    seg_router = SegmentEnergyRouter()
    soc_router = SOCResourceRouter()
    algo = (algorithm or OPTIMIZATION_CONFIG.get('route_optimization_algorithm', 'dijkstra')).lower()

    results: List[Dict[str, Any]] = []
    for _, trip in routes_df[routes_df['date'] == date.strftime('%Y-%m-%d')].iterrows():
        vehicle_id = trip['vehicle_id']
        ctx = contexts[vehicle_id]
        origin = db._find_nearest_node(trip['origin_lat'], trip['origin_lon'])
        dest = db._find_nearest_node(trip['destination_lat'], trip['destination_lon'])
        debug(f"Trip {vehicle_id} {trip['origin_lat']},{trip['origin_lon']} -> {trip['destination_lat']},{trip['destination_lon']} nodes=({origin}->{dest})", "optimize_fleet")
        depart_time = datetime.fromisoformat(trip.get('start_time', f"{date.strftime('%Y-%m-%d')}T08:00:00"))

        if not soc_planning:
            try:
                res = route_and_evaluate(
                    G, seg_router, origin, dest, ctx['vehicle'], ctx['driver'], ctx['weather'], depart_time, algo
                )
            except Exception as e:
                warning(f"Routing failed ({algo}), trying fallback: {e}", "optimize_fleet")
                res = fallback_evaluate_direct(
                    trip['origin_lat'], trip['origin_lon'], trip['destination_lat'], trip['destination_lon'],
                    ctx['vehicle'], ctx['weather'], depart_time
                )
            res.update({'vehicle_id': vehicle_id, 'date': date.strftime('%Y-%m-%d')})
            results.append(res)
        else:
            # SOC planning version: plan with charging; for comparison, still compute ML energy/time on resulting path
            nominal_cap = float(ctx['vehicle'].get('battery_capacity', 60.0))
            init_soc = float(trip.get('initial_soc', 0.7))
            gamma = float(OPTIMIZATION_CONFIG.get('gamma_time_weight', 0.02))
            price_w = float(OPTIMIZATION_CONFIG.get('price_weight_kwh_per_usd', 0.0))
            soc_plan = soc_router.soc_aware_shortest_path(
                G, origin, dest,
                vehicle_context=ctx['vehicle'],
                driver_context=ctx['driver'],
                weather_context=ctx['weather'],
                departure_time=depart_time,
                nominal_battery_capacity_kwh=nominal_cap,
                initial_soc=init_soc,
                min_soc=0.12,
                soc_step=0.02,
                gamma_time_weight=gamma,
                price_weight_kwh_per_usd=price_w,
                charging_stations=None,
            )
            path = soc_plan.get('path', [])
            if path:
                try:
                    reroute_eval = route_and_evaluate(
                        G, seg_router, origin, dest, ctx['vehicle'], ctx['driver'], ctx['weather'], depart_time, algo
                    )
                except Exception as e:
                    warning(f"Routing failed after SOC plan, using fallback: {e}", "optimize_fleet")
                    reroute_eval = fallback_evaluate_direct(
                        trip['origin_lat'], trip['origin_lon'], trip['destination_lat'], trip['destination_lon'],
                        ctx['vehicle'], ctx['weather'], depart_time
                    )
                reroute_eval.update({'vehicle_id': vehicle_id, 'date': date.strftime('%Y-%m-%d')})
                reroute_eval['soc_metrics'] = soc_plan.get('metrics', {})
                results.append(reroute_eval)

    return {
        'date': date.strftime('%Y-%m-%d'),
        'trips': results,
    }


def main(
    data_dir: str = "data/synthetic",
    output_dir: str = "data/analysis/optimized",
    algorithm: Optional[str] = None,
    soc_planning: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    # Load datasets
    routes = pd.read_csv(Path(data_dir) / 'routes.csv')
    fleet = pd.read_csv(Path(data_dir) / 'fleet_info.csv')
    weather = pd.read_csv(Path(data_dir) / 'weather.csv')

    days = sorted(routes['date'].unique())
    db = NetworkDatabase()
    G = db.load_or_create_network()

    daily_results: List[Dict[str, Any]] = []
    for d in days:
        date = datetime.fromisoformat(d)
        res = optimize_fleet_day(G, routes, fleet, weather, date, algorithm=algorithm, soc_planning=soc_planning)
        daily_results.append(res)

    # Aggregate and save
    all_rows: List[Dict[str, Any]] = []
    for day in daily_results:
        for trip in day['trips']:
            row = {
                'date': day['date'],
                'vehicle_id': trip.get('vehicle_id'),
                'ml_energy_kwh': trip.get('ml_energy_kwh'),
                'physics_kwh': trip.get('physics_kwh'),
                'travel_time_s': trip.get('travel_time_s'),
                'length_m': trip.get('length_m'),
            }
            # add SOC metrics if present
            socm = trip.get('soc_metrics') or {}
            row.update({
                'soc_energy_kwh': socm.get('energy_kwh'),
                'soc_travel_time_s': socm.get('travel_time_s'),
                'soc_num_charges': socm.get('num_charges'),
            })
            all_rows.append(row)

    df_out = pd.DataFrame(all_rows)
    csv_path = Path(output_dir) / 'fleet_optimization_results.csv'
    df_out.to_csv(csv_path, index=False)

    with open(Path(output_dir) / 'fleet_optimization_summary.json', 'w') as f:
        json.dump({
            'total_trips': len(all_rows),
            'avg_ml_energy_kwh': float(df_out['ml_energy_kwh'].mean()) if len(df_out) else None,
            'avg_physics_kwh': float(df_out['physics_kwh'].mean()) if len(df_out) else None,
            'avg_travel_time_s': float(df_out['travel_time_s'].mean()) if len(df_out) else None,
        }, f, indent=2)

    print_summary("FLEET OPTIMIZATION SUMMARY", {
        'trips': len(all_rows),
        'avg_ml_energy_kwh': float(df_out['ml_energy_kwh'].mean()) if len(df_out) else None,
        'avg_physics_kwh': float(df_out['physics_kwh'].mean()) if len(df_out) else None,
        'avg_travel_time_s': float(df_out['travel_time_s'].mean()) if len(df_out) else None,
        'out_csv': str(csv_path),
    })


if __name__ == "__main__":
    main()


