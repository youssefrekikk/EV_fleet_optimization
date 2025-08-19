from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import pandas as pd
import networkx as nx
import sys
import os
import argparse
import numpy as np
# Ensure project root is on sys.path so 'config' and 'src' are importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from config.ev_config import FLEET_CONFIG, CHARGING_CONFIG
from config.optimization_config import OPTIMIZATION_CONFIG
from src.data_generation.road_network_db import NetworkDatabase
from src.models.route_optimization.optimization import (
    SegmentEnergyRouter,
    SOCResourceRouter,
)
from src.data_generation.advanced_energy_model import AdvancedEVEnergyModel
from src.utils.logger import info, warning, debug, error, print_summary
from tqdm import tqdm
import numpy as np

# Charging infrastructure manager for real station data
from src.data_processing.openchargemap_api2 import ChargingInfrastructureManager

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
    validate_physics: bool = True,
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
    physics = {'physics_kwh': None, 'distance_km': length_m / 1000.0}
    if validate_physics:
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
    validate_physics: bool = True,
    trip_sample_frac: Optional[float] = None,
    trip_sample_n: Optional[int] = None,
    data_dir: Optional[str] = None,
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
    # Load combined charging infrastructure once for the dayc
    stations_df = None
    try:
        cim = ChargingInfrastructureManager()
        stations_df = cim.get_combined_infrastructure()
        if stations_df is not None and not stations_df.empty:
            info(f"Loaded {len(stations_df)} charging stations for SOC planning", "optimize_fleet")
        else:
            stations_df = None
    except Exception as e:
        warning(f"Could not load charging infrastructure: {e}", "optimize_fleet")

    def _stations_to_router_format(df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        if df is None or df.empty:
            return None
        cols = set(df.columns)
        lat_col = 'latitude' if 'latitude' in cols else None
        lon_col = 'longitude' if 'longitude' in cols else None
        if lat_col is None or lon_col is None:
            return None
        out: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            try:
                st = {
                    'station_id': r.get('station_id'),
                    'latitude': float(r[lat_col]),
                    'longitude': float(r[lon_col]),
                    'max_power_kw': float(r.get('max_power_kw', 50.0)),
                    'estimated_cost_per_kwh': float(r.get('estimated_cost_per_kwh', np.nan)) if pd.notna(r.get('estimated_cost_per_kwh', np.nan)) else None,
                    'cost_usd_per_kwh': float(r.get('cost_usd_per_kwh', np.nan)) if pd.notna(r.get('cost_usd_per_kwh', np.nan)) else None,
                }
                out.append(st)
            except Exception:
                continue
        return out if out else None

    stations_list = _stations_to_router_format(stations_df)

    results: List[Dict[str, Any]] = []
    trips_day = routes_df[routes_df['date'] == date.strftime('%Y-%m-%d')]
    # Optional per-day trip sampling
    if trip_sample_n is not None and trip_sample_n > 0 and len(trips_day) > trip_sample_n:
        trips_day = trips_day.sample(n=trip_sample_n, random_state=42)
    elif trip_sample_frac is not None and 0 < trip_sample_frac < 1:
        trips_day = trips_day.sample(frac=trip_sample_frac, random_state=42)
    # cache nearest node lookups to speed up
    node_cache: Dict[Tuple[float, float], Any] = {}
    def nearest_cached(lat: float, lon: float) -> Any:
        key = (round(float(lat), 5), round(float(lon), 5))
        if key in node_cache:
            return node_cache[key]
        node = db._find_nearest_node(float(lat), float(lon))
        node_cache[key] = node
        return node

    # Shared ML weight cache across trips on the same day to reuse edge predictions
    shared_ml_cache: Dict = {}
    for _, trip in tqdm(trips_day.iterrows(), total=len(trips_day), desc=f"Trips {date.strftime('%Y-%m-%d')}"):
        vehicle_id = trip['vehicle_id']
        ctx = contexts[vehicle_id]
        origin = nearest_cached(trip['origin_lat'], trip['origin_lon'])
        dest = nearest_cached(trip['destination_lat'], trip['destination_lon'])
        debug(f"Trip {vehicle_id} {trip['origin_lat']},{trip['origin_lon']} -> {trip['destination_lat']},{trip['destination_lon']} nodes=({origin}->{dest})", "optimize_fleet")
        depart_time = datetime.fromisoformat(trip.get('start_time', f"{date.strftime('%Y-%m-%d')}T08:00:00"))

        if not soc_planning:
            try:
                # Use shared cache to speed repeated edge evaluations
                path = seg_router.find_energy_optimal_path(
                    G, origin, dest, ctx['vehicle'], ctx['driver'], ctx['weather'], depart_time, algorithm=algo, shared_cache=shared_ml_cache
                )
            except Exception as e:
                # Try alternate algorithm before falling back to direct evaluation
                alt_algo = 'astar' if (algo or '').lower() == 'dijkstra' else 'dijkstra'
                warning(f"Routing failed ({algo}), trying alternate algorithm {alt_algo}: {e}", "optimize_fleet")
                try:
                    path = seg_router.find_energy_optimal_path(
                        G, origin, dest, ctx['vehicle'], ctx['driver'], ctx['weather'], depart_time, algorithm=alt_algo, shared_cache=shared_ml_cache
                    )
                except Exception as e2:
                    warning(f"Alternate routing failed ({alt_algo}), using fallback: {e2}", "optimize_fleet")
                    res = fallback_evaluate_direct(
                        trip['origin_lat'], trip['origin_lon'], trip['destination_lat'], trip['destination_lon'],
                        ctx['vehicle'], ctx['weather'], depart_time
                    )
                    res.update({'vehicle_id': vehicle_id, 'date': date.strftime('%Y-%m-%d'), 'trip_id': trip.get('trip_id')})
                    results.append(res)
                    continue

            # Evaluate metrics on this path using the same shared cache
            w = seg_router.make_weight_function(G, ctx['vehicle'], ctx['driver'], ctx['weather'], depart_time, shared_cache=shared_ml_cache)
            energy_kwh = 0.0
            travel_time_s = 0.0
            length_m = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edata = (G.edges[u, v, 0] if G.has_edge(u, v, 0) else list(G[u][v].values())[0]).copy()
                edata.setdefault('key', 0)
                energy_kwh += float(w(u, v, edata))
                travel_time_s += float(edata.get('travel_time', 0.0))
                length_m += float(edata.get('length', 0.0))
            physics = {'physics_kwh': None, 'distance_km': length_m / 1000.0}
            if validate_physics:
                physics = seg_router.validate_path_with_physics(G, path, ctx['vehicle'], ctx['weather'], depart_time)
            res = {
                'path': path,
                'ml_energy_kwh': energy_kwh,
                'travel_time_s': travel_time_s,
                'length_m': length_m,
                'physics_kwh': physics['physics_kwh'],
                'distance_km': physics['distance_km'],
            }
            res.update({'vehicle_id': vehicle_id, 'date': date.strftime('%Y-%m-%d'), 'trip_id': trip.get('trip_id')})
            results.append(res)
        else:
            # SOC planning version: plan with charging; for comparison, still compute ML energy/time on resulting path
            nominal_cap = float(ctx['vehicle'].get('battery_capacity', 60.0))
            init_soc = float(trip.get('initial_soc', 0.7))
            gamma = float(OPTIMIZATION_CONFIG.get('gamma_time_weight', 0.02))
            price_w = float(OPTIMIZATION_CONFIG.get('price_weight_kwh_per_usd', 0.0))
            soc_objective = str(OPTIMIZATION_CONFIG.get('soc_objective', 'energy'))
            alpha_usd_per_hour = float(OPTIMIZATION_CONFIG.get('alpha_usd_per_hour', 0.0))
            beta_kwh_to_usd = float(OPTIMIZATION_CONFIG.get('beta_kwh_to_usd', 0.0))
            # Determine terminal reserve based on planning mode
            planning_mode = str(OPTIMIZATION_CONFIG.get('planning_mode', 'myopic'))
            reserve_soc = float(OPTIMIZATION_CONFIG.get('reserve_soc', 0.15))
            reserve_kwh = float(OPTIMIZATION_CONFIG.get('reserve_kwh', 0.0))
            terminal_min_soc = 0.12
            if planning_mode in ('next_trip', 'rolling_horizon'):
                # If next trip exists, approximate next energy and convert to SOC reserve
                next_energy_kwh = None
                # Try to estimate from baseline rows if available
                try:
                    # Find this vehicle's trips for the day in routes_df and locate next by time
                    veh_trips = trips_day[trips_day['vehicle_id'] == vehicle_id]
                    # Use baseline total_consumption_kwh as proxy for each trip; fall back to reserve_kwh
                    next_rows = veh_trips[veh_trips['start_time'] > trip.get('start_time')]
                    if len(next_rows) > 0:
                        next_energy_kwh = float(next_rows.iloc[0].get('total_consumption_kwh', np.nan))
                except Exception:
                    next_energy_kwh = None
                eff_cap = float(ctx['vehicle'].get('battery_capacity', 60.0))
                if reserve_kwh > 0:
                    terminal_min_soc = max(terminal_min_soc, min(0.95, reserve_kwh / max(eff_cap, 1e-6)))
                elif next_energy_kwh is not None and not np.isnan(next_energy_kwh):
                    terminal_min_soc = max(terminal_min_soc, min(0.95, next_energy_kwh / max(eff_cap, 1e-6)))
                else:
                    terminal_min_soc = max(terminal_min_soc, reserve_soc)

            soc_plan = soc_router.soc_aware_shortest_path(
                G, origin, dest,
                vehicle_context=ctx['vehicle'],
                driver_context=ctx['driver'],
                weather_context=ctx['weather'],
                departure_time=depart_time,
                nominal_battery_capacity_kwh=nominal_cap,
                initial_soc=init_soc,
                min_soc=0.12,
                terminal_min_soc=terminal_min_soc,
                soc_step=0.02,
                gamma_time_weight=gamma,
                price_weight_kwh_per_usd=price_w,
                objective=soc_objective,
                alpha_usd_per_hour=alpha_usd_per_hour,
                beta_kwh_to_usd=beta_kwh_to_usd,
                charging_stations=stations_list,
            )
            path = soc_plan.get('path', [])
            if path:
                try:
                    path2 = seg_router.find_energy_optimal_path(
                        G, origin, dest, ctx['vehicle'], ctx['driver'], ctx['weather'], depart_time, algorithm=algo, shared_cache=shared_ml_cache
                    )
                    w2 = seg_router.make_weight_function(G, ctx['vehicle'], ctx['driver'], ctx['weather'], depart_time, shared_cache=shared_ml_cache)
                    e2 = 0.0; t2 = 0.0; lm2 = 0.0
                    for u, v in zip(path2[:-1], path2[1:]):
                        edata = (G.edges[u, v, 0] if G.has_edge(u, v, 0) else list(G[u][v].values())[0]).copy()
                        edata.setdefault('key', 0)
                        e2 += float(w2(u, v, edata))
                        t2 += float(edata.get('travel_time', 0.0))
                        lm2 += float(edata.get('length', 0.0))
                    phys2 = {'physics_kwh': None, 'distance_km': lm2 / 1000.0}
                    if validate_physics:
                        phys2 = seg_router.validate_path_with_physics(G, path2, ctx['vehicle'], ctx['weather'], depart_time)
                    reroute_eval = {
                        'path': path2,
                        'ml_energy_kwh': e2,
                        'travel_time_s': t2,
                        'length_m': lm2,
                        'physics_kwh': phys2['physics_kwh'],
                        'distance_km': phys2['distance_km'],
                    }
                except Exception as e:
                    warning(f"Routing failed after SOC plan, using fallback: {e}", "optimize_fleet")
                    reroute_eval = fallback_evaluate_direct(
                        trip['origin_lat'], trip['origin_lon'], trip['destination_lat'], trip['destination_lon'],
                        ctx['vehicle'], ctx['weather'], depart_time
                    )
                reroute_eval.update({'vehicle_id': vehicle_id, 'date': date.strftime('%Y-%m-%d'), 'trip_id': trip.get('trip_id')})
                reroute_eval['soc_metrics'] = soc_plan.get('metrics', {})
                # Carry explicit SOC cost in USD if present for KPI join later
                if isinstance(soc_plan.get('metrics'), dict) and ('cost_usd' in soc_plan['metrics']):
                    reroute_eval['soc_cost_usd'] = soc_plan['metrics']['cost_usd']
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
    validate_physics: bool = False,
    trip_sample_frac: Optional[float] = None,
    trip_sample_n: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    only_dates: Optional[List[str]] = None,
    max_days: Optional[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    # Load datasets
    routes = pd.read_csv(Path(data_dir) / 'routes.csv')
    fleet = pd.read_csv(Path(data_dir) / 'fleet_info.csv')
    weather = pd.read_csv(Path(data_dir) / 'weather.csv')

    # Determine which days to run
    all_days = sorted(routes['date'].unique())
    if only_dates:
        days = [d for d in all_days if d in set(only_dates)]
    else:
        days = all_days
        if start_date:
            days = [d for d in days if d >= start_date]
        if end_date:
            days = [d for d in days if d <= end_date]
    # Use config default if CLI not set
    if (max_days is None) and (OPTIMIZATION_CONFIG.get('fleet_eval_max_days') is not None):
        try:
            cfg_max_days = int(OPTIMIZATION_CONFIG.get('fleet_eval_max_days'))
            if cfg_max_days > 0:
                max_days = cfg_max_days
        except Exception:
            pass
    if max_days is not None and max_days > 0:
        days = days[:max_days]

    db = NetworkDatabase()
    G = db.load_or_create_network()

    # Determine per-day sampling defaults from config if not set
    if (trip_sample_frac is None) and (trip_sample_n is None):
        cfg_frac = OPTIMIZATION_CONFIG.get('fleet_eval_trip_sample_frac')
        if isinstance(cfg_frac, (float, int)) and 0 < float(cfg_frac) < 1:
            trip_sample_frac = float(cfg_frac)

    # Allow CLI to override SOC objective weights at runtime
    if algorithm is not None:
        OPTIMIZATION_CONFIG['route_optimization_algorithm'] = algorithm
    if hasattr(sys.modules[__name__], 'args'):
        pass

    daily_results: List[Dict[str, Any]] = []
    for d in tqdm(days, desc="Days"):
        date = datetime.fromisoformat(d)
        # Apply CLI overrides for SOC objective if provided
        if args is not None:
            if args.soc_objective is not None:
                OPTIMIZATION_CONFIG['soc_objective'] = args.soc_objective
            if args.alpha_usd_per_hour is not None:
                OPTIMIZATION_CONFIG['alpha_usd_per_hour'] = args.alpha_usd_per_hour
            if args.beta_kwh_to_usd is not None:
                OPTIMIZATION_CONFIG['beta_kwh_to_usd'] = args.beta_kwh_to_usd
            if args.planning_mode is not None:
                OPTIMIZATION_CONFIG['planning_mode'] = args.planning_mode
            if args.reserve_soc is not None:
                OPTIMIZATION_CONFIG['reserve_soc'] = args.reserve_soc
            if args.reserve_kwh is not None:
                OPTIMIZATION_CONFIG['reserve_kwh'] = args.reserve_kwh
            if args.horizon_trips is not None:
                OPTIMIZATION_CONFIG['horizon_trips'] = args.horizon_trips
            if args.horizon_hours is not None:
                OPTIMIZATION_CONFIG['horizon_hours'] = args.horizon_hours

        daily_results = []
        for d in tqdm(days, desc="Days"):
            date = datetime.fromisoformat(d)
            res = optimize_fleet_day(
                G, routes, fleet, weather, date,
                algorithm=algorithm,
                soc_planning=soc_planning,      # <--- this now works correctly
                validate_physics=validate_physics,
                trip_sample_frac=trip_sample_frac,
                trip_sample_n=trip_sample_n,
                data_dir=data_dir,
            )
            daily_results.append(res)
        res = optimize_fleet_day(
            G, routes, fleet, weather, date,
            algorithm=algorithm,
            soc_planning=soc_planning,
            validate_physics=validate_physics,
            trip_sample_frac=trip_sample_frac,
            trip_sample_n=trip_sample_n,
            data_dir=data_dir,
        )
        daily_results.append(res)

    # Aggregate and save per-trip optimized results
    all_rows: List[Dict[str, Any]] = []
    for day in daily_results:
        for trip in day['trips']:
            row = {
                'date': day['date'],
                'vehicle_id': trip.get('vehicle_id'),
                'trip_id': trip.get('trip_id'),
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

    optimized_df = pd.DataFrame(all_rows)
    results_csv_path = Path(output_dir) / 'fleet_optimization_results.csv'
    optimized_df.to_csv(results_csv_path, index=False)

    # KPI comparison against baseline (same trips)
    # Join on date + vehicle_id + trip_id
    baseline_cols = [
        'date', 'vehicle_id', 'trip_id',
        'total_consumption_kwh', 'total_distance_km', 'total_time_minutes'
    ]
    missing_baseline_cols = [c for c in baseline_cols if c not in routes.columns]
    kpi_summary = {}
    if not missing_baseline_cols and not optimized_df.empty:
        baseline_df = routes[baseline_cols].copy()
        merged = pd.merge(
            optimized_df,
            baseline_df,
            on=['date', 'vehicle_id', 'trip_id'],
            how='inner'
        )
        # Compute metrics (safe denominators)
        eps = 1e-6
        merged['baseline_energy_kwh'] = merged['total_consumption_kwh']
        merged['optimized_energy_kwh'] = merged['ml_energy_kwh']
        merged['energy_delta_kwh'] = merged['optimized_energy_kwh'] - merged['baseline_energy_kwh']
        merged['energy_delta_pct'] = merged['energy_delta_kwh'] / merged['baseline_energy_kwh'].where(merged['baseline_energy_kwh'].abs() > eps, np.nan) * 100.0

        merged['baseline_time_s'] = merged['total_time_minutes'] * 60.0
        merged['optimized_time_s'] = merged['travel_time_s']
        merged['time_delta_s'] = merged['optimized_time_s'] - merged['baseline_time_s']
        merged['time_delta_pct'] = merged['time_delta_s'] / merged['baseline_time_s'].where(merged['baseline_time_s'].abs() > eps, np.nan) * 100.0

        merged['baseline_distance_m'] = merged['total_distance_km'] * 1000.0
        merged['optimized_distance_m'] = merged['length_m']
        merged['distance_delta_m'] = merged['optimized_distance_m'] - merged['baseline_distance_m']
        merged['distance_delta_pct'] = merged['distance_delta_m'] / merged['baseline_distance_m'].where(merged['baseline_distance_m'].abs() > eps, np.nan) * 100.0

        # Estimated cost using base electricity cost (approximation)
        base_cost = float(CHARGING_CONFIG.get('base_electricity_cost', 0.15))
        merged['baseline_cost_est_usd'] = merged['baseline_energy_kwh'] * base_cost
        merged['optimized_cost_est_usd'] = merged['optimized_energy_kwh'] * base_cost
        merged['cost_delta_usd'] = merged['optimized_cost_est_usd'] - merged['baseline_cost_est_usd']
        merged['cost_delta_pct'] = merged['cost_delta_usd'] / merged['baseline_cost_est_usd'].where(merged['baseline_cost_est_usd'].abs() > eps, np.nan) * 100.0

        # Attach SOC USD cost if available from planning runs
        if 'soc_cost_usd' in optimized_df.columns:
            merged = pd.merge(
                merged,
                optimized_df[['date','vehicle_id','trip_id','soc_cost_usd']],
                on=['date','vehicle_id','trip_id'],
                how='left'
            )

        # Baseline true USD from charging_sessions.csv if present
        baseline_cost_usd_true = None
        sessions_path = Path(Path(routes_csv_path).parent if 'routes_csv_path' in locals() else data_dir) / 'charging_sessions.csv'
        try:
            if sessions_path.exists():
                sessions = pd.read_csv(sessions_path)
                # Derive date from start_time
                sessions['date'] = pd.to_datetime(sessions['start_time']).dt.date.astype(str)
                # Aggregate by date, vehicle; we lack trip_id linkage, so align at day+vehicle level
                agg = sessions.groupby(['date','vehicle_id'], as_index=False)['cost_usd'].sum().rename(columns={'cost_usd':'baseline_cost_usd_true'})
                merged = pd.merge(merged, agg, on=['date','vehicle_id'], how='left')
                baseline_cost_usd_true = 'baseline_cost_usd_true'
        except Exception:
            pass

        # Save per-trip KPI table
        kpi_csv = Path(output_dir) / 'fleet_optimization_kpis.csv'
        merged.to_csv(kpi_csv, index=False)

        # Build summary (ignore NaNs)
        def _mean(series: pd.Series) -> Optional[float]:
            s = series.dropna()
            return float(s.mean()) if len(s) else None

        def _count_valid(series: pd.Series) -> int:
            return int(series.dropna().shape[0])

        kpi_summary = {
            'total_trips_evaluated': int(len(merged)),
            'avg_baseline_energy_kwh': _mean(merged['baseline_energy_kwh']),
            'avg_optimized_energy_kwh': _mean(merged['optimized_energy_kwh']),
            'avg_energy_delta_kwh': _mean(merged['energy_delta_kwh']),
            'avg_energy_delta_pct': _mean(merged['energy_delta_pct']),
            'energy_pct_valid_n': _count_valid(merged['energy_delta_pct']),
            'avg_baseline_time_s': _mean(merged['baseline_time_s']),
            'avg_optimized_time_s': _mean(merged['optimized_time_s']),
            'avg_time_delta_s': _mean(merged['time_delta_s']),
            'avg_time_delta_pct': _mean(merged['time_delta_pct']),
            'time_pct_valid_n': _count_valid(merged['time_delta_pct']),
            'avg_baseline_cost_usd_est': _mean(merged['baseline_cost_est_usd']),
            'avg_optimized_cost_usd_est': _mean(merged['optimized_cost_est_usd']),
            'avg_cost_delta_usd_est': _mean(merged['cost_delta_usd']),
            'avg_cost_delta_pct_est': _mean(merged['cost_delta_pct']),
            'cost_pct_valid_n': _count_valid(merged['cost_delta_pct']),
            'cost_estimation_mode': 'flat_rate_per_kwh',
            'flat_rate_usd_per_kwh': base_cost,
        }

        # If we have SOC cost and/or true baseline USD, add USD KPIs
        if 'soc_cost_usd' in merged.columns:
            kpi_summary.update({
                'avg_optimized_cost_usd_soc': _mean(merged['soc_cost_usd']),
            })
        if baseline_cost_usd_true is not None:
            # Roll up baseline true USD to match per-trip granularity by day-vehicle average share
            # Provide day-vehicle total for transparency
            kpi_summary.update({
                'avg_baseline_cost_usd_true_dayveh': _mean(merged[baseline_cost_usd_true]),
            })

    # Also keep a compact summary for quick glance
    summary_json_path = Path(output_dir) / 'fleet_optimization_summary.json'
    with open(summary_json_path, 'w') as f:
        json.dump({
            'total_trips': int(len(optimized_df)),
            'avg_ml_energy_kwh': float(optimized_df['ml_energy_kwh'].mean()) if len(optimized_df) else None,
            'avg_physics_kwh': float(optimized_df['physics_kwh'].mean()) if ('physics_kwh' in optimized_df.columns and len(optimized_df)) else None,
            'avg_travel_time_s': float(optimized_df['travel_time_s'].mean()) if len(optimized_df) else None,
            'kpis': kpi_summary,
        }, f, indent=2)

    print_summary("FLEET OPTIMIZATION SUMMARY", {
        'trips': int(len(optimized_df)),
        'avg_ml_energy_kwh': float(optimized_df['ml_energy_kwh'].mean()) if len(optimized_df) else None,
        'avg_physics_kwh': float(optimized_df['physics_kwh'].mean()) if ('physics_kwh' in optimized_df.columns and len(optimized_df)) else None,
        'avg_travel_time_s': float(optimized_df['travel_time_s'].mean()) if len(optimized_df) else None,
        'results_csv': str(results_csv_path),
        'summary_json': str(summary_json_path),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize fleet routes with ML energy-aware routing")
    parser.add_argument('--data-dir', type=str, default="data/synthetic")
    parser.add_argument('--output-dir', type=str, default="data/analysis/optimized")
    parser.add_argument('--algorithm', type=str, choices=['dijkstra', 'astar', 'custom_astar'], default=None, 
                       help='Routing algorithm: dijkstra (simple), astar (NetworkX), or custom_astar (optimized)')
    parser.add_argument('--soc-planning', action='store_true', help='Enable SOC-aware routing')
    parser.add_argument('--no-physics', action='store_true', help='Disable per-trip physics validation')
    parser.add_argument('--sample-frac', type=float, default=None, help='Sample fraction of trips per day (0-1)')
    parser.add_argument('--sample-n', type=int, default=None, help='Sample N trips per day')
    parser.add_argument('--start-date', type=str, default=None, help='Inclusive start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='Inclusive end date YYYY-MM-DD')
    parser.add_argument('--dates', type=str, nargs='*', default=None, help='Specific dates YYYY-MM-DD to run')
    parser.add_argument('--max-days', type=int, default=None, help='Limit number of days to run (overrides config)')
    parser.add_argument('--soc-objective', type=str, choices=['energy','cost','time','weighted'], default=None)
    parser.add_argument('--alpha-usd-per-hour', type=float, default=None)
    parser.add_argument('--beta-kwh-to-usd', type=float, default=None)
    parser.add_argument('--planning-mode', type=str, choices=['myopic','next_trip','rolling_horizon'], default=None)
    parser.add_argument('--reserve-soc', type=float, default=None)
    parser.add_argument('--reserve-kwh', type=float, default=None)
    parser.add_argument('--horizon-trips', type=int, default=None)
    parser.add_argument('--horizon-hours', type=float, default=None)

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        algorithm=args.algorithm,
        soc_planning=args.soc_planning,
        validate_physics=not args.no_physics,
        trip_sample_frac=args.sample_frac,
        trip_sample_n=args.sample_n,
        start_date=args.start_date,
        end_date=args.end_date,
        only_dates=args.dates,
        max_days=args.max_days,
    )


