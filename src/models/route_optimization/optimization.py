from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.models.consumption_prediction.consumption_model_v2 import (
    SegmentEnergyPredictor,
)
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value
from config.ev_config import EV_MODELS, CHARGING_CONFIG
from config.optimization_config import OPTIMIZATION_CONFIG
from config.physics_constants import PHYSICS_CONSTANTS
from src.data_generation.advanced_energy_model import AdvancedEVEnergyModel
from src.utils.logger import info, warning, debug, error
from src.utils.tariff import get_price_per_kwh


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _safe_get_edge_attr(edata: Dict[str, Any], key: str, default: float) -> float:
    value_ = edata.get(key, default)
    try:
        return float(value_)
    except Exception:
        return float(default)


def _edge_distance_m(G: nx.MultiDiGraph, u: Any, v: Any, edata: Dict[str, Any]) -> float:
    # Prefer stored length
    length = edata.get("length")
    if isinstance(length, (int, float)) and length > 0:
        return float(length)
    # Fallback: geodesic-ish straight-line using node coordinates if present
    try:
        uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
        vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]
        # Approx meters using haversine approximation
        lat1, lon1, lat2, lon2 = map(np.radians, [uy, ux, vy, vx])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371000.0 * c
    except Exception:
        return 0.0


class EnergyWeightFunction:
    """
    Callable weight function for NetworkX that returns ML-predicted energy (kWh) per edge.

    Caches computed weights per edge for the given routing context to accelerate repeated queries.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        predictor: SegmentEnergyPredictor,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
        shared_cache: Optional[Dict[Tuple[Any, Any, int, Tuple], float]] = None,
    ) -> None:
        self.G = graph
        self.predictor = predictor
        self.vehicle = vehicle_context or {}
        self.driver = driver_context or {}
        self.weather = weather_context or {}
        self.departure_time = departure_time
        self.cache: Dict[Tuple[Any, Any, int, Tuple], float] = {}
        self.shared_cache = shared_cache

        # Precompute simple context fields used for caching keys
        self._context_key = (
            str(self.vehicle.get("model", "unknown")),
            str(self.driver.get("driving_style", self.vehicle.get("driving_style", "normal"))),
            str(_season_from_month(self.departure_time.month)),
            int(round(float(self.weather.get("temperature", 15)) / 5.0)),  # coarse bucket
            int(bool(self.weather.get("is_raining", False))),
            int(self.departure_time.hour),
        )

    def __call__(self, u: Any, v: Any, edata: Dict[str, Any]) -> float:
        # MultiDiGraph may have parallel edges distinguished by 'key'. Use 0 by default.
        key = edata.get("key", 0)
        cache_key = (u, v, key, self._context_key)
        if self.shared_cache is not None:
            if cache_key in self.shared_cache:
                return self.shared_cache[cache_key]
        if cache_key in self.cache:
            return self.cache[cache_key]

        distance_m = _edge_distance_m(self.G, u, v, edata)
        speed_kph = _safe_get_edge_attr(edata, "speed_kph", 40.0)
        travel_time_s = _safe_get_edge_attr(edata, "travel_time", distance_m / max(speed_kph * 1000 / 3600, 1e-6))

        start_time = self.departure_time
        end_time = start_time + timedelta(seconds=travel_time_s)

        # Minimal row with context; engineer_features will handle dates and missing fields
        row = {
            "distance_m": distance_m,
            "start_time": start_time,
            "end_time": end_time,
            "start_speed_kmh": speed_kph,
            "end_speed_kmh": speed_kph,
            # Vehicle/driver
            "model": self.vehicle.get("model"),
            "driver_profile": self.driver.get("driver_profile", self.vehicle.get("driver_profile")),
            "driving_style": self.driver.get("driving_style", self.vehicle.get("driving_style", "normal")),
            "driver_personality": self.driver.get("driver_personality", self.vehicle.get("driver_personality")),
            # Weather
            "weather_temp_c": self.weather.get("temperature"),
            "weather_wind_kmh": self.weather.get("wind_speed_kmh"),
            "weather_is_raining": self.weather.get("is_raining"),
            # Season
            "season": _season_from_month(start_time.month),
        }

        try:
            df = pd.DataFrame([row])
            df_fe = self.predictor.engineer_features(df.copy())
            # Select only known feature columns if available
            if self.predictor.feature_columns is not None:
                X = df_fe[self.predictor.feature_columns]
            else:
                # Fallback: numeric-only
                X = df_fe.select_dtypes(include=[np.number])
            pred_kwh = self.predictor.predict(
                X,
                model_name=None,
                distance_m=np.array([max(distance_m, 0.0)]),
            )[0]
            weight = max(float(pred_kwh), 1e-9)
        except Exception:
            # Robust fallback: proportional to distance with mild penalty for low speed
            base_kwh_per_km = 0.18  # conservative generic value
            speed_penalty = 1.0 + (40.0 / max(speed_kph, 1.0) - 1.0) * 0.2
            weight = max((distance_m / 1000.0) * base_kwh_per_km * speed_penalty, 1e-9)

        # Save to caches
        self.cache[cache_key] = weight
        if self.shared_cache is not None:
            self.shared_cache[cache_key] = weight
        return weight


class SegmentEnergyRouter:
    """Utility to compute energy-optimal routes using the trained segment-level predictor."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.predictor = SegmentEnergyPredictor()

        # Default to the packaged model path if not provided
        if model_path is None:
            model_path = Path(__file__).parent.parent / "consumption_prediction" / "segment_energy_model.pkl"
        self.model_path = Path(model_path)

        if self.model_path.exists():
            self.predictor.load_model(str(self.model_path))
        else:
            # Model not found; allow running with runtime-fitted predictor if user sets it later
            pass

    def make_weight_function(
        self,
        graph: nx.MultiDiGraph,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
        shared_cache: Optional[Dict[Tuple[Any, Any, int, Tuple], float]] = None,
    ) -> EnergyWeightFunction:
        return EnergyWeightFunction(
            graph=graph,
            predictor=self.predictor,
            vehicle_context=vehicle_context,
            driver_context=driver_context,
            weather_context=weather_context,
            departure_time=departure_time,
            shared_cache=shared_cache,
        )

    @staticmethod
    def _sum_edge_attribute(
        G: nx.MultiDiGraph, path: List[Any], attr: str, default: float = 0.0
    ) -> float:
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            # Prefer first parallel edge
            try:
                edata = G.edges[u, v, 0]
            except Exception:
                # Fallback to any available edge data
                edata = list(G[u][v].values())[0]
            total += float(edata.get(attr, default))
        return total

    def shortest_path_by_energy(
        self,
        G: nx.MultiDiGraph,
        origin: Any,
        destination: Any,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
        shared_cache: Optional[Dict[Tuple[Any, Any, int, Tuple], float]] = None,
    ) -> List[Any]:
        info(f"Routing (Dijkstra) from {origin} to {destination}", "optimization")
        weight_fn = self.make_weight_function(
            G, vehicle_context, driver_context, weather_context, departure_time, shared_cache=shared_cache
        )
        path = nx.shortest_path(G, origin, destination, weight=weight_fn)
        info(f"Dijkstra path length: {len(path)} nodes", "optimization")
        return path

    def benchmark_against_time_and_distance(
        self,
        G: nx.MultiDiGraph,
        origin: Any,
        destination: Any,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
    ) -> Dict[str, Any]:
        # Energy-optimal
        weight_fn = self.make_weight_function(
            G, vehicle_context, driver_context, weather_context, departure_time
        )
        path_energy = nx.shortest_path(G, origin, destination, weight=weight_fn)

        # Time-optimal (existing baseline)
        path_time = nx.shortest_path(G, origin, destination, weight="travel_time")

        # Distance-optimal
        path_distance = nx.shortest_path(G, origin, destination, weight="length")

        # Compute totals
        total_length_energy = self._sum_edge_attribute(G, path_energy, "length")
        total_length_time = self._sum_edge_attribute(G, path_time, "length")
        total_length_distance = self._sum_edge_attribute(G, path_distance, "length")

        total_time_energy = self._sum_edge_attribute(G, path_energy, "travel_time")
        total_time_time = self._sum_edge_attribute(G, path_time, "travel_time")
        total_time_distance = self._sum_edge_attribute(G, path_distance, "travel_time")

        # For energy totals, reuse the same weight function to avoid re-implementing per-edge energy accumulation
        def accumulate_energy(path: List[Any]) -> float:
            total_kwh = 0.0
            for u, v in zip(path[:-1], path[1:]):
                try:
                    edata = G.edges[u, v, 0].copy()
                    edata["key"] = 0
                except Exception:
                    edata = list(G[u][v].values())[0].copy()
                    edata.setdefault("key", 0)
                total_kwh += float(weight_fn(u, v, edata))
            return total_kwh

        energy_kwh_energy = accumulate_energy(path_energy)
        energy_kwh_time = accumulate_energy(path_time)
        energy_kwh_distance = accumulate_energy(path_distance)

        return {
            "paths": {
                "energy": path_energy,
                "time": path_time,
                "distance": path_distance,
            },
            "metrics": {
                "energy": {
                    "energy_kwh": energy_kwh_energy,
                    "length_m": total_length_energy,
                    "travel_time_s": total_time_energy,
                },
                "time": {
                    "energy_kwh": energy_kwh_time,
                    "length_m": total_length_time,
                    "travel_time_s": total_time_time,
                },
                "distance": {
                    "energy_kwh": energy_kwh_distance,
                    "length_m": total_length_distance,
                    "travel_time_s": total_time_distance,
                },
            },
        }

    # --- A* energy-aware routing ---
    @staticmethod
    def _node_distance_m(G: nx.MultiDiGraph, u: Any, v: Any) -> float:
        try:
            uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
            vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]
            lat1, lon1, lat2, lon2 = map(np.radians, [uy, ux, vy, vx])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return 6371000.0 * c
        except Exception:
            return 0.0

    def a_star_path_by_energy(
        self,
        G: nx.MultiDiGraph,
        origin: Any,
        destination: Any,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
        heuristic_kwh_per_km: Optional[float] = None,
        shared_cache: Optional[Dict[Tuple[Any, Any, int, Tuple], float]] = None,
    ) -> List[Any]:
        """
        Energy-aware A* with an admissible heuristic: optimistic kWh/km times straight-line distance.
        heuristic_kwh_per_km should be a lower-bound estimate of consumption per km.
        """
        info(f"Routing (A*) from {origin} to {destination}", "optimization")
        weight_fn = self.make_weight_function(
            G, vehicle_context, driver_context, weather_context, departure_time, shared_cache=shared_cache
        )

        def h(u: Any, v: Any) -> float:
            d_m = self._node_distance_m(G, u, v)
            if heuristic_kwh_per_km is None:
                # Physics-informed lower bound using rolling resistance only
                try:
                    model_name = vehicle_context.get("model")
                    specs = EV_MODELS.get(model_name, {})
                    mass_kg = float(specs.get("weight", 1800))
                except Exception:
                    mass_kg = 1800.0
                g = float(PHYSICS_CONSTANTS.get("gravity", 9.81))
                cr = float(PHYSICS_CONSTANTS.get("rolling_resistance", 0.012))
                force_n = cr * mass_kg * g
                kwh_per_km = max(0.005, (force_n * 1000.0) / 3.6e6)
            else:
                kwh_per_km = max(0.0, float(heuristic_kwh_per_km))
            return (d_m / 1000.0) * kwh_per_km

        path = nx.astar_path(G, origin, destination, heuristic=h, weight=weight_fn)
        info(f"A* path length: {len(path)} nodes", "optimization")
        return path

    def find_energy_optimal_path(
        self,
        G: nx.MultiDiGraph,
        origin: Any,
        destination: Any,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
        algorithm: Optional[str] = None,
        shared_cache: Optional[Dict[Tuple[Any, Any, int, Tuple], float]] = None,
    ) -> List[Any]:
        algo = (algorithm or OPTIMIZATION_CONFIG.get("route_optimization_algorithm", "dijkstra")).lower()
        info(f"find_energy_optimal_path using {algo}", "optimization")
        if algo == "astar":
            return self.a_star_path_by_energy(G, origin, destination, vehicle_context, driver_context, weather_context, departure_time, shared_cache=shared_cache)
        return self.shortest_path_by_energy(G, origin, destination, vehicle_context, driver_context, weather_context, departure_time, shared_cache=shared_cache)

    def validate_path_with_physics(
        self,
        G: nx.MultiDiGraph,
        path: List[Any],
        vehicle_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
    ) -> Dict[str, float]:
        coords: List[Tuple[float, float]] = []
        speeds_kmh: List[float] = []
        for idx, (u, v) in enumerate(zip(path[:-1], path[1:])):
            nu, nv = G.nodes[u], G.nodes[v]
            if idx == 0:
                coords.append((nu.get("y"), nu.get("x")))
            coords.append((nv.get("y"), nv.get("x")))
            try:
                edata = G.edges[u, v, 0]
            except Exception:
                edata = list(G[u][v].values())[0]
            speed = float(edata.get("speed_kph", 40.0))
            if idx == 0:
                speeds_kmh.append(speed)
            speeds_kmh.append(speed)

        elevations = [50.0] * len(coords)
        gps_trace = []
        current_time = departure_time
        for i, (coord, speed_kmh) in enumerate(zip(coords, speeds_kmh)):
            gps_trace.append({
                "timestamp": current_time.isoformat(),
                "latitude": coord[0],
                "longitude": coord[1],
                "speed_kmh": speed_kmh,
                "elevation_m": elevations[i],
                "heading": 0.0,
            })
            if i < len(coords) - 1:
                d_m = _edge_distance_m(G, path[i], path[i + 1], {})
                dt_s = (d_m / 1000.0) / max(speed_kmh / 3600.0, 1e-6)
                current_time += timedelta(seconds=dt_s)

        vehicle_for_physics = dict(vehicle_context)
        model_name = vehicle_for_physics.get("model")
        if "battery_capacity" not in vehicle_for_physics and model_name in EV_MODELS:
            vehicle_for_physics["battery_capacity"] = EV_MODELS[model_name]["battery_capacity"]

        physics = AdvancedEVEnergyModel()
        res = physics.calculate_energy_consumption(gps_trace, vehicle_for_physics, weather_context)
        debug(f"Physics validation: {res}", "optimization")
        return {"physics_kwh": float(res.get("total_consumption_kwh", 0.0)), "distance_km": float(res.get("total_distance_km", 0.0))}


# --- SOC-aware routing and LP optimization ---

def _effective_battery_capacity_kwh(
    ambient_temp_c: float, nominal_capacity_kwh: float
) -> float:
    try:
        from config.physics_constants import PHYSICS_CONSTANTS
    except Exception:
        PHYSICS_CONSTANTS = {
            "reference_temp_k": 293.15,
            "temp_capacity_alpha": 0.05,
        }
    temp_k = ambient_temp_c + 273.15
    ref_k = PHYSICS_CONSTANTS.get("reference_temp_k", 293.15)
    alpha = PHYSICS_CONSTANTS.get("temp_capacity_alpha", 0.05)
    capacity_factor = (temp_k / ref_k) ** alpha
    return float(nominal_capacity_kwh) * capacity_factor


def _nearest_node_linear(G: nx.MultiDiGraph, lat: float, lon: float) -> Any:
    best = None
    best_d2 = float("inf")
    lat_r, lon_r = np.radians([lat, lon])
    for node, data in G.nodes(data=True):
        if "y" not in data or "x" not in data:
            continue
        y, x = data["y"], data["x"]
        y_r, x_r = np.radians([y, x])
        dlat = y_r - lat_r
        dlon = x_r - lon_r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(y_r) * np.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = 6371000.0 * c
        if d < best_d2:
            best_d2 = d
            best = node
    return best


class SOCResourceRouter:
    def __init__(self, predictor: Optional[SegmentEnergyPredictor] = None) -> None:
        self.predictor = predictor or SegmentEnergyPredictor()

    def make_weight_function(
        self,
        graph: nx.MultiDiGraph,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
    ) -> EnergyWeightFunction:
        return EnergyWeightFunction(
            graph=graph,
            predictor=self.predictor,
            vehicle_context=vehicle_context,
            driver_context=driver_context,
            weather_context=weather_context,
            departure_time=departure_time,
        )

    def soc_aware_shortest_path(
        self,
        G: nx.MultiDiGraph,
        origin: Any,
        destination: Any,
        vehicle_context: Dict[str, Any],
        driver_context: Dict[str, Any],
        weather_context: Dict[str, Any],
        departure_time: datetime,
        nominal_battery_capacity_kwh: float,
        initial_soc: float,
        min_soc: float = 0.12,
        terminal_min_soc: Optional[float] = None,
        soc_step: float = 0.02,
        gamma_time_weight: Optional[float] = None,
        price_weight_kwh_per_usd: float = 0.0,
        objective: str = "energy",  # 'energy' | 'cost' | 'time' | 'weighted'
        alpha_usd_per_hour: float = 0.0,
        beta_kwh_to_usd: float = 0.0,
        charging_stations: Optional[List[Dict[str, Any]]] = None,
        station_overhead_min: float = 5.0,
        max_charge_rate_kw_default: float = 50.0,
    ) -> Dict[str, Any]:
        ambient_c = float(weather_context.get("temperature", 15.0))
        effective_capacity = _effective_battery_capacity_kwh(
            ambient_c, nominal_battery_capacity_kwh
        )
        if gamma_time_weight is None:
            try:
                gamma_time_weight = float(OPTIMIZATION_CONFIG.get("gamma_time_weight", 0.02))
            except Exception:
                gamma_time_weight = 0.02
        info(
            f"SOC routing init: init_soc={initial_soc:.2f}, min_soc={min_soc:.2f}, soc_step={soc_step:.2f}, "
            f"gamma={gamma_time_weight}, price_w={price_weight_kwh_per_usd}",
            "optimization",
        )

        station_by_node: Dict[Any, Dict[str, Any]] = {}
        if charging_stations:
            for st in charging_stations:
                node_id = st.get("node_id")
                if node_id is None and "latitude" in st and "longitude" in st:
                    node_id = _nearest_node_linear(G, st["latitude"], st["longitude"])
                if node_id is None:
                    continue
                max_power = float(st.get("max_power_kw", max_charge_rate_kw_default))
                prev = station_by_node.get(node_id)
                if prev is None or max_power > prev.get("max_power_kw", 0):
                    station_by_node[node_id] = {
                        "max_power_kw": max_power,
                        "cost_per_kwh": float(st.get("cost_per_kwh", st.get("estimated_cost_per_kwh", 0.35))),
                    }

        energy_w = self.make_weight_function(
            G, vehicle_context, driver_context, weather_context, departure_time
        )

        num_buckets = int(round(1.0 / soc_step)) + 1
        soc_to_bucket = lambda soc: int(np.clip(round(soc / soc_step), 0, num_buckets - 1))
        bucket_to_soc = lambda b: np.clip(b * soc_step, 0.0, 1.0)
        min_bucket = soc_to_bucket(min_soc)
        term_bucket = soc_to_bucket(terminal_min_soc) if terminal_min_soc is not None else min_bucket
        init_bucket = soc_to_bucket(initial_soc)

        import heapq

        pq: List[Tuple[float, Any, int]] = []
        heapq.heappush(pq, (0.0, origin, init_bucket))
        best_cost: Dict[Tuple[Any, int], float] = {(origin, init_bucket): 0.0}
        best_time_s: Dict[Tuple[Any, int], float] = {(origin, init_bucket): 0.0}
        parent: Dict[Tuple[Any, int], Tuple[Any, int]] = {}
        action_at: Dict[Tuple[Any, int], Dict[str, Any]] = {}

        visited: set = set()

        relax_drive = 0
        relax_charge = 0
        while pq:
            cost, node, b = heapq.heappop(pq)
            state = (node, b)
            if state in visited:
                continue
            visited.add(state)
            if node == destination and b >= term_bucket:
                break

            for _, v, key, edata in G.out_edges(node, keys=True, data=True):
                edata_local = dict(edata)
                edata_local["key"] = key
                e_kwh = float(energy_w(node, v, edata_local))
                travel_time_s = float(edata.get("travel_time", 0.0))
                soc_drop = e_kwh / max(effective_capacity, 1e-6)
                new_soc = bucket_to_soc(b) - soc_drop
                if new_soc < min_soc - 1e-9:
                    continue
                new_b = soc_to_bucket(new_soc)
                time_h = travel_time_s / 3600.0
                if objective == "time":
                    edge_cost = time_h
                elif objective == "cost":
                    edge_cost = alpha_usd_per_hour * time_h
                elif objective == "weighted":
                    edge_cost = beta_kwh_to_usd * e_kwh + alpha_usd_per_hour * time_h
                else:
                    edge_cost = e_kwh + (gamma_time_weight or 0.0) * time_h
                new_cost = cost + edge_cost
                new_time_s = best_time_s[state] + travel_time_s
                next_state = (v, new_b)
                if new_cost + 1e-12 < best_cost.get(next_state, float("inf")):
                    best_cost[next_state] = new_cost
                    best_time_s[next_state] = new_time_s
                    parent[next_state] = state
                    action_at[next_state] = {"type": "drive", "energy_kwh": e_kwh, "time_s": travel_time_s}
                    heapq.heappush(pq, (new_cost, v, new_b))
                    relax_drive += 1

            st = station_by_node.get(node)
            if st is not None and b < num_buckets - 1:
                max_power_kw = st.get("max_power_kw", max_charge_rate_kw_default)
                for delta_bucket in (1, 2, 3, 5, 10):
                    nb = min(num_buckets - 1, b + delta_bucket)
                    if nb == b:
                        continue
                    added_soc = bucket_to_soc(nb) - bucket_to_soc(b)
                    added_kwh = added_soc * effective_capacity
                    time_h = added_kwh / max(max_power_kw, 0.1)
                    time_s = time_h * 3600.0 + station_overhead_min * 60.0
                    arrival_time = departure_time + timedelta(seconds=best_time_s[state])
                    price_per_kwh = get_price_per_kwh(st, arrival_time, CHARGING_CONFIG if isinstance(CHARGING_CONFIG, dict) else {}, is_home=False)
                    if objective == "time":
                        charge_cost = time_h
                    elif objective == "cost":
                        charge_cost = added_kwh * price_per_kwh + alpha_usd_per_hour * time_h
                    elif objective == "weighted":
                        charge_cost = beta_kwh_to_usd * added_kwh + alpha_usd_per_hour * time_h
                    else:
                        charge_cost = added_kwh + (gamma_time_weight or 0.0) * time_h
                    new_cost = cost + charge_cost
                    new_time = best_time_s[state] + time_s
                    next_state = (node, nb)
                    if new_cost + 1e-12 < best_cost.get(next_state, float("inf")):
                        best_cost[next_state] = new_cost
                        best_time_s[next_state] = new_time
                        parent[next_state] = state
                        action_at[next_state] = {
                            "type": "charge",
                            "added_kwh": added_kwh,
                            "time_s": time_s,
                            "power_kw": max_power_kw,
                            "price_usd_per_kwh": price_per_kwh,
                            "cost_added": charge_cost,
                        }
                        heapq.heappush(pq, (new_cost, node, nb))
                        relax_charge += 1

        best_final_state = None
        best_final_cost = float("inf")
        for b in range(term_bucket, num_buckets):
            s = (destination, b)
            c = best_cost.get(s)
            if c is not None and c < best_final_cost:
                best_final_cost = c
                best_final_state = s

        if best_final_state is None:
            warning("SOC routing infeasible: no terminal state above min SOC", "optimization")
            return {"feasible": False, "reason": "No SOC-feasible path found"}

        path_nodes: List[Any] = []
        actions: List[Dict[str, Any]] = []
        s = best_final_state
        while s in parent:
            prev = parent[s]
            actions.append({"at": s[0], **action_at[s]})
            path_nodes.append(s[0])
            s = prev
        path_nodes.append(origin)
        path_nodes.reverse()
        actions.reverse()

        total_energy = sum(a.get("energy_kwh", 0.0) for a in actions if a["type"] == "drive")
        total_time_s = sum(a.get("time_s", 0.0) for a in actions)
        num_charges = sum(1 for a in actions if a["type"] == "charge")
        # Always compute USD cost from charge actions for KPI comparability, regardless of optimization objective
        total_cost_usd = 0.0
        for a in actions:
            if a["type"] == "charge":
                total_cost_usd += a.get("added_kwh", 0.0) * a.get("price_usd_per_kwh", 0.0)

        info(
            f"SOC routing done: path_nodes={len(path_nodes)}, drives={relax_drive}, charges={relax_charge}, "
            f"total_energy_kwh={total_energy:.3f}, total_time_s={total_time_s:.1f}, "
            f"total_cost_usd={(total_cost_usd if total_cost_usd is not None else 0.0):.2f}",
            "optimization",
        )
        return {
            "feasible": True,
            "path": path_nodes,
            "actions": actions,
            "metrics": {
                "energy_kwh": total_energy,
                "travel_time_s": total_time_s,
                "num_charges": num_charges,
                "cost_usd": total_cost_usd,
            },
        }


def optimize_charging_schedule_lp(
    edge_energy_kwh: List[float],
    edge_travel_time_s: List[float],
    stations_along_path: List[Optional[Dict[str, Any]]],
    nominal_battery_capacity_kwh: float,
    ambient_temp_c: float,
    initial_soc: float,
    min_soc: float = 0.12,
    objective: str = "min_time",
) -> Dict[str, Any]:
    assert len(edge_energy_kwh) == len(edge_travel_time_s)
    n_edges = len(edge_energy_kwh)
    eff_cap = _effective_battery_capacity_kwh(ambient_temp_c, nominal_battery_capacity_kwh)

    prob = LpProblem("ChargingOptimization", LpMinimize)
    charge_vars = []
    for i in range(n_edges + 1):
        st = stations_along_path[i]
        ub = 0.0 if st is None else eff_cap
        var = LpVariable(f"charge_{i}", lowBound=0.0, upBound=ub)
        charge_vars.append(var)

    soc = [None] * (n_edges + 1)
    soc[0] = initial_soc * eff_cap + charge_vars[0]
    prob += soc[0] >= min_soc * eff_cap
    for i in range(n_edges):
        soc[i + 1] = soc[i] - edge_energy_kwh[i] + charge_vars[i + 1]
        prob += soc[i + 1] >= min_soc * eff_cap
        prob += soc[i + 1] <= eff_cap

    if objective == "min_time":
        total_time_s = sum(edge_travel_time_s) + lpSum([
            (charge_vars[i] / max(float(stations_along_path[i].get("max_power_kw", 22.0)), 0.1)) * 3600.0
            for i in range(n_edges + 1) if stations_along_path[i] is not None
        ])
        prob += total_time_s
    else:
        prob += lpSum([
            charge_vars[i] * float(stations_along_path[i].get("cost_per_kwh", 0.35))
            for i in range(n_edges + 1) if stations_along_path[i] is not None
        ])

    status = prob.solve()
    if LpStatus[status] != "Optimal":
        return {"feasible": False, "status": LpStatus[status]}

    charges = [value(v) for v in charge_vars]
    final_soc_kwh = value(soc[-1]) if soc[-1] is not None else None
    # Compute explicit time metrics for interpretability
    charge_time_s = 0.0
    if objective == "min_time":
        for i, c in enumerate(charges):
            st = stations_along_path[i]
            if st is None:
                continue
            power_kw = max(float(st.get("max_power_kw", 22.0)), 0.1)
            charge_time_s += (c / power_kw) * 3600.0
    total_time_s = sum(edge_travel_time_s) + charge_time_s

    return {
        "feasible": True,
        "charges_kwh": charges,
        "final_soc_kwh": final_soc_kwh,
        "objective": value(prob.objective),
        "total_time_s": total_time_s,
        "charge_time_s": charge_time_s,
    }


# Placeholders for OR-Tools integration (VRP, multi-vehicle)

def formulate_rcsp_milp_placeholder():
    """
    Sketch: Resource-Constrained Shortest Path (SOC-aware).
    - Nodes expanded by SOC buckets
    - Edge transitions consume SOC using ML edge energy
    - Charging arcs add SOC with time/cost penalties
    Implement with PuLP/Pyomo/Gurobi when instance sizes are manageable.
    """
    pass


def ortools_vrp_placeholder():
    """
    Sketch: OR-Tools VRP with energy cost callback.
    - Cost = ML-predicted energy between stops
    - Battery constraints as dimension with capacity
    - Optional charging stops as extra nodes with service times
    Implement once visit sets and fleet size are defined.
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        warning("OR-Tools not available; install ortools to enable VRP prototype", "optimization")
        return None

    # Minimal scaffold (not executing) to illustrate wiring
    def build_model(num_vehicles: int, stops: List[Dict[str, float]], depot_index: int = 0):
        manager = pywrapcp.RoutingIndexManager(len(stops), num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)

        def distance_km(i, j):
            a, b = stops[i], stops[j]
            dy = a['lat'] - b['lat']
            dx = a['lon'] - b['lon']
            return float((dy * dy + dx * dx) ** 0.5 * 111.0)

        # Energy/Cost callback placeholder using approximate kWh from km
        def cost_callback(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            km = distance_km(i, j)
            kwh = km * 0.18
            usd = kwh * 0.30
            return int(usd * 1000)  # integer cost

        transit_cb = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        # Not solved here; caller would populate real ML/ToU callbacks and solve
        return manager, routing, search_params

    return {"build_model": build_model}

