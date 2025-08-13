# EV Fleet Optimization Config Variables Explained

This document explains all the configuration variables used in the EV Fleet Optimization Studio, their meaning, typical values, and how they affect the system.

## 🚗 Fleet Configuration

### `fleet_size`
- **What**: Number of EVs to simulate in the fleet
- **Range**: 1-500
- **Typical**: 10-100 for testing, 200+ for production analysis
- **Impact**: More vehicles = more realistic fleet behavior, longer generation time

### `simulation_days`
- **What**: How many days of operations to generate
- **Range**: 1-60
- **Typical**: 7-14 for pattern analysis, 30+ for seasonal trends
- **Impact**: More days = better statistical significance, longer processing

### `region`
- **What**: Geographic area for simulation
- **Current**: "bay_area" (San Francisco, Oakland, San Jose, Peninsula, North Bay)
- **Impact**: Affects road network, weather patterns, charging infrastructure

## 🔌 Charging Infrastructure

### `enable_home_charging`
- **What**: Allow drivers with home access to charge overnight
- **Type**: Boolean
- **Typical**: True (80% of Bay Area residents have home charging)
- **Impact**: Reduces public charging demand, more realistic behavior

### `home_charging_availability`
- **What**: Share of drivers with home charging access
- **Range**: 0.0-1.0
- **Typical**: 0.8 (80%)
- **Impact**: Higher values = less public charging pressure, more suburban behavior

### `home_charging_power`
- **What**: Power level of home chargers in kW
- **Range**: 1.0-350.0 kW
- **Typical**: 7.4 kW (Level 2)
- **Impact**: Higher power = faster overnight charging, less range anxiety

### `public_fast_charging_power`
- **What**: Power level of public DC fast chargers in kW
- **Range**: 20.0-350.0 kW
- **Typical**: 150 kW
- **Impact**: Higher power = faster public charging, shorter stops

## 🛰️ Route Optimization

### `route_optimization_algorithm`
- **What**: Algorithm for finding energy-optimal routes
- **Options**: "dijkstra", "astar"
- **Typical**: "dijkstra"
- **Impact**: 
  - Dijkstra: Guaranteed optimal, slower
  - A*: Faster with heuristic, may be suboptimal

### `gamma_time_weight` (γ)
- **What**: Time vs energy trade-off in routing cost
- **Range**: 0.0-1.0 kWh/hour
- **Typical**: 0.02 kWh/hour
- **Impact**: 
  - 0.0 = ignore time, optimize only for energy
  - 0.02 = 1 hour of time = 0.02 kWh penalty
  - Higher values = prioritize speed over efficiency
- **Formula**: Total Cost = Energy Cost + γ × Time Cost

### `price_weight_kwh_per_usd`
- **What**: Cost sensitivity for SOC planning objective
- **Range**: 0.0-10.0 kWh/USD
- **Typical**: 0.0 (ignore cost)
- **Impact**: 
  - 0.0 = optimize only for energy
  - 2.0 = $1 of charging cost = 2 kWh penalty
  - Higher values = make cost more important than energy

### `battery_buffer_percentage`
- **What**: Safety margin above minimum SOC in planning
- **Range**: 5-50%
- **Typical**: 15%
- **Impact**: Higher values = more conservative planning, less range anxiety

### `max_detour_for_charging_km`
- **What**: Maximum distance to go out of way for charging
- **Range**: 0.0-50.0 km
- **Typical**: 5.0 km
- **Impact**: Higher values = more charging options, potentially longer routes

## ⚡ Advanced Optimization

### `fleet_eval_max_days`
- **What**: Limit number of days to process in fleet evaluation
- **Range**: 0-365 (0 = all days)
- **Typical**: 1-7 for testing
- **Impact**: Lower values = faster evaluation, good for iteration

### `fleet_eval_trip_sample_frac`
- **What**: Fraction of trips per day to process
- **Range**: 0.0-1.0
- **Typical**: 0.7 (70%)
- **Impact**: Lower values = faster processing, good for large fleets

### `soc_objective`
- **What**: What to optimize for in SOC-aware routing
- **Options**: "energy", "cost", "time", "weighted"
- **Typical**: "energy"
- **Impact**:
  - "energy": Minimize total kWh consumption
  - "cost": Minimize total charging cost
  - "time": Minimize total travel + charging time
  - "weighted": Balance of all objectives

### `planning_mode`
- **What**: Look-ahead depth for battery planning
- **Options**: "myopic", "next_trip", "rolling_horizon"
- **Typical**: "myopic"
- **Impact**:
  - "myopic": No future knowledge, simple planning
  - "next_trip": Reserve energy for next trip
  - "rolling_horizon": Reserve for K future trips

### `alpha_usd_per_hour`
- **What**: Value of time for cost-based objective
- **Range**: 0.0-200.0 USD/hour
- **Typical**: 0.0 (ignore time value)
- **Impact**: Higher values = time is more valuable, prefer faster routes

### `beta_kwh_to_usd`
- **What**: Convert kWh to USD for weighted objective
- **Range**: 0.0-5.0 USD/kWh
- **Typical**: 0.15 (Bay Area electricity rate)
- **Impact**: Higher values = energy is more expensive

### `reserve_soc`
- **What**: Terminal SOC floor when no future energy known
- **Range**: 0.0-0.95
- **Typical**: 0.15 (15%)
- **Impact**: Higher values = more conservative planning

### `reserve_kwh`
- **What**: Reserve energy in kWh (overrides SOC if > 0)
- **Range**: 0.0-200.0 kWh
- **Typical**: 0.0 (use SOC floor)
- **Impact**: Absolute energy reserve regardless of battery capacity

### `horizon_trips`
- **What**: Number of future trips to consider in planning
- **Range**: 0-10
- **Typical**: 1-3
- **Impact**: Higher values = better planning but slower computation

### `horizon_hours`
- **What**: Time horizon in hours for planning
- **Range**: 0.0-72.0 hours
- **Typical**: 0.0 (use trip count)
- **Impact**: Alternative to trip count when trip schedule is unknown

## 🚗 Fleet Composition

### `ev_models_market_share`
- **What**: Distribution of EV models in the fleet
- **Type**: Dict[str, float] (must sum to 1.0)
- **Typical**: Tesla dominant in Bay Area
- **Impact**: Affects average efficiency, charging behavior, range patterns

### `driver_profiles_proportion`
- **What**: Distribution of driver behavior types
- **Type**: Dict[str, float] (must sum to 1.0)
- **Options**: commuter, rideshare, delivery, casual
- **Impact**: Affects trip patterns, charging urgency, route preferences

## 🔧 How to Use These Variables

### Quick Start
1. **Small Fleet Test**: fleet_size=10, simulation_days=7, fleet_eval_max_days=1
2. **Energy Focus**: soc_objective="energy", gamma_time_weight=0.01
3. **Cost Focus**: soc_objective="cost", price_weight_kwh_per_usd=2.0
4. **Speed Focus**: gamma_time_weight=0.1, soc_objective="time"

### Production Settings
1. **Large Fleet**: fleet_size=200+, simulation_days=30+, fleet_eval_trip_sample_frac=0.5
2. **Conservative Planning**: battery_buffer_percentage=20, reserve_soc=0.20
3. **Advanced Planning**: planning_mode="rolling_horizon", horizon_trips=5

### Bay Area Specific
1. **Home Charging**: home_charging_availability=0.8 (suburban pattern)
2. **Fast Charging**: public_fast_charging_power=150 (common DC fast)
3. **EV Mix**: Tesla models dominant, commuter profile high

## 📊 Performance Impact

### Fastest to Slowest
1. **Quick Test**: 1-2 minutes (small fleet, few days, basic routing)
2. **Standard Run**: 5-15 minutes (medium fleet, week simulation, ML routing)
3. **Full Analysis**: 30+ minutes (large fleet, month simulation, SOC planning)

### Memory Usage
- **Small**: 2-4 GB RAM
- **Medium**: 4-8 GB RAM  
- **Large**: 8+ GB RAM

## 🔍 Troubleshooting

### Common Issues
1. **Slow Generation**: Reduce fleet_size or simulation_days
2. **Memory Errors**: Lower fleet_eval_trip_sample_frac
3. **Poor Routes**: Check gamma_time_weight and algorithm choice
4. **Charging Issues**: Verify home_charging_availability and max_detour_for_charging_km

### Optimization Tips
1. Start with small values and scale up
2. Use sampling for large fleet evaluation
3. Balance energy vs time with gamma parameter
4. Consider Bay Area specific patterns (home charging, Tesla dominance)
