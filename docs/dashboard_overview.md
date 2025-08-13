# EV Fleet Optimization Studio — Bay Area

This dashboard orchestrates the end-to-end workflow for EV fleet research and prototyping in the SF Bay Area.

## Workflow
1. Config Studio
   - Fleet: size, days, region
   - Charging: home access, powers, efficiencies
   - Optimization: routing algorithm, time/energy trade-offs, SOC planning knobs
   - Distributions: EV model market shares, driver profile proportions
2. Data Generation
   - Build/load OSM road network (NetworkX/OSMnx)
   - Generate synthetic fleet routes, weather, charging sessions
   - Save datasets to `data/synthetic/`
3. Model Training
   - Train segment energy models (RF, XGBoost, LightGBM, CatBoost)
   - Inspect metrics (MAE/RMSE/R2/MAPE/SMAPE)
4. Optimization
   - Run energy-aware routing (Dijkstra/A*)
   - Enable SOC planning to include charging stops and costs
   - Export results + KPIs to `data/analysis/optimized/`
5. Map & Analytics
   - Visualize routes, charging stations, segment energy
   - Drill into trips and compare baseline vs optimized KPIs

## Key Config Variables
- route_optimization_algorithm: 'dijkstra' | 'astar'
- gamma_time_weight: time vs energy trade-off (kWh/hour)
- price_weight_kwh_per_usd: cost sensitivity for SOC objective
- fleet_eval_max_days, fleet_eval_trip_sample_frac: speed up fleet runs
- soc_objective: 'energy' | 'cost' | 'time' | 'weighted'
- planning_mode: 'myopic' | 'next_trip' | 'rolling_horizon'
- reserve_soc, reserve_kwh: terminal SOC floor rules
- horizon_trips, horizon_hours: look-ahead depth
- home_charging_availability, home_charging_power, public_fast_charging_power
- ev_models_market_share, driver_profiles_proportion: distributions (sum to 1)

## 📚 Documentation
- **[Config Variables Explained](config_variables_explained.md)** - Detailed explanation of all config parameters
- [Route Optimization](route_optimization.md) - Routing algorithms and SOC planning
- [Advanced Energy Model](advanced_energy_model.md) - Physics-based energy consumption
- [Synthetic EV Generator](synthetic_ev_generator.md) - Data generation and fleet simulation

## 🚀 Quick Start
1. **Config Studio**: Set fleet_size=10, simulation_days=7 for quick testing
2. **Data Generation**: Build network, generate synthetic data
3. **Model Training**: Start with Random Forest for baseline
4. **Optimization**: Use Dijkstra with gamma=0.02 for energy focus
5. **Analysis**: Explore maps and compare KPIs

## 🔧 Performance Tips
- Start small: 10 vehicles, 7 days, 1 evaluation day
- Use sampling: fleet_eval_trip_sample_frac=0.7 for large fleets
- Choose algorithm: Dijkstra for accuracy, A* for speed
- SOC planning: Enable for realistic charging behavior


