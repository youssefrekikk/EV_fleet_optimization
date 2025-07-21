# EV Fleet Optimization - Synthetic Data Generation

## Overview

This project generates realistic synthetic data for Electric Vehicle (EV) fleet optimization research. The synthetic data includes GPS traces, energy consumption patterns, charging behavior, and road network information for the San Francisco Bay Area.

**Key Features:**
- Physics-based energy consumption modeling
- Real road network integration with intelligent fallbacks
- Diverse driver profiles and vehicle models
- Realistic charging behavior simulation
- Scalable data generation for research and optimization

## Why San Francisco Bay Area?

We chose the Bay Area as our target region for several strategic reasons:

1. **EV Adoption Leader**: Highest EV penetration rate in the US (~15% of new car sales)
2. **Diverse Geography**: Urban (SF), suburban (Peninsula), industrial (East Bay) - perfect for testing different fleet scenarios
3. **Charging Infrastructure**: Dense network of charging stations with varied operators and pricing
4. **Complex Routing**: Bridges, hills, and traffic patterns create realistic optimization challenges
5. **Weather Variation**: Mild climate with seasonal changes affecting battery efficiency
6. **Data Availability**: Rich OpenStreetMap coverage and public charging station data
7. **Research Relevance**: Many fleet operators and tech companies testing EV solutions here

## Structure

```
src/data_generation/
├── synthetic_ev_generator.py    # Main data generation orchestrator
├── road_network_db.py          # Road network management and persistence
└── openchargemap_api.py        # Real charging station data integration

config/
├── ev_config.py                # Main configuration hub
├── ev_models.py                # EV specifications and market data
├── driver_profiles.py          # Driving behavior patterns
└── physics_constants.py       # Energy consumption physics

data/
├── synthetic/                  # Generated datasets
└── networks/                   # Cached road networks
```

## Quick Start

# Set OpenChargeMap API key (optional)
.env file with  OPENCHARGEMAP_API_KEY="your_api_key_here"

# Generate synthetic data
python src/data_generation/synthetic_ev_generator.py

# Output files will be in data/synthetic/
```

## Core Components

### 1. `synthetic_ev_generator.py` - The Data Generation Engine

**Purpose**: Orchestrates the entire synthetic data generation process for EV fleet optimization.

**Key Responsibilities**:
- **Fleet Management**: Generates diverse EV fleet with realistic vehicle models, driver profiles, and charging access
- **Route Generation**: Creates realistic daily driving patterns using real road networks
- **Energy Modeling**: Calculates physics-based energy consumption including temperature effects, terrain, and driving style
- **Charging Simulation**: Models both home and public charging behavior with realistic pricing and availability
- **Data Export**: Produces clean datasets in multiple formats (CSV, Parquet) for optimization algorithms

**Key Features**:
```python
# Generate complete fleet dataset
generator = SyntheticEVGenerator()
datasets = generator.generate_complete_dataset(num_days=30)

# Includes:
# - routes: GPS traces with energy consumption
# - charging_sessions: Home and public charging events  
# - vehicle_states: Daily summaries per vehicle
# - weather: Environmental conditions affecting efficiency
# - fleet_info: Vehicle specifications and driver profiles
```

**Why This Approach**:
- **Realistic Patterns**: Uses real road networks and physics-based models instead of random data
- **Scalable**: Can generate thousands of vehicles and trips efficiently
- **Configurable**: Easy to adjust fleet size, simulation period, and behavior parameters
- **Research-Ready**: Outputs structured data perfect for machine learning and optimization algorithms

### 2. `road_network_db.py` - Smart Network Management

**Purpose**: Handles road network data with intelligent caching and fallback strategies.

**The Problem We Solved**:
- OpenStreetMap (OSM) data is unreliable - sometimes fails to load
- Bay Area is geographically complex (bridges, disconnected regions)
- Large networks are slow to process repeatedly
- Need realistic routing for optimization algorithms

**Our Solution**:
```python
class NetworkDatabase:
    def load_or_create_rich_network(self):
        # 1. Try loading cached network (fast)
        # 2. Try San Francisco OSM data (most reliable)
        # 3. Extend SF with Bay Area connections (bridges/highways)
        # 4. Fallback to simple mock network (always works)
```

**Network Enhancement Strategy**:
1. **Load San Francisco**: Most reliable OSM query (10,000+ real road nodes)
2. **Add Bay Area Cities**: Synthetic nodes for Oakland, San Jose, Palo Alto, etc.
3. **Connect with Highways**: Realistic highway connections (Bay Bridge, 101, 880)
4. **Validate Connectivity**: Ensure routes possible between all major cities

**Why This Architecture**:
- **Reliability**: Always produces a working network
- **Performance**: Caches networks to avoid repeated OSM calls
- **Realism**: Uses real SF roads extended with realistic Bay Area connections
- **Flexibility**: Easy to rebuild or clear cache when needed

### 3. `config/ev_config.py` - Centralized Configuration

**Purpose**: Single source of truth for all simulation parameters.

**Configuration Categories**:

```python
# Fleet composition and operation
FLEET_CONFIG = {
    'fleet_size': 50,           # Number of vehicles
    'simulation_days': 30,      # Data generation period
    'region': 'bay_area',       # Geographic focus
    'operating_hours': (6, 22)  # Active hours
}

# Charging infrastructure
CHARGING_CONFIG = {
    'enable_home_charging': True,
    'home_charging_availability': 0.8,  # 80% have home charging
    'home_charging_power': 7.4,         # kW (Level 2)
    'peak_hours': [(17, 21)],           # Peak pricing periods
    'base_electricity_cost': 0.15       # USD per kWh
}

# Environmental conditions
WEATHER_CONFIG = {
    'base_temperature': 20,      # Celsius
    'seasonal_amplitude': 8,     # Temperature variation
    'rain_probability': 0.15     # 15% chance of rain
}
```

**Why Centralized Configuration**:
- **Consistency**: All components use same parameters
- **Experimentation**: Easy to test different scenarios
- **Documentation**: Clear understanding of all assumptions
- **Reproducibility**: Consistent results across runs

## Data Generation Process

### Step 1: Fleet Initialization
```python
# Generate diverse fleet
vehicles = generator.generate_fleet_vehicles()

# Each vehicle has:
# - Model (Tesla Model 3, Nissan Leaf, etc.)
# - Driver profile (commuter, rideshare, delivery, casual)
# - Driving style (eco-friendly, normal, aggressive)
# - Home charging access (80% have access)
# - Home location (realistic Bay Area distribution)
```

### Step 2: Daily Route Generation
```python
# For each vehicle, each day:
routes = generator.generate_daily_routes(vehicle, date)

# Routes include:
# - Origin/destination coordinates
# - GPS trace with timestamps
# - Speed and elevation profiles
# - Physics-based energy consumption
# - Weather impact on efficiency
```

### Step 3: Charging Behavior Simulation
```python
# Realistic charging decisions:
charging_sessions = generator.generate_charging_sessions(vehicle, routes, date)

# Considers:
# - Battery state of charge
# - Charging threshold by driving style
# - Home vs public charging preferences
# - Peak hour pricing
# - Station availability and selection
```

### Step 4: Energy Consumption Modeling
```python
# Physics-based calculation:
consumption = generator._calculate_energy_consumption(gps_trace, vehicle, date)

# Includes:
# - Rolling resistance
# - Aerodynamic drag  
# - Elevation changes
# - Acceleration/deceleration
# - HVAC usage (temperature dependent)
# - Regenerative braking recovery
```

## Vehicle Models and Driver Profiles

### EV Models (from `config/ev_models.py`)
- **Tesla Model 3**: 75 kWh, 491 km range, 250 kW charging
- **Tesla Model Y**: 75 kWh, 455 km range, 250 kW charging  
- **Nissan Leaf**: 62 kWh, 385 km range, 100 kW charging
- **Chevy Bolt**: 65 kWh, 417 km range, 55 kW charging
- **Ford Mustang Mach-E**: 88 kWh, 502 km range, 150 kW charging
- **Hyundai Ioniq 5**: 77.4 kWh, 481 km range, 235 kW charging

### Driver Profiles (from `config/driver_profiles.py`)

**Commuter (40% of fleet)**:
- Daily distance: 65-130 km
- 2-4 trips per day
- High home charging probability (90%)
- Peak hours: 7-9 AM, 5-7 PM

**Rideshare (25% of fleet)**:
- Daily distance: 240-480 km
- 15-25 trips per day
- Lower home charging access (60%)
- Extended operating hours

**Delivery (20% of fleet)**:
- Daily distance: 160-320 km
- 20-40 trips per day
- Limited home charging (30%)
- Business hours operation

**Casual (15% of fleet)**:
- Daily distance: 30-100 km
- 1-3 trips per day
- High home charging access (95%)
- Off-peak driving patterns

## Output Data Structure

### Routes Dataset
```csv
vehicle_id,trip_id,date,origin_lat,origin_lon,destination_lat,destination_lon,
total_distance_km,total_consumption_kwh,efficiency_kwh_per_100km,
temperature_celsius,weather_is_raining,driver_profile,vehicle_model
```

### Charging Sessions Dataset  
```csv
vehicle_id,session_id,charging_type,start_time,end_time,
energy_delivered_kwh,cost_usd,start_soc,end_soc,charging_power_kw,
station_operator,connector_type,is_emergency_charging
```

### Vehicle States Dataset
```csv
vehicle_id,date,total_distance_km,total_consumption_kwh,
num_trips,num_charging_sessions,driver_profile,vehicle_model,
efficiency_kwh_per_100km
```

### Weather Dataset
```csv
date,temperature,is_raining,wind_speed_kmh,humidity,season
```

### Fleet Info Dataset
```csv
vehicle_id,model,battery_capacity,efficiency,max_charging_speed,
driver_profile,home_location_lat,home_location_lon,current_battery_soc
```

## Configuration for Different Scenarios

### Urban Delivery Fleet
```python
config_delivery = {
    'fleet': {'fleet_size': 100},
    'driver_profiles': {'delivery': 0.8, 'commuter': 0.2},
    'charging': {'home_charging_availability': 0.3}
}
```

### Rideshare Fleet
```python
config_rideshare = {
    'fleet': {'fleet_size': 200},
    'driver_profiles': {'rideshare': 0.9, 'casual': 0.1},
    'charging': {'home_charging_availability': 0.3}
}
```

### Corporate Commuter Fleet
```python
config_corporate = {
    'fleet': {'fleet_size': 50},
    'driver_profiles': {'commuter': 1.0},
    'charging': {'home_charging_availability': 0.95}
}
```

## Usage for Fleet Optimization

This synthetic data is designed for:

1. **Route Optimization**: Realistic travel times and distances
2. **Charging Optimization**: When/where to charge decisions
3. **Fleet Sizing**: How many vehicles needed for demand patterns
4. **Infrastructure Planning**: Optimal charging station placement
5. **Energy Management**: Predicting consumption and grid impact






