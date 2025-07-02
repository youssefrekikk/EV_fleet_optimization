import pandas as pd
import numpy as np
import json
from collections import Counter

def extract_charging_patterns(input_path='data/raw/detailed_ev_charging_stations.csv'):
    """
    Extract realistic patterns and distributions from charging station data
    for use in synthetic data generation
    """
    df = pd.read_csv(input_path)
    
    patterns = {}
    
    # 1. Charger Type Distribution
    charger_type_dist = df['Charger Type'].value_counts(normalize=True).to_dict()
    patterns['charger_type_distribution'] = charger_type_dist
    
    # 2. Charging Capacity Distribution by Type
    capacity_by_type = {}
    for charger_type in df['Charger Type'].unique():
        capacities = df[df['Charger Type'] == charger_type]['Charging Capacity (kW)'].tolist()
        capacity_by_type[charger_type] = {
            'mean': np.mean(capacities),
            'std': np.std(capacities),
            'min': np.min(capacities),
            'max': np.max(capacities),
            'values': capacities
        }
    patterns['capacity_by_type'] = capacity_by_type
    
    # 3. Cost Distribution by Type
    cost_by_type = {}
    for charger_type in df['Charger Type'].unique():
        costs = df[df['Charger Type'] == charger_type]['Cost (USD/kWh)'].tolist()
        cost_by_type[charger_type] = {
            'mean': np.mean(costs),
            'std': np.std(costs),
            'min': np.min(costs),
            'max': np.max(costs),
            'values': costs
        }
    patterns['cost_by_type'] = cost_by_type
    
    # 4. Availability Patterns
    availability_patterns = df['Availability'].value_counts(normalize=True).to_dict()
    patterns['availability_patterns'] = availability_patterns
    
    # 5. Usage Statistics Distribution
    usage_stats = df['Usage Stats (avg users/day)'].tolist()
    patterns['usage_statistics'] = {
        'mean': np.mean(usage_stats),
        'std': np.std(usage_stats),
        'percentiles': {
            '25': np.percentile(usage_stats, 25),
            '50': np.percentile(usage_stats, 50),
            '75': np.percentile(usage_stats, 75),
            '90': np.percentile(usage_stats, 90)
        }
    }
    
    # 6. Station Operator Distribution
    operator_dist = df['Station Operator'].value_counts(normalize=True).to_dict()
    patterns['operator_distribution'] = operator_dist
    
    # 7. Connector Types Distribution
    all_connectors = []
    for connector_str in df['Connector Types'].dropna():
        connectors = [c.strip() for c in str(connector_str).split(',')]
        all_connectors.extend(connectors)
    
    connector_dist = dict(Counter(all_connectors))
    total_connectors = sum(connector_dist.values())
    connector_dist = {k: v/total_connectors for k, v in connector_dist.items()}
    patterns['connector_distribution'] = connector_dist
    
    # 8. Rating Distribution
    ratings = df['Reviews (Rating)'].tolist()
    patterns['rating_distribution'] = {
        'mean': np.mean(ratings),
        'std': np.std(ratings),
        'values': ratings
    }
    
    # 9. Parking Spots Distribution
    parking_spots = df['Parking Spots'].tolist()
    patterns['parking_spots'] = {
        'mean': np.mean(parking_spots),
        'std': np.std(parking_spots),
        'values': parking_spots
    }
    
    # 10. Maintenance Frequency Distribution
    maintenance_dist = df['Maintenance Frequency'].value_counts(normalize=True).to_dict()
    patterns['maintenance_frequency'] = maintenance_dist
    
    # 11. Installation Year Distribution
    years = df['Installation Year'].tolist()
    patterns['installation_years'] = {
        'mean': np.mean(years),
        'std': np.std(years),
        'min': np.min(years),
        'max': np.max(years),
        'values': years
    }
    
    # 12. Renewable Energy Distribution
    renewable_dist = df['Renewable Energy Source'].value_counts(normalize=True).to_dict()
    patterns['renewable_energy'] = renewable_dist
    
    return patterns

def create_charging_station_generator(patterns):
    """
    Create a function that generates realistic charging station characteristics
    """
    def generate_station_characteristics():
        """Generate realistic charging station characteristics"""
        
        # Select charger type based on distribution
        charger_type = np.random.choice(
            list(patterns['charger_type_distribution'].keys()),
            p=list(patterns['charger_type_distribution'].values())
        )
        
        # Generate capacity based on charger type
        capacity_info = patterns['capacity_by_type'][charger_type]
        if charger_type == 'AC Level 1':
            capacity = np.random.normal(1.4, 0.2)
            capacity = np.clip(capacity, 1.0, 2.0)
        elif charger_type == 'AC Level 2':
            capacity = np.random.choice([7.4, 11, 22], p=[0.6, 0.3, 0.1])
        elif charger_type == 'DC Fast Charger':
            capacity = np.random.choice([50, 75, 100, 150, 175], p=[0.3, 0.2, 0.2, 0.2, 0.1])
        else:  # Tesla or other
            capacity = np.random.choice([150, 250, 350], p=[0.4, 0.5, 0.1])
        
        # Generate cost based on charger type
        cost_info = patterns['cost_by_type'][charger_type]
        cost = np.random.normal(cost_info['mean'], cost_info['std'])
        cost = np.clip(cost, 0.05, 1.0)
        
        # Generate availability
        availability = np.random.choice(
            list(patterns['availability_patterns'].keys()),
            p=list(patterns['availability_patterns'].values())
        )
        
        # Generate usage statistics
        usage_mean = patterns['usage_statistics']['mean']
        usage_std = patterns['usage_statistics']['std']
        usage = max(1, int(np.random.normal(usage_mean, usage_std)))
        
        # Generate operator
        operator = np.random.choice(
            list(patterns['operator_distribution'].keys()),
            p=list(patterns['operator_distribution'].values())
        )
        
        # Generate connectors
        num_connectors = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        available_connectors = list(patterns['connector_distribution'].keys())
        connector_probs = list(patterns['connector_distribution'].values())
        connectors = np.random.choice(
            available_connectors, 
            size=min(num_connectors, len(available_connectors)),
            replace=False,
            p=connector_probs
        ).tolist()
        
        # Generate rating
        rating_mean = patterns['rating_distribution']['mean']
        rating_std = patterns['rating_distribution']['std']
        rating = np.clip(np.random.normal(rating_mean, rating_std), 1.0, 5.0)
        
        # Generate parking spots
        parking_mean = patterns['parking_spots']['mean']
        parking_std = patterns['parking_spots']['std']
        parking_spots = max(1, int(np.random.normal(parking_mean, parking_std)))
        
        # Generate maintenance frequency
        maintenance = np.random.choice(
            list(patterns['maintenance_frequency'].keys()),
            p=list(patterns['maintenance_frequency'].values())
        )
        
        # Generate installation year
        year_mean = patterns['installation_years']['mean']
        year_std = patterns['installation_years']['std']
        installation_year = int(np.clip(
            np.random.normal(year_mean, year_std), 
            2008, 2024
        ))
        
        # Generate renewable energy
        renewable = np.random.choice(
            list(patterns['renewable_energy'].keys()),
            p=list(patterns['renewable_energy'].values())
        )
        
        return {
            'charger_type': charger_type,
            'capacity_kw': round(capacity, 1),
            'cost_usd_per_kwh': round(cost, 3),
            'availability': availability,
            'usage_per_day': usage,
            'operator': operator,
            'connectors': connectors,
            'rating': round(rating, 1),
            'parking_spots': parking_spots,
            'maintenance_frequency': maintenance,
            'installation_year': installation_year,
            'renewable_energy': renewable == 'Yes'
        }
    
    return generate_station_characteristics

# Save patterns for use in data generation
def save_patterns(patterns, output_path='config/charging_patterns.json'):
    """Save extracted patterns to JSON file"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    patterns_clean = convert_numpy(patterns)
    
    with open(output_path, 'w') as f:
        json.dump(patterns_clean, f, indent=2)
    
    print(f"Charging patterns saved to {output_path}")

if __name__ == "__main__":
    # Extract patterns
    patterns = extract_charging_patterns()
    
    # Save patterns
    save_patterns(patterns)
    
    # Test the generator
    generator = create_charging_station_generator(patterns)
    
    print("Sample generated charging station characteristics:")
    for i in range(3):
        print(f"\nStation {i+1}:")
        characteristics = generator()
        for key, value in characteristics.items():
            print(f"  {key}: {value}")
