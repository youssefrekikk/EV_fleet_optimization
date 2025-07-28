"""
Test script for the EV consumption prediction model
"""
import pandas as pd
import matplotlib.pyplot as plt
from consumption_model import EVConsumptionPredictor
from pathlib import Path

def test_predictions():
    """Test various prediction scenarios"""
    
    # Load the trained model
    predictor = EVConsumptionPredictor()
    model_path = Path(__file__).parent / "consumption_model.pkl"
    
    if model_path.exists():
        predictor.load_model(model_path)
        print("✓ Model loaded successfully")
    else:
        print("✗ Model not found. Please run consumption_model.py first.")
        return
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Short city trip',
            'distance_km': 10,
            'time_minutes': 25,
            'temperature': 22,
            'vehicle_model': 'tesla_model_3',
            'driver_profile': 'commuter'
        },
        {
            'name': 'Highway commute',
            'distance_km': 60,
            'time_minutes': 50,
            'temperature': 18,
            'vehicle_model': 'tesla_model_y',
            'driver_profile': 'commuter'
        },
        {
            'name': 'Cold weather trip',
            'distance_km': 40,
            'time_minutes': 55,
            'temperature': -2,
            'vehicle_model': 'nissan_leaf',
            'driver_profile': 'casual'
        },
        {
            'name': 'Long distance',
            'distance_km': 150,
            'time_minutes': 120,
            'temperature': 25,
            'vehicle_model': 'tesla_model_3',
            'driver_profile': 'casual'
        }
    ]
    
    print("\n" + "="*60)
    print("EV CONSUMPTION PREDICTION TEST RESULTS")
    print("="*60)
    
    results = []
    for scenario in scenarios:
        name = scenario.pop('name')
        prediction = predictor.predict_trip_consumption(**scenario)
        
        results.append({
            'Scenario': name,
            'Distance (km)': scenario['distance_km'],
            'Time (min)': scenario['time_minutes'],
            'Temperature (°C)': scenario['temperature'],
            'Vehicle': scenario['vehicle_model'].replace('_', ' ').title(),
            'Driver': scenario['driver_profile'].title(),
            'Consumption (kWh)': f"{prediction['predicted_consumption_kwh']:.2f}",
            'Efficiency (kWh/100km)': f"{prediction['efficiency_kwh_per_100km']:.1f}"
        })
        
        print(f"\n{name}:")
        print(f"  Distance: {scenario['distance_km']} km")
        print(f"  Time: {scenario['time_minutes']} minutes")
        print(f"  Speed: {scenario['distance_km']/(scenario['time_minutes']/60):.1f} km/h")
        print(f"  Temperature: {scenario['temperature']}°C")
        print(f"  Vehicle: {scenario['vehicle_model'].replace('_', ' ').title()}")
        print(f"  Driver: {scenario['driver_profile'].title()}")
        print(f"  → Predicted consumption: {prediction['predicted_consumption_kwh']:.2f} kWh")
        print(f"  → Efficiency: {prediction['efficiency_kwh_per_100km']:.1f} kWh/100km")
    
    # Create comparison chart
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Consumption comparison
    scenarios_short = [r['Scenario'][:15] for r in results]
    consumptions = [float(r['Consumption (kWh)']) for r in results]
    
    bars1 = ax1.bar(scenarios_short, consumptions, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Predicted Energy Consumption by Scenario')
    ax1.set_ylabel('Consumption (kWh)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, consumptions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Efficiency comparison
    efficiencies = [float(r['Efficiency (kWh/100km)']) for r in results]
    
    bars2 = ax2.bar(scenarios_short, efficiencies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('Predicted Efficiency by Scenario')
    ax2.set_ylabel('Efficiency (kWh/100km)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('consumption_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "="*60)
    print("Summary:")
    print(f"✓ Tested {len(scenarios)} different scenarios")
    print(f"✓ Best efficiency: {min(efficiencies):.1f} kWh/100km")
    print(f"✓ Worst efficiency: {max(efficiencies):.1f} kWh/100km")
    print(f"✓ Average consumption: {sum(consumptions)/len(consumptions):.1f} kWh")
    print("✓ Chart saved as 'consumption_predictions.png'")
    print("="*60)

def test_temperature_sensitivity():
    """Test how consumption varies with temperature"""
    
    predictor = EVConsumptionPredictor()
    model_path = Path(__file__).parent / "consumption_model.pkl"
    predictor.load_model(model_path)
    
    # Test temperature range
    temperatures = range(-10, 36, 5)
    consumptions = []
    
    for temp in temperatures:
        prediction = predictor.predict_trip_consumption(
            distance_km=50,
            time_minutes=60,
            temperature=temp,
            vehicle_model='tesla_model_3',
            driver_profile='commuter'
        )
        consumptions.append(prediction['predicted_consumption_kwh'])
    
    # Plot temperature sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, consumptions, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    plt.title('EV Consumption vs Temperature\n(50km trip in Tesla Model 3)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Predicted Consumption (kWh)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=sum(consumptions)/len(consumptions), color='gray', linestyle='--', alpha=0.7, label='Average')
    plt.legend()
    
    # Add annotations for extreme values
    min_idx = consumptions.index(min(consumptions))
    max_idx = consumptions.index(max(consumptions))
    
    plt.annotate(f'Lowest: {min(consumptions):.2f} kWh\nat {temperatures[min_idx]}°C',
                xy=(temperatures[min_idx], min(consumptions)),
                xytext=(temperatures[min_idx] + 5, min(consumptions) + 0.5),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.annotate(f'Highest: {max(consumptions):.2f} kWh\nat {temperatures[max_idx]}°C',
                xy=(temperatures[max_idx], max(consumptions)),
                xytext=(temperatures[max_idx] - 5, max(consumptions) - 0.5),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('temperature_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTemperature Sensitivity Analysis:")
    print(f"Temperature range: {min(temperatures)}°C to {max(temperatures)}°C")
    print(f"Consumption range: {min(consumptions):.2f} - {max(consumptions):.2f} kWh")
    print(f"Difference: {max(consumptions) - min(consumptions):.2f} kWh ({((max(consumptions) - min(consumptions))/min(consumptions)*100):.1f}% increase)")

if __name__ == "__main__":
    print("Testing EV Consumption Prediction Model")
    print("="*40)
    
    test_predictions()
    print("\n" + "="*40)
    test_temperature_sensitivity()
