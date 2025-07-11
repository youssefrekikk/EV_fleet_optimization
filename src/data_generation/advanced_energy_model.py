"""
Advanced physics-based EV energy consumption model
Based on latest research incorporating temperature, weather, and battery dynamics
"""
import sys
import os
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.physics_constants import PHYSICS_CONSTANTS, TEMPERATURE_EFFICIENCY , BATTERY_PARAMETERS

logger = logging.getLogger(__name__)

class AdvancedEVEnergyModel:
    """
    Advanced EV energy consumption model incorporating:
    - Temperature-dependent battery internal resistance
    - Capacity corrections for temperature and discharge rate
    - Physics-based driving forces (aerodynamic, rolling, elevation)
    - HVAC load modeling
    - Regenerative braking efficiency curves
    """
    
    def __init__(self):
        # Load physics constants
        self.constants = PHYSICS_CONSTANTS
        self.temp_efficiency = TEMPERATURE_EFFICIENCY
        self.battery_params = BATTERY_PARAMETERS
        
        # Quick access to commonly used constants
        self.AIR_DENSITY = self.constants['air_density']
        self.GRAVITY = self.constants['gravity']
        self.ROLLING_RESISTANCE_BASE = self.constants['rolling_resistance']
        self.GAS_CONSTANT = self.constants['gas_constant']
        self.ACTIVATION_ENERGY = self.constants['activation_energy']
        self.REFERENCE_TEMP = self.constants['reference_temp_k']
        self.TEMP_CAPACITY_ALPHA = self.constants['temp_capacity_alpha']
        
    def calculate_energy_consumption(self, gps_trace: List[Dict], vehicle: Dict, 
                                   weather_conditions: Dict) -> Dict:
        """
        Calculate energy consumption using advanced physics-based model
        """
        
        if not gps_trace or len(gps_trace) < 2:
            return self._get_zero_consumption_result(weather_conditions)
        
        # Vehicle specifications
        vehicle_specs = self._get_vehicle_specs(vehicle)
        
        # Initialize tracking variables
        total_consumption = 0.0
        total_distance = 0.0
        consumption_breakdown = {
            'rolling_resistance': 0.0,
            'aerodynamic_drag': 0.0,
            'elevation_change': 0.0,
            'acceleration': 0.0,
            'hvac': 0.0,
            'auxiliary': 0.0,
            'regenerative_braking': 0.0,
            'battery_thermal_loss': 0.0
        }
        
        # Temperature-dependent battery parameters
        ambient_temp_k = weather_conditions['temperature'] + 273.15
        battery_params = self._calculate_battery_parameters(ambient_temp_k, vehicle_specs)
        
        # Process each GPS segment
        for i in range(len(gps_trace) - 1):
            try:
                segment_result = self._calculate_segment_consumption(
                    gps_trace[i], gps_trace[i + 1], 
                    vehicle_specs, battery_params, weather_conditions
                )
                
                if segment_result['distance_km'] > 0:  # Only count valid segments
                    total_consumption += segment_result['energy_kwh']
                    total_distance += segment_result['distance_km']
                    
                    # Update breakdown
                    for component, value in segment_result['breakdown'].items():
                        consumption_breakdown[component] += value
                        
            except Exception as e:
                logger.debug(f"Segment calculation error: {e}")
                continue
        
        # Calculate final metrics
        efficiency_kwh_per_100km = (total_consumption / total_distance * 100) if total_distance > 0 else 0
        
        # Apply minimum realistic consumption (prevent zero consumption)
        if total_distance > 0 and total_consumption < 0.05:
            # Minimum consumption based on vehicle weight and distance
            min_consumption = self._calculate_minimum_consumption(total_distance, vehicle_specs)
            total_consumption = max(total_consumption, min_consumption)
            efficiency_kwh_per_100km = total_consumption / total_distance * 100
        
        return {
            'total_consumption_kwh': round(total_consumption, 3),
            'total_distance_km': round(total_distance, 2),
            'efficiency_kwh_per_100km': round(efficiency_kwh_per_100km, 2),
            'temperature_celsius': weather_conditions['temperature'],
            'temperature_efficiency_factor': battery_params['temp_efficiency_factor'],
            'battery_internal_resistance_ohm': battery_params['internal_resistance'],
            'consumption_breakdown': {k: round(v, 4) for k, v in consumption_breakdown.items()},
            'weather_conditions': weather_conditions,
            'model_version': 'advanced_physics_v1.0'
        }
    
    def _calculate_battery_parameters(self, temp_k: float, vehicle_specs: Dict) -> Dict:
        """Calculate temperature-dependent battery parameters"""
        
        # Internal resistance temperature dependence: R(T) = R0 * exp(Ea/RT)
        temp_factor = (self.ACTIVATION_ENERGY / self.GAS_CONSTANT) * (1/temp_k - 1/self.REFERENCE_TEMP)
        internal_resistance = vehicle_specs['base_internal_resistance'] * math.exp(temp_factor)
        
        # Capacity temperature correction: C(T) = Cn * (T/Tref)^α
        capacity_factor = (temp_k / self.REFERENCE_TEMP) ** self.TEMP_CAPACITY_ALPHA
        effective_capacity = vehicle_specs['battery_capacity'] * capacity_factor
        
        # Overall temperature efficiency factor
        temp_efficiency_factor = self._get_temperature_efficiency_factor(temp_k - 273.15)
        
        return {
            'internal_resistance': internal_resistance,
            'effective_capacity': effective_capacity,
            'capacity_factor': capacity_factor,
            'temp_efficiency_factor': temp_efficiency_factor
        }
    
    def _calculate_segment_consumption(self, point1: Dict, point2: Dict, 
                                     vehicle_specs: Dict, battery_params: Dict,
                                     weather: Dict) -> Dict:
        """Calculate energy consumption for a single GPS segment"""
        
        # Calculate segment distance and validate
        distance_m = geodesic(
            (point1['latitude'], point1['longitude']),
            (point2['latitude'], point2['longitude'])
        ).meters
        
        if distance_m < 0.1:  # Skip segments shorter than 10cm
            return {'distance_km': 0, 'energy_kwh': 0, 'breakdown': {k: 0 for k in ['rolling_resistance', 'aerodynamic_drag', 'elevation_change', 'acceleration', 'hvac', 'auxiliary', 'regenerative_braking', 'battery_thermal_loss']}}
        
        # Calculate time difference and validate
        try:
            time1 = datetime.fromisoformat(point1['timestamp'].replace('Z', '+00:00'))
            time2 = datetime.fromisoformat(point2['timestamp'].replace('Z', '+00:00'))
            time_diff_s = (time2 - time1).total_seconds()
        except:
            # Fallback: estimate time from distance and speed
            avg_speed_ms = (point1.get('speed_kmh', 30) + point2.get('speed_kmh', 30)) / 2 / 3.6
            time_diff_s = distance_m / max(avg_speed_ms, 1.0)  # Prevent division by zero
        
        if time_diff_s <= 0:
            time_diff_s = distance_m / (30 / 3.6)  # Assume 30 km/h if time is invalid
        
        # Vehicle dynamics calculations
        speed1_ms = point1.get('speed_kmh', 0) / 3.6
        speed2_ms = point2.get('speed_kmh', 0) / 3.6
        avg_speed_ms = (speed1_ms + speed2_ms) / 2
        acceleration = (speed2_ms - speed1_ms) / time_diff_s
        
        elevation1 = point1.get('elevation_m', 0)
        elevation2 = point2.get('elevation_m', 0)
        elevation_change = elevation2 - elevation1
        
        # Physics-based force calculations
        forces = self._calculate_driving_forces(
            avg_speed_ms, acceleration, elevation_change, distance_m,
            vehicle_specs, weather
        )
        
        # Energy calculations (in Joules)
        energy_breakdown = {}
        
        # 1. Rolling resistance energy
        energy_breakdown['rolling_resistance'] = forces['rolling'] * distance_m
        
        # 2. Aerodynamic drag energy
        energy_breakdown['aerodynamic_drag'] = forces['aero'] * distance_m
        
        # 3. Elevation energy (can be negative for downhill)
        energy_breakdown['elevation_change'] = forces['elevation'] * distance_m
        
        # 4. Acceleration energy (kinetic energy change)
        kinetic_energy_change = 0.5 * vehicle_specs['mass'] * (speed2_ms**2 - speed1_ms**2)
        if acceleration > 0:  # Accelerating
            energy_breakdown['acceleration'] = kinetic_energy_change
            energy_breakdown['regenerative_braking'] = 0
        else:  # Decelerating - regenerative braking
            regen_efficiency = self._get_regenerative_efficiency(abs(acceleration), avg_speed_ms)
            energy_breakdown['acceleration'] = 0
            energy_breakdown['regenerative_braking'] = kinetic_energy_change * regen_efficiency
        
        # 5. HVAC energy
        hvac_power_w = self._calculate_hvac_power(weather['temperature'], vehicle_specs)
        energy_breakdown['hvac'] = hvac_power_w * time_diff_s
        
        # 6. Auxiliary systems energy
        aux_power_w = vehicle_specs['auxiliary_power'] * 1000  # Convert kW to W
        energy_breakdown['auxiliary'] = aux_power_w * time_diff_s
        
        # 7. Battery thermal losses (I²R losses)
        current_estimate = self._estimate_battery_current(forces, avg_speed_ms, vehicle_specs)
        thermal_loss_w = current_estimate**2 * battery_params['internal_resistance']
        energy_breakdown['battery_thermal_loss'] = thermal_loss_w * time_diff_s
        
        # Sum total energy and apply efficiency factors
        total_mechanical_energy = (
            energy_breakdown['rolling_resistance'] +
            energy_breakdown['aerodynamic_drag'] +
            max(0, energy_breakdown['elevation_change']) +  # Only uphill counts as consumption
            max(0, energy_breakdown['acceleration'])  # Only acceleration counts as consumption
        )
        
        total_auxiliary_energy = (
            energy_breakdown['hvac'] +
            energy_breakdown['auxiliary'] +
            energy_breakdown['battery_thermal_loss']
        )
        
        # Apply drivetrain efficiency to mechanical energy
        # Apply drivetrain efficiency to mechanical energy
        drivetrain_efficiency = (self.constants['motor_efficiency'] * 
                            self.constants['inverter_efficiency'] * 
                            self.constants['transmission_efficiency'])

        
        mechanical_energy_kwh = total_mechanical_energy / 3.6e6 / drivetrain_efficiency
        auxiliary_energy_kwh = total_auxiliary_energy / 3.6e6
        
        # Subtract regenerative braking energy
        regen_energy_kwh = abs(energy_breakdown['regenerative_braking']) / 3.6e6
        
        # Apply temperature efficiency factor
        temp_factor = battery_params['temp_efficiency_factor']
        
        total_energy_kwh = (mechanical_energy_kwh + auxiliary_energy_kwh - regen_energy_kwh) / temp_factor
        
        # Ensure minimum positive consumption
        total_energy_kwh = max(0.001, total_energy_kwh)  # Minimum 1 Wh
        
        # Convert breakdown to kWh
        breakdown_kwh = {k: v / 3.6e6 for k, v in energy_breakdown.items()}
        
        return {
            'distance_km': distance_m / 1000,
            'energy_kwh': total_energy_kwh,
            'breakdown': breakdown_kwh
        }
    
    def _calculate_driving_forces(self, speed_ms: float, acceleration: float, 
                                elevation_change: float, distance_m: float,
                                vehicle_specs: Dict, weather: Dict) -> Dict:
        """Calculate physics-based driving forces"""
        
        # Rolling resistance force
        # F_roll = Cr * m * g * cos(θ) where θ is road grade
        road_grade = math.atan(elevation_change / distance_m) if distance_m > 0 else 0
        rolling_coeff = self._get_rolling_resistance_coefficient(speed_ms, weather)
        rolling_force = rolling_coeff * vehicle_specs['mass'] * self.GRAVITY * math.cos(road_grade)
        
        # Aerodynamic drag force
        # F_aero = 0.5 * ρ * Cd * A * v²
        air_density = self._get_air_density(weather)
        aero_force = 0.5 * air_density * vehicle_specs['drag_coefficient'] * vehicle_specs['frontal_area'] * speed_ms**2
        
        # Add wind resistance
        wind_speed_ms = weather.get('wind_speed_kmh', 0) / 3.6
        relative_wind_speed = speed_ms + wind_speed_ms  # Simplified head/tail wind
        aero_force = 0.5 * air_density * vehicle_specs['drag_coefficient'] * vehicle_specs['frontal_area'] * relative_wind_speed**2
        
        # Elevation force (gravitational potential energy change)
        elevation_force = vehicle_specs['mass'] * self.GRAVITY * math.sin(road_grade)
        
        return {
            'rolling': rolling_force,
            'aero': aero_force,
            'elevation': elevation_force
        }
    
    def _get_rolling_resistance_coefficient(self, speed_ms: float, weather: Dict) -> float:
        """Calculate speed and weather-dependent rolling resistance coefficient"""
        
        # Base rolling resistance
        cr_base = self.ROLLING_RESISTANCE_BASE
        
        # Speed dependency: Cr increases with speed
        speed_kmh = speed_ms * 3.6
        speed_factor = 1 + (speed_kmh / 100) * self.constants['rolling_resistance_speed_factor']
        
        # Temperature dependency: tires get softer in heat, stiffer in cold
        temp_c = weather['temperature']
        if temp_c < 0:
            temp_factor = self.constants['rolling_resistance_cold_factor']
        elif temp_c > 35:
            temp_factor = self.constants['rolling_resistance_hot_factor']
        else:
            temp_factor = 1.0
        
        # Rain increases rolling resistance
        rain_factor = self.constants['rolling_resistance_rain_factor'] if weather.get('is_raining', False) else 1.0
        
        return cr_base * speed_factor * temp_factor * rain_factor


    def _get_air_density(self, weather: Dict) -> float:
        """Calculate air density based on temperature and humidity"""
        
        temp_k = weather['temperature'] + 273.15
        humidity = weather.get('humidity', 0.5)
        
        # Dry air density at temperature
        dry_air_density = self.AIR_DENSITY * (273.15 / temp_k)
        
        # Humidity correction (humid air is less dense)
        humidity_factor = 1 - self.constants['humidity_density_factor'] * humidity * 0.01
        
        return dry_air_density * humidity_factor

    
    def _calculate_hvac_power(self, temp_c: float, vehicle_specs: Dict) -> float:
        """Calculate HVAC power consumption based on temperature"""
        
        target_temp = self.constants['target_cabin_temp']
        temp_diff = abs(temp_c - target_temp)
        
        if temp_diff <= 2:
            # Minimal HVAC usage
            return vehicle_specs.get('hvac_base_power', self.constants['hvac_base_power']) * 1000 * 0.2
        
        # Determine heating vs cooling
        if temp_c < target_temp:
            # Heating mode
            if vehicle_specs.get('has_heat_pump', True):
                # Heat pump efficiency
                cop = max(1.5, self.constants['hvac_cop_heat_pump'] - (target_temp - temp_c) * 0.1)
                heating_load = self._calculate_heating_load(temp_diff, vehicle_specs)
                return heating_load / cop
            else:
                # Resistive heating
                heating_load = self._calculate_heating_load(temp_diff, vehicle_specs)
                return heating_load / self.constants['hvac_cop_resistive']
        else:
            # Cooling mode
            cooling_load = self._calculate_cooling_load(temp_diff, vehicle_specs)
            cop_cooling = max(2.0, 3.5 - (temp_c - target_temp) * 0.05)
            return cooling_load / cop_cooling


    def _calculate_heating_load(self, temp_diff: float, vehicle_specs: Dict) -> float:
        """Calculate heating load in watts"""
        
        # Base heating load
        base_load = 2000  # 2 kW base
        
        # Scale with temperature difference
        load_factor = min(3.0, 1 + temp_diff * 0.15)  # Max 3x at extreme cold
        
        # Vehicle size factor
        size_factor = vehicle_specs.get('mass', 1800) / 1800  # Normalize to average car
        
        return base_load * load_factor * size_factor
    
    def _calculate_cooling_load(self, temp_diff: float, vehicle_specs: Dict) -> float:
        """Calculate cooling load in watts"""
        
        # Base cooling load
        base_load = 3000  # 3 kW base (cooling typically needs more power)
        
        # Scale with temperature difference
        load_factor = min(2.5, 1 + temp_diff * 0.1)  # Max 2.5x at extreme heat
        
        # Vehicle size factor
        size_factor = vehicle_specs.get('mass', 1800) / 1800
        
        return base_load * load_factor * size_factor
    
    def _get_regenerative_efficiency(self, deceleration: float, speed_ms: float) -> float:
        """Calculate regenerative braking efficiency based on deceleration and speed"""
        
        # Base regenerative efficiency
        base_efficiency = self.constants['regen_efficiency']
        
        # Speed dependency: more efficient at higher speeds
        speed_factor = min(1.0, speed_ms / self.constants['regen_speed_threshold'])
        
        # Deceleration dependency: less efficient at very high deceleration
        if deceleration > self.constants['regen_hard_braking_threshold']:
            decel_factor = self.constants['regen_hard_braking_efficiency']
        elif deceleration > self.constants['regen_moderate_braking_threshold']:
            decel_factor = self.constants['regen_moderate_braking_efficiency']
        else:
            decel_factor = 1.0
        
        return base_efficiency * speed_factor * decel_factor

    
    def _estimate_battery_current(self, forces: Dict, speed_ms: float, vehicle_specs: Dict) -> float:
        """Estimate battery current for thermal loss calculation"""
        
        # Total mechanical power needed
        total_force = forces['rolling'] + forces['aero'] + max(0, forces['elevation'])
        mechanical_power_w = total_force * speed_ms
        
        # Convert to electrical power (account for efficiency losses)
        electrical_power_w = mechanical_power_w / 0.85  # 85% overall efficiency
        
        # Estimate current: P = V * I, so I = P / V
        nominal_voltage = vehicle_specs.get('nominal_voltage', 400)  # 400V typical
        current_a = electrical_power_w / nominal_voltage
        
        return abs(current_a)
    
    def _get_temperature_efficiency_factor(self, temp_c: float) -> float:
        """Get temperature efficiency factor using lookup table with interpolation"""
        
        # Round to nearest temperature in our lookup table
        temp_keys = sorted(self.temp_efficiency.keys())
        
        # If exact match, return it
        if temp_c in self.temp_efficiency:
            return self.temp_efficiency[temp_c]
        
        # Find bounds for interpolation
        if temp_c <= temp_keys[0]:
            return self.temp_efficiency[temp_keys[0]]
        elif temp_c >= temp_keys[-1]:
            return self.temp_efficiency[temp_keys[-1]]
        
        # Linear interpolation between two closest points
        for i in range(len(temp_keys) - 1):
            if temp_keys[i] <= temp_c <= temp_keys[i + 1]:
                t1, t2 = temp_keys[i], temp_keys[i + 1]
                eff1, eff2 = self.temp_efficiency[t1], self.temp_efficiency[t2]
                # Linear interpolation
                factor = (temp_c - t1) / (t2 - t1)
                return eff1 + factor * (eff2 - eff1)
        
        # Fallback
        return 1.0

    
    def _get_vehicle_specs(self, vehicle: Dict) -> Dict:
        """Extract and enhance vehicle specifications"""
        
        # Import EV models here to avoid circular imports
        from config.ev_config import EV_MODELS
        
        model_name = vehicle['model']
        base_specs = EV_MODELS.get(model_name, EV_MODELS['tesla_model_3'])
        battery_specs = self.battery_params.get(model_name, self.battery_params['tesla_model_3'])
        
        return {
            'mass': base_specs['weight'],
            'drag_coefficient': base_specs['drag_coefficient'],
            'frontal_area': base_specs['frontal_area'],
            'battery_capacity': vehicle['battery_capacity'],
            'base_internal_resistance': battery_specs['internal_resistance'],
            'nominal_voltage': battery_specs['nominal_voltage'],
            'auxiliary_power': self.constants['auxiliary_power'],
            'hvac_base_power': self.constants['hvac_base_power'],
            'has_heat_pump': battery_specs['has_heat_pump']
        }

    
    def _calculate_minimum_consumption(self, distance_km: float, vehicle_specs: Dict) -> float:
        """Calculate minimum realistic consumption to prevent zero values"""
        
        # Assume minimum driving time
        min_time_hours = max(self.constants['min_driving_time_hours'], distance_km / 60)
        
        # Auxiliary power consumption
        aux_consumption = vehicle_specs['auxiliary_power'] * min_time_hours
        
        # Add minimum drivetrain losses
        min_drivetrain_loss = distance_km * self.constants['min_consumption_per_km']
        
        return aux_consumption + min_drivetrain_loss

    
    def _get_zero_consumption_result(self, weather_conditions: Dict) -> Dict:
        """Return zero consumption result with proper structure"""
        
        return {
            'total_consumption_kwh': 0.0,
            'total_distance_km': 0.0,
            'efficiency_kwh_per_100km': 0.0,
            'temperature_celsius': weather_conditions.get('temperature', 20),
            'temperature_efficiency_factor': 1.0,
            'battery_internal_resistance_ohm': 0.1,
            'consumption_breakdown': {
                'rolling_resistance': 0.0,
                'aerodynamic_drag': 0.0,
                'elevation_change': 0.0,
                'acceleration': 0.0,
                'hvac': 0.0,
                'auxiliary': 0.0,
                'regenerative_braking': 0.0,
                'battery_thermal_loss': 0.0
            },
            'weather_conditions': weather_conditions,
            'model_version': 'advanced_physics_v1.0'
        }

