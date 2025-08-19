"""
Optimization Configuration Service for the Streamlit Dashboard.

This service provides a centralized way to manage optimization parameters
with proper validation and explanations for each parameter.
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import streamlit as st

class OptimizationConfigService:
    """Service for managing optimization configuration parameters."""
    
    def __init__(self):
        self.config_file = Path("config/ui_overrides.yaml")
        self.default_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default optimization configuration."""
        return {
            # Routing Algorithm Configuration
            "routing": {
                "algorithm": {
                    "value": "astar",
                    "options": ["dijkstra", "astar","custom_astar"],
                    "description": "Routing algorithm to use for pathfinding",
                    "help": "Dijkstra: Guaranteed optimal but slower. A*: Faster with admissible heuristic."
                },
                "gamma_time_weight": {
                    "value": 0.02,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "description": "Time penalty weight (kWh per hour)",
                    "help": "Converts time penalties to energy equivalents. Higher values prioritize faster routes."
                }
            },
            
            # Charging and Cost Optimization
            "charging": {
                "price_weight_kwh_per_usd": {
                    "value": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "description": "Price sensitivity (kWh penalty per USD)",
                    "help": "How much to penalize expensive charging. Higher values prioritize cheaper charging."
                },
                "battery_buffer_percentage": {
                    "value": 0.15,
                    "min": 0.05,
                    "max": 0.30,
                    "step": 0.01,
                    "description": "Battery safety buffer (%)",
                    "help": "Minimum SOC to maintain during planning. Higher values provide more safety margin."
                },
                "max_detour_for_charging_km": {
                    "value": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "description": "Max detour for charging (km)",
                    "help": "Maximum distance to deviate from route for charging. Lower values keep routes direct."
                }
            },
            
            # Planning and Horizon Configuration
            "planning": {
                "prediction_horizon_hours": {
                    "value": 24,
                    "min": 1,
                    "max": 168,
                    "step": 1,
                    "description": "Prediction horizon (hours)",
                    "help": "How far ahead to plan. Longer horizons provide better optimization but slower computation."
                },
                "reoptimization_frequency_hours": {
                    "value": 4,
                    "min": 1,
                    "max": 24,
                    "step": 1,
                    "description": "Reoptimization frequency (hours)",
                    "help": "How often to recalculate optimal routes. More frequent updates adapt to changing conditions."
                },
                "planning_mode": {
                    "value": "myopic",
                    "options": ["myopic", "next_trip", "rolling_horizon"],
                    "description": "Planning strategy",
                    "help": "Myopic: No future planning. Next trip: Reserve for next trip. Rolling horizon: Plan ahead K trips."
                },
                "reserve_soc": {
                    "value": 0.15,
                    "min": 0.05,
                    "max": 0.40,
                    "step": 0.01,
                    "description": "SOC reserve fraction",
                    "help": "Fraction of battery to reserve for future needs. Higher values provide more flexibility."
                },
                "reserve_kwh": {
                    "value": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "description": "Absolute SOC reserve (kWh)",
                    "help": "Absolute energy to reserve (overrides reserve_soc if > 0). Useful for known future requirements."
                },
                "horizon_trips": {
                    "value": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "description": "Look-ahead trips",
                    "help": "Number of future trips to consider in planning. Higher values provide better optimization."
                },
                "horizon_hours": {
                    "value": 0.0,
                    "min": 0.0,
                    "max": 168.0,
                    "step": 1.0,
                    "description": "Time horizon (hours)",
                    "help": "Alternative to horizon_trips: time-based look-ahead. 0 means use horizon_trips instead."
                }
            },
            
            # Fleet Evaluation Settings
            "evaluation": {
                "fleet_eval_max_days": {
                    "value": 1,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "description": "Max days to evaluate",
                    "help": "Maximum number of days to process for fleet evaluation. Higher values provide more comprehensive analysis."
                },
                "fleet_eval_trip_sample_frac": {
                    "value": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Trip sampling fraction",
                    "help": "Fraction of trips to sample for evaluation. Lower values speed up analysis but reduce accuracy."
                }
            },
            
            # SOC Routing Objective
            "soc_routing": {
                "soc_objective": {
                    "value": "energy",
                    "options": ["energy", "cost", "time", "weighted"],
                    "description": "SOC routing objective",
                    "help": "Energy: Minimize consumption. Cost: Minimize charging costs. Time: Minimize travel time. Weighted: Balance all factors."
                },
                "alpha_usd_per_hour": {
                    "value": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "description": "Value of time (USD/hour)",
                    "help": "How much time is worth in monetary terms. Used for cost and weighted objectives."
                },
                "beta_kwh_to_usd": {
                    "value": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Energy to cost conversion",
                    "help": "Conversion factor from kWh to USD. Used for weighted objectives to balance energy and cost."
                }
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load the current optimization configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    overrides = yaml.safe_load(f) or {}
                return overrides.get('optimization', {})
            except Exception as e:
                st.warning(f"Could not load optimization config: {e}")
                return {}
        return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save the optimization configuration."""
        try:
            # Load existing overrides
            overrides = {}
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    overrides = yaml.safe_load(f) or {}
            
            # Update optimization section
            overrides['optimization'] = config
            
            # Save back to file
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(overrides, f, default_flow_style=False, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Could not save optimization config: {e}")
            return False
    
    def get_merged_config(self) -> Dict[str, Any]:
        """Get the merged configuration (defaults + overrides)."""
        overrides = self.load_config()
        merged = self._deep_merge(self.default_config, overrides)
        return merged
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, preserving structure."""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_parameter_value(self, section: str, param: str) -> Any:
        """Get the current value of a specific parameter."""
        config = self.get_merged_config()
        return config.get(section, {}).get(param, {}).get('value')
    
    def update_parameter(self, section: str, param: str, value: Any) -> bool:
        """Update a specific parameter value."""
        config = self.load_config()
        
        if section not in config:
            config[section] = {}
        if param not in config[section]:
            config[section][param] = {}
        
        config[section][param]['value'] = value
        
        return self.save_config(config)
    
    def reset_to_defaults(self) -> bool:
        """Reset all optimization parameters to default values."""
        return self.save_config({})
    
    def get_parameter_info(self, section: str, param: str) -> Dict[str, Any]:
        """Get detailed information about a parameter."""
        config = self.get_merged_config()
        return config.get(section, {}).get(param, {})
    
    def validate_parameter(self, section: str, param: str, value: Any) -> tuple[bool, str]:
        """Validate a parameter value."""
        info = self.get_parameter_info(section, param)
        
        if not info:
            return False, "Parameter not found"
        
        # Check min/max constraints
        if 'min' in info and value < info['min']:
            return False, f"Value must be at least {info['min']}"
        
        if 'max' in info and value > info['max']:
            return False, f"Value must be at most {info['max']}"
        
        # Check options for enum parameters
        if 'options' in info and value not in info['options']:
            return False, f"Value must be one of: {', '.join(info['options'])}"
        
        return True, "Valid"

# Global instance
optimization_config_service = OptimizationConfigService()
