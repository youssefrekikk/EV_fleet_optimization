import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

class DataAnalyticsService:
    """Service for analyzing generated EV fleet datasets"""
    
    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)
        self.expected_files = [
            'routes.csv', 'charging_sessions.csv', 'vehicle_states.csv', 
            'weather.csv', 'fleet_info.csv', 'segments.csv',
            'charging_stations.csv', 'charging_networks.csv'
        ]
    
    def get_available_files(self) -> Dict[str, Dict]:
        """Get information about available CSV files"""
        files_info = {}
        
        for filename in self.expected_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    # Get basic file info
                    file_size = filepath.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    
                    # Try to read a sample to get column info
                    try:
                        # Check if file is actually empty by reading first few lines
                        with open(filepath, 'r') as f:
                            first_lines = [f.readline().strip() for _ in range(3)]
                        
                        # If all lines are empty or file is empty, mark as empty
                        if not any(first_lines) or all(line == '' for line in first_lines):
                            files_info[filename] = {
                                'exists': True,
                                'file_size_mb': round(file_size_mb, 2),
                                'error': "File appears to be empty",
                                'last_modified': datetime.fromtimestamp(filepath.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            }
                        else:
                            df_sample = pd.read_csv(filepath, nrows=5)
                            columns = list(df_sample.columns)
                            data_types = df_sample.dtypes.to_dict()
                            
                            # Get full dataset info
                            df_full = pd.read_csv(filepath)
                            total_rows = len(df_full)
                            total_columns = len(df_full.columns)
                            
                            # Calculate memory usage
                            memory_usage_mb = df_full.memory_usage(deep=True).sum() / (1024 * 1024)
                            
                            files_info[filename] = {
                                'exists': True,
                                'file_size_mb': round(file_size_mb, 2),
                                'total_rows': total_rows,
                                'total_columns': total_columns,
                                'memory_usage_mb': round(memory_usage_mb, 2),
                                'columns': columns,
                                'data_types': {col: str(dtype) for col, dtype in data_types.items()},
                                'last_modified': datetime.fromtimestamp(filepath.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            }
                        
                    except Exception as e:
                        files_info[filename] = {
                            'exists': True,
                            'file_size_mb': round(file_size_mb, 2),
                            'error': f"Could not read file: {str(e)}",
                            'last_modified': datetime.fromtimestamp(filepath.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                except Exception as e:
                    files_info[filename] = {
                        'exists': True,
                        'error': f"Could not analyze file: {str(e)}"
                    }
            else:
                files_info[filename] = {
                    'exists': False
                }
        
        return files_info
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary"""
        files_info = self.get_available_files()
        
        total_files = len([f for f in files_info.values() if f.get('exists', False)])
        total_rows = sum(f.get('total_rows', 0) for f in files_info.values() if f.get('exists', False))
        total_size_mb = sum(f.get('file_size_mb', 0) for f in files_info.values() if f.get('exists', False))
        
        return {
            'total_files': total_files,
            'total_rows': total_rows,
            'total_size_mb': round(total_size_mb, 2),
            'files_info': files_info
        }
    
    def get_column_analysis(self, filename: str) -> Dict:
        """Get detailed analysis of a specific file's columns"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return {'error': f'File {filename} not found'}
        
        try:
            df = pd.read_csv(filepath)
            
            column_analysis = {}
            for col in df.columns:
                col_data = df[col]
                dtype = str(col_data.dtype)
                
                analysis = {
                    'dtype': dtype,
                    'total_count': len(col_data),
                    'non_null_count': col_data.count(),
                    'null_count': col_data.isnull().sum(),
                    'null_percentage': round((col_data.isnull().sum() / len(col_data)) * 100, 2)
                }
                
                # Add type-specific analysis
                if dtype in ['int64', 'float64']:
                    analysis.update({
                        'min': float(col_data.min()) if not col_data.empty else None,
                        'max': float(col_data.max()) if not col_data.empty else None,
                        'mean': float(col_data.mean()) if not col_data.empty else None,
                        'std': float(col_data.std()) if not col_data.empty else None,
                        'unique_values': col_data.nunique()
                    })
                elif dtype == 'object':
                    analysis.update({
                        'unique_values': col_data.nunique(),
                        'most_common': col_data.value_counts().head(3).to_dict() if not col_data.empty else {},
                        'avg_length': col_data.astype(str).str.len().mean() if not col_data.empty else 0
                    })
                elif dtype == 'bool':
                    analysis.update({
                        'true_count': int(col_data.sum()) if not col_data.empty else 0,
                        'false_count': int((~col_data).sum()) if not col_data.empty else 0,
                        'true_percentage': round((col_data.sum() / len(col_data)) * 100, 2) if not col_data.empty else 0
                    })
                
                column_analysis[col] = analysis
            
            return {
                'filename': filename,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'column_analysis': column_analysis
            }
            
        except Exception as e:
            return {'error': f'Error analyzing {filename}: {str(e)}'}
    
    def get_fleet_analytics(self) -> Dict:
        """Get fleet-specific analytics"""
        try:
            # Load key datasets with error handling
            datasets = {}
            
            for dataset_name in ['fleet_info.csv', 'vehicle_states.csv', 'routes.csv', 'charging_sessions.csv']:
                filepath = self.data_dir / dataset_name
                if filepath.exists():
                    try:
                        # Check if file is not empty
                        with open(filepath, 'r') as f:
                            first_line = f.readline().strip()
                        
                        if first_line:  # File has content
                            datasets[dataset_name.replace('.csv', '')] = pd.read_csv(filepath)
                        else:
                            datasets[dataset_name.replace('.csv', '')] = pd.DataFrame()
                    except Exception as e:
                        print(f"Error loading {dataset_name}: {e}")
                        datasets[dataset_name.replace('.csv', '')] = pd.DataFrame()
                else:
                    datasets[dataset_name.replace('.csv', '')] = pd.DataFrame()
            
            fleet_info = datasets.get('fleet_info', pd.DataFrame())
            vehicle_states = datasets.get('vehicle_states', pd.DataFrame())
            routes = datasets.get('routes', pd.DataFrame())
            charging_sessions = datasets.get('charging_sessions', pd.DataFrame())
            
            analytics = {}
            
            # Fleet composition
            if not fleet_info.empty:
                analytics['fleet_composition'] = {
                    'total_vehicles': len(fleet_info),
                    'vehicle_models': fleet_info['model'].value_counts().to_dict(),
                    'driver_profiles': fleet_info['driver_profile'].value_counts().to_dict(),
                    'home_charging_availability': {
                        'with_home_charging': int(fleet_info['has_home_charging'].sum()),
                        'without_home_charging': int((~fleet_info['has_home_charging']).sum())
                    }
                }
            
            # Vehicle performance
            if not vehicle_states.empty:
                analytics['vehicle_performance'] = {
                    'total_distance_km': float(vehicle_states['total_distance_km'].sum()),
                    'total_consumption_kwh': float(vehicle_states['total_consumption_kwh'].sum()),
                    'avg_efficiency_kwh_per_100km': float(vehicle_states['efficiency_kwh_per_100km'].mean()),
                    'total_trips': int(vehicle_states['num_trips'].sum()),
                    'total_charging_sessions': int(vehicle_states['num_charging_sessions'].sum()),
                    'avg_trips_per_vehicle': float(vehicle_states['num_trips'].mean()),
                    'avg_charging_sessions_per_vehicle': float(vehicle_states['num_charging_sessions'].mean())
                }
            
            # Route analysis
            if not routes.empty:
                # Handle different column names that might exist
                duration_col = 'total_duration_minutes' if 'total_duration_minutes' in routes.columns else 'total_time_minutes'
                
                analytics['route_analysis'] = {
                    'total_routes': len(routes),
                    'avg_distance_km': float(routes['total_distance_km'].mean()),
                    'avg_duration_minutes': float(routes[duration_col].mean()) if duration_col in routes.columns else 0.0,
                    'avg_consumption_kwh': float(routes['total_consumption_kwh'].mean()),
                    'distance_distribution': {
                        'short_trips_0_10km': int(len(routes[routes['total_distance_km'] <= 10])),
                        'medium_trips_10_50km': int(len(routes[(routes['total_distance_km'] > 10) & (routes['total_distance_km'] <= 50)])),
                        'long_trips_50km_plus': int(len(routes[routes['total_distance_km'] > 50]))
                    }
                }
            
            # Charging analysis
            if not charging_sessions.empty:
                # Handle different column names that might exist
                duration_col = 'duration_minutes' if 'duration_minutes' in charging_sessions.columns else 'duration_hours'
                
                analytics['charging_analysis'] = {
                    'total_sessions': len(charging_sessions),
                    'home_charging_sessions': int(len(charging_sessions[charging_sessions['charging_type'] == 'home'])),
                    'public_charging_sessions': int(len(charging_sessions[charging_sessions['charging_type'] == 'public'])),
                    'total_energy_delivered_kwh': float(charging_sessions['energy_delivered_kwh'].sum()),
                    'total_cost_usd': float(charging_sessions['cost_usd'].sum()),
                    'avg_session_duration_minutes': float(charging_sessions[duration_col].mean() * 60) if duration_col == 'duration_hours' else float(charging_sessions[duration_col].mean()),
                    'avg_energy_per_session_kwh': float(charging_sessions['energy_delivered_kwh'].mean()),
                    'avg_cost_per_session_usd': float(charging_sessions['cost_usd'].mean())
                }
            
            return analytics
            
        except Exception as e:
            return {'error': f'Error generating fleet analytics: {str(e)}'}
    
    def create_visualizations(self) -> Dict:
        """Create various visualizations for the dataset"""
        try:
            viz_data = {}
            
            # Load datasets with error handling
            datasets = {}
            
            for dataset_name in ['vehicle_states.csv', 'routes.csv', 'charging_sessions.csv', 'fleet_info.csv']:
                filepath = self.data_dir / dataset_name
                if filepath.exists():
                    try:
                        # Check if file is not empty
                        with open(filepath, 'r') as f:
                            first_line = f.readline().strip()
                        
                        if first_line:  # File has content
                            datasets[dataset_name.replace('.csv', '')] = pd.read_csv(filepath)
                        else:
                            datasets[dataset_name.replace('.csv', '')] = pd.DataFrame()
                    except Exception as e:
                        print(f"Error loading {dataset_name}: {e}")
                        datasets[dataset_name.replace('.csv', '')] = pd.DataFrame()
                else:
                    datasets[dataset_name.replace('.csv', '')] = pd.DataFrame()
            
            vehicle_states = datasets.get('vehicle_states', pd.DataFrame())
            routes = datasets.get('routes', pd.DataFrame())
            charging_sessions = datasets.get('charging_sessions', pd.DataFrame())
            fleet_info = datasets.get('fleet_info', pd.DataFrame())
            
            # 1. Vehicle Efficiency Distribution
            if not vehicle_states.empty:
                fig_efficiency = px.histogram(
                    vehicle_states, 
                    x='efficiency_kwh_per_100km',
                    nbins=20,
                    title='Vehicle Energy Efficiency Distribution',
                    labels={'efficiency_kwh_per_100km': 'Efficiency (kWh/100km)', 'count': 'Number of Vehicles'}
                )
                fig_efficiency.update_layout(height=400)
                viz_data['efficiency_distribution'] = fig_efficiency
            
            # 2. Distance vs Consumption Scatter
            if not routes.empty:
                # Check if avg_speed_kmh exists, otherwise use a different color or no color
                color_col = 'avg_speed_kmh' if 'avg_speed_kmh' in routes.columns else None
                
                if color_col:
                    fig_distance_consumption = px.scatter(
                        routes,
                        x='total_distance_km',
                        y='total_consumption_kwh',
                        color=color_col,
                        title='Distance vs Energy Consumption',
                        labels={
                            'total_distance_km': 'Distance (km)',
                            'total_consumption_kwh': 'Energy Consumption (kWh)',
                            'avg_speed_kmh': 'Average Speed (km/h)'
                        }
                    )
                else:
                    fig_distance_consumption = px.scatter(
                        routes,
                        x='total_distance_km',
                        y='total_consumption_kwh',
                        title='Distance vs Energy Consumption',
                        labels={
                            'total_distance_km': 'Distance (km)',
                            'total_consumption_kwh': 'Energy Consumption (kWh)'
                        }
                    )
                
                fig_distance_consumption.update_layout(height=400)
                viz_data['distance_consumption_scatter'] = fig_distance_consumption
            
            # 3. Charging Session Analysis
            if not charging_sessions.empty:
                fig_charging = px.box(
                    charging_sessions,
                    x='charging_type',
                    y='energy_delivered_kwh',
                    title='Energy Delivered by Charging Type',
                    labels={'charging_type': 'Charging Type', 'energy_delivered_kwh': 'Energy (kWh)'}
                )
                fig_charging.update_layout(height=400)
                viz_data['charging_analysis'] = fig_charging
            
            # 4. Fleet Composition
            if not fleet_info.empty:
                # Vehicle models
                model_counts = fleet_info['model'].value_counts()
                fig_models = px.pie(
                    values=model_counts.values,
                    names=model_counts.index,
                    title='Fleet Composition by Vehicle Model'
                )
                fig_models.update_layout(height=400)
                viz_data['fleet_models'] = fig_models
                
                # Driver profiles
                profile_counts = fleet_info['driver_profile'].value_counts()
                fig_profiles = px.bar(
                    x=profile_counts.index,
                    y=profile_counts.values,
                    title='Driver Profile Distribution',
                    labels={'x': 'Driver Profile', 'y': 'Number of Drivers'}
                )
                fig_profiles.update_layout(height=400)
                viz_data['driver_profiles'] = fig_profiles
            
            # 5. Daily Activity Pattern
            if not routes.empty and 'date' in routes.columns:
                routes['date'] = pd.to_datetime(routes['date'])
                # Handle different column names for trip_id
                trip_id_col = 'trip_id' if 'trip_id' in routes.columns else 'vehicle_id'
                
                daily_activity = routes.groupby('date').agg({
                    trip_id_col: 'count',
                    'total_distance_km': 'sum',
                    'total_consumption_kwh': 'sum'
                }).reset_index()
                
                fig_daily = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Daily Trip Count', 'Daily Distance and Consumption'),
                    vertical_spacing=0.1
                )
                
                fig_daily.add_trace(
                    go.Scatter(x=daily_activity['date'], y=daily_activity[trip_id_col], 
                              mode='lines+markers', name='Trips'),
                    row=1, col=1
                )
                
                fig_daily.add_trace(
                    go.Scatter(x=daily_activity['date'], y=daily_activity['total_distance_km'], 
                              mode='lines+markers', name='Distance (km)'),
                    row=2, col=1
                )
                
                fig_daily.add_trace(
                    go.Scatter(x=daily_activity['date'], y=daily_activity['total_consumption_kwh'], 
                              mode='lines+markers', name='Consumption (kWh)'),
                    row=2, col=1
                )
                
                fig_daily.update_layout(height=600, title_text="Daily Activity Patterns")
                viz_data['daily_activity'] = fig_daily
            
            return viz_data
            
        except Exception as e:
            return {'error': f'Error creating visualizations: {str(e)}'}
    
    def get_data_quality_report(self) -> Dict:
        """Generate a comprehensive data quality report"""
        files_info = self.get_available_files()
        
        quality_report = {
            'overall_score': 0,
            'files_analyzed': 0,
            'total_issues': 0,
            'file_reports': {}
        }
        
        total_score = 0
        total_files = 0
        
        for filename, file_info in files_info.items():
            if not file_info.get('exists', False):
                continue
            
            total_files += 1
            file_score = 100
            issues = []
            
            # Check for errors
            if 'error' in file_info:
                issues.append(f"File read error: {file_info['error']}")
                file_score -= 50
            
            # Check for null values
            if 'columns' in file_info:
                column_analysis = self.get_column_analysis(filename)
                if 'column_analysis' in column_analysis:
                    for col, analysis in column_analysis['column_analysis'].items():
                        null_pct = analysis.get('null_percentage', 0)
                        if null_pct > 20:
                            issues.append(f"High null values in {col}: {null_pct}%")
                            file_score -= 10
                        elif null_pct > 5:
                            issues.append(f"Moderate null values in {col}: {null_pct}%")
                            file_score -= 5
            
            # Check file size
            if file_info.get('file_size_mb', 0) == 0:
                issues.append("Empty file")
                file_score -= 30
            
            # Check row count
            if file_info.get('total_rows', 0) == 0:
                issues.append("No data rows")
                file_score -= 30
            
            file_score = max(0, file_score)
            total_score += file_score
            
            quality_report['file_reports'][filename] = {
                'score': file_score,
                'issues': issues,
                'file_size_mb': file_info.get('file_size_mb', 0),
                'total_rows': file_info.get('total_rows', 0),
                'total_columns': file_info.get('total_columns', 0)
            }
        
        quality_report['files_analyzed'] = total_files
        quality_report['overall_score'] = round(total_score / max(total_files, 1), 1)
        quality_report['total_issues'] = sum(len(report['issues']) for report in quality_report['file_reports'].values())
        
        return quality_report
    
    def get_dataset_schema(self) -> Dict:
        """Get the schema (column names) of all available datasets"""
        schema = {}
        
        for filename in self.expected_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    # Check if file has content
                    with open(filepath, 'r') as f:
                        first_line = f.readline().strip()
                    
                    if first_line:
                        df_sample = pd.read_csv(filepath, nrows=1)
                        schema[filename] = {
                            'columns': list(df_sample.columns),
                            'dtypes': {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
                        }
                    else:
                        schema[filename] = {'error': 'Empty file'}
                except Exception as e:
                    schema[filename] = {'error': str(e)}
            else:
                schema[filename] = {'error': 'File not found'}
        
        return schema

# Global instance
data_analytics_service = DataAnalyticsService()
