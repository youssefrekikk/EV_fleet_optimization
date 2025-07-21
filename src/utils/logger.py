"""
Centralized logging module for EV Fleet Optimization project
Provides easy on/off switching and consistent logging across all modules
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class EVFleetLogger:
    """
    Centralized logger for EV Fleet Optimization project
    Provides easy switching between different logging levels and outputs
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = False,
                 log_dir: str = "debug_logs",
                 detailed_logging: bool = False,
                 log_format: str = "detailed"):
        """
        Initialize the logger
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Whether to log to console
            enable_file: Whether to log to files
            log_dir: Directory for log files
            detailed_logging: Enable detailed logging for debugging
            log_format: Log format style ("simple", "detailed", "minimal")
        """
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.log_dir = log_dir
        self.detailed_logging = detailed_logging
        self.log_format = log_format
        
        # Create log directory if needed
        if self.enable_file:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
        
        # Track detailed logging files
        self.detailed_log_files = {}
        
    def _setup_loggers(self):
        """Setup all loggers with proper configuration"""
        
        # Create main logger
        self.logger = logging.getLogger('ev_fleet')
        self.logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        if self.log_format == "detailed":
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        elif self.log_format == "simple":
            formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
        else:  # minimal
            formatter = logging.Formatter('%(message)s')
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(self.log_dir, f'ev_fleet_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance for a specific module"""
        if name:
            return logging.getLogger(f'ev_fleet.{name}')
        return self.logger
    
    def debug(self, message: str, module: str = None):
        """Log debug message"""
        logger = self.get_logger(module)
        logger.debug(message)
    
    def info(self, message: str, module: str = None):
        """Log info message"""
        logger = self.get_logger(module)
        logger.info(message)
    
    def warning(self, message: str, module: str = None):
        """Log warning message"""
        logger = self.get_logger(module)
        logger.warning(message)
    
    def error(self, message: str, module: str = None):
        """Log error message"""
        logger = self.get_logger(module)
        logger.error(message)
    
    def critical(self, message: str, module: str = None):
        """Log critical message"""
        logger = self.get_logger(module)
        logger.critical(message)
    
    def print_summary(self, title: str, data: Dict[str, Any]):
        """Print a formatted summary"""
        if not self.enable_console:
            return
            
        print(f"\n{'='*50}")
        print(title)
        print(f"{'='*50}")
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if value >= 1000:
                    print(f"  {key}: {value:,}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        print(f"{'='*50}")
    
    def create_detailed_log_file(self, name: str, vehicle_id: str = "unknown") -> str:
        """Create a detailed log file for specific analysis"""
        if not self.detailed_logging:
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{vehicle_id}_{timestamp}.log"
        filepath = os.path.join(self.log_dir, filename)
        
        # Store reference
        self.detailed_log_files[name] = filepath
        
        # Create file with header
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Detailed Log: {name}\n")
            f.write(f"Vehicle ID: {vehicle_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
        
        return filepath
    
    def log_detailed(self, message: str, log_name: str, vehicle_id: str = "unknown"):
        """Log detailed message to specific log file"""
        if not self.detailed_logging:
            return
            
        if log_name not in self.detailed_log_files:
            self.create_detailed_log_file(log_name, vehicle_id)
        
        filepath = self.detailed_log_files[log_name]
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    
    def log_route_failure(self, origin, destination, origin_node, dest_node, 
                         network, reason: str, extra: str = None, vehicle_id: str = "unknown"):
        """Log route generation failures to a dedicated file"""
        if not self.enable_file:
            return
            
        failure_log_file = os.path.join(self.log_dir, "route_failures.log")
        
        with open(failure_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"ROUTE FAILURE: {datetime.now().isoformat()}\n")
            f.write(f"Vehicle ID: {vehicle_id}\n")
            f.write(f"Reason: {reason}\n")
            f.write(f"Origin: {origin}\n")
            f.write(f"Destination: {destination}\n")
            f.write(f"Origin node: {origin_node}\n")
            f.write(f"Destination node: {dest_node}\n")
            
            if origin_node is not None and dest_node is not None and network is not None:
                try:
                    node1_data = network.nodes[origin_node]
                    node2_data = network.nodes[dest_node]
                    f.write(f"Origin node coords: ({node1_data['y']}, {node1_data['x']})\n")
                    f.write(f"Dest node coords: ({node2_data['y']}, {node2_data['x']})\n")
                    
                    # Connected components
                    import networkx as nx
                    comp_map = {n: i for i, comp in enumerate(nx.connected_components(network.to_undirected())) for n in comp}
                    comp1 = comp_map.get(origin_node, 'N/A')
                    comp2 = comp_map.get(dest_node, 'N/A')
                    f.write(f"Origin node component: {comp1}\n")
                    f.write(f"Dest node component: {comp2}\n")
                except Exception as e:
                    f.write(f"Error logging node/component info: {e}\n")
            
            if extra:
                f.write(f"Extra: {extra}\n")
            f.write(f"{'='*60}\n")
    
    def update_config(self, **kwargs):
        """Update logger configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Re-setup loggers with new configuration
        self._setup_loggers()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current logger status"""
        return {
            'log_level': logging.getLevelName(self.log_level),
            'enable_console': self.enable_console,
            'enable_file': self.enable_file,
            'log_dir': self.log_dir,
            'detailed_logging': self.detailed_logging,
            'log_format': self.log_format,
            'log_file': self.log_file,
            'detailed_log_files': list(self.detailed_log_files.keys())
        }


# Global logger instance
_global_logger = None

def get_logger(name: str = None) -> logging.Logger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        # Initialize with default settings
        _global_logger = EVFleetLogger()
    return _global_logger.get_logger(name)

def setup_logger(**kwargs) -> EVFleetLogger:
    """Setup the global logger with custom configuration"""
    global _global_logger
    _global_logger = EVFleetLogger(**kwargs)
    return _global_logger

def get_global_logger() -> EVFleetLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EVFleetLogger()
    return _global_logger

# Convenience functions
def debug(message: str, module: str = None):
    """Log debug message"""
    get_global_logger().debug(message, module)

def info(message: str, module: str = None):
    """Log info message"""
    get_global_logger().info(message, module)

def warning(message: str, module: str = None):
    """Log warning message"""
    get_global_logger().warning(message, module)

def error(message: str, module: str = None):
    """Log error message"""
    get_global_logger().error(message, module)

def critical(message: str, module: str = None):
    """Log critical message"""
    get_global_logger().critical(message, module)

def print_summary(title: str, data: Dict[str, Any]):
    """Print a formatted summary"""
    get_global_logger().print_summary(title, data)

def log_detailed(message: str, log_name: str, vehicle_id: str = "unknown"):
    """Log detailed message"""
    get_global_logger().log_detailed(message, log_name, vehicle_id)

def log_route_failure(origin, destination, origin_node, dest_node, 
                     network, reason: str, extra: str = None, vehicle_id: str = "unknown"):
    """Log route failure"""
    get_global_logger().log_route_failure(origin, destination, origin_node, dest_node, 
                                         network, reason, extra, vehicle_id) 