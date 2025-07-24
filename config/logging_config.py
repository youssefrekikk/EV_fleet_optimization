"""
Logging configuration for EV Fleet Optimization project
Easy switching between different logging modes
"""

# =============================================================================
# LOGGING CONFIGURATIONS
# =============================================================================

# Production mode - minimal logging
PRODUCTION_LOGGING = {
    'log_level': 'WARNING',
    'enable_console': True,
    'enable_file': False,
    'detailed_logging': False,
    'log_format': 'minimal'
}

# Development mode - standard logging
DEVELOPMENT_LOGGING = {
    'log_level': 'INFO',
    'enable_console': True,
    'enable_file': True,
    'detailed_logging': False,
    'log_format': 'simple'
}

# Debug mode - detailed logging
DEBUG_LOGGING = {
    'log_level': 'DEBUG',
    'enable_console': True,
    'enable_file': True,
    'detailed_logging': True,
    'log_format': 'detailed'
}

# Silent mode - no logging
SILENT_LOGGING = {
    'log_level': 'CRITICAL',
    'enable_console': False,
    'enable_file': False,
    'detailed_logging': False,
    'log_format': 'minimal'
}

# Testing mode - file only logging
TESTING_LOGGING = {
    'log_level': 'DEBUG',
    'enable_console': False,
    'enable_file': True,
    'detailed_logging': True,
    'log_format': 'detailed'
}

# =============================================================================
# QUICK SWITCHES
# =============================================================================

# Change this to switch logging modes
CURRENT_LOGGING_MODE = 'DEBUG'  # Options: PRODUCTION, DEVELOPMENT, DEBUG, SILENT, TESTING

# =============================================================================
# MODULE-SPECIFIC LOGGING
# =============================================================================

# Enable/disable logging for specific modules
MODULE_LOGGING = {
    'synthetic_ev_generator': True,
    'advanced_energy_model': True,
    'openchargemap_api': True,
    'road_network_db': True,
    'charging_infrastructure': True
}

# =============================================================================
# DETAILED LOGGING SETTINGS
# =============================================================================

# Enable detailed logging for specific components
DETAILED_LOGGING_COMPONENTS = {
    'energy_calculation': False,  # Very verbose energy calculations
    'route_generation': False,    # Route generation details
    'charging_sessions': True,    # Charging session details
    'infrastructure': False,      # Infrastructure management
    'gps_trace': False           # GPS trace generation
}

# =============================================================================
# LOG FILE SETTINGS
# =============================================================================

LOG_DIR = "debug_logs"
MAX_LOG_FILE_SIZE_MB = 100
LOG_FILE_BACKUP_COUNT = 5

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_logging_config(mode: str = None) -> dict:
    """Get logging configuration for specified mode"""
    if mode is None:
        mode = CURRENT_LOGGING_MODE
    
    configs = {
        'PRODUCTION': PRODUCTION_LOGGING,
        'DEVELOPMENT': DEVELOPMENT_LOGGING,
        'DEBUG': DEBUG_LOGGING,
        'SILENT': SILENT_LOGGING,
        'TESTING': TESTING_LOGGING
    }
    
    return configs.get(mode, DEVELOPMENT_LOGGING)

def is_module_logging_enabled(module_name: str) -> bool:
    """Check if logging is enabled for a specific module"""
    return MODULE_LOGGING.get(module_name, True)

def is_detailed_logging_enabled(component: str) -> bool:
    """Check if detailed logging is enabled for a specific component"""
    return DETAILED_LOGGING_COMPONENTS.get(component, False)

def switch_to_production():
    """Switch to production logging mode"""
    global CURRENT_LOGGING_MODE
    CURRENT_LOGGING_MODE = 'PRODUCTION'

def switch_to_development():
    """Switch to development logging mode"""
    global CURRENT_LOGGING_MODE
    CURRENT_LOGGING_MODE = 'DEVELOPMENT'

def switch_to_debug():
    """Switch to debug logging mode"""
    global CURRENT_LOGGING_MODE
    CURRENT_LOGGING_MODE = 'DEBUG'

def switch_to_silent():
    """Switch to silent logging mode"""
    global CURRENT_LOGGING_MODE
    CURRENT_LOGGING_MODE = 'SILENT'

def switch_to_testing():
    """Switch to testing logging mode"""
    global CURRENT_LOGGING_MODE
    CURRENT_LOGGING_MODE = 'TESTING'

def enable_module_logging(module_name: str):
    """Enable logging for a specific module"""
    MODULE_LOGGING[module_name] = True

def disable_module_logging(module_name: str):
    """Disable logging for a specific module"""
    MODULE_LOGGING[module_name] = False

def enable_detailed_logging(component: str):
    """Enable detailed logging for a specific component"""
    DETAILED_LOGGING_COMPONENTS[component] = True

def disable_detailed_logging(component: str):
    """Disable detailed logging for a specific component"""
    DETAILED_LOGGING_COMPONENTS[component] = False

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
# Quick usage examples:

# 1. Switch logging mode
switch_to_debug()  # Enable detailed logging
switch_to_production()  # Minimal logging for production

# 2. Enable/disable specific modules
disable_module_logging('advanced_energy_model')  # Disable energy model logging
enable_module_logging('synthetic_ev_generator')  # Enable generator logging

# 3. Enable detailed logging for specific components
enable_detailed_logging('energy_calculation')  # Very verbose energy logs
enable_detailed_logging('route_generation')    # Route generation details

# 4. Get current configuration
config = get_logging_config()  # Get current mode config
""" 