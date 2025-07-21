# EV Fleet Logging System

A centralized logging system for the EV Fleet Optimization project that provides easy on/off switching and consistent logging across all modules.

## Features

- **Easy Mode Switching**: Switch between production, development, debug, silent, and testing modes
- **Module-Specific Control**: Enable/disable logging for specific modules
- **Detailed Logging**: Enable detailed logging for specific components (energy calculations, route generation, etc.)
- **Multiple Output Formats**: Console, file, or both
- **Consistent Interface**: Same logging interface across all modules

## Quick Start

### 1. Basic Usage

```python
from config.logging_config import *
from src.utils.logger import setup_logger, info, warning, error, debug

# Switch to desired mode
switch_to_debug()  # or switch_to_production(), switch_to_development(), etc.

# Setup logger
logger = setup_logger(**get_logging_config())

# Use logging functions
info("This is an info message", "module_name")
warning("This is a warning", "module_name")
error("This is an error", "module_name")
debug("This is a debug message", "module_name")
```

### 2. Module-Specific Control

```python
# Disable logging for specific modules
disable_module_logging('advanced_energy_model')
disable_module_logging('openchargemap_api')

# Enable logging for specific modules
enable_module_logging('synthetic_ev_generator')
```

### 3. Detailed Logging

```python
# Enable detailed logging for specific components
enable_detailed_logging('energy_calculation')
enable_detailed_logging('route_generation')

# Use detailed logging
from src.utils.logger import log_detailed
log_detailed("Detailed message", "component_name", "vehicle_id")
```

## Logging Modes

### Production Mode
- **Level**: WARNING
- **Console**: Yes
- **File**: No
- **Detailed Logging**: No
- **Format**: Minimal
- **Use Case**: Production deployment

### Development Mode
- **Level**: INFO
- **Console**: Yes
- **File**: Yes
- **Detailed Logging**: No
- **Format**: Simple
- **Use Case**: Daily development

### Debug Mode
- **Level**: DEBUG
- **Console**: Yes
- **File**: Yes
- **Detailed Logging**: Yes
- **Format**: Detailed
- **Use Case**: Debugging and analysis

### Silent Mode
- **Level**: CRITICAL
- **Console**: No
- **File**: No
- **Detailed Logging**: No
- **Format**: Minimal
- **Use Case**: Performance testing

### Testing Mode
- **Level**: DEBUG
- **Console**: No
- **File**: Yes
- **Detailed Logging**: Yes
- **Format**: Detailed
- **Use Case**: Automated testing

## Configuration

### Switching Modes

```python
# Quick mode switches
switch_to_production()
switch_to_development()
switch_to_debug()
switch_to_silent()
switch_to_testing()

# Get configuration for specific mode
config = get_logging_config('DEBUG')
logger = setup_logger(**config)
```

### Module-Specific Settings

```python
# Check if module logging is enabled
if is_module_logging_enabled('synthetic_ev_generator'):
    info("Module is enabled", "synthetic_ev_generator")

# Enable/disable specific modules
enable_module_logging('module_name')
disable_module_logging('module_name')
```

### Detailed Logging Components

```python
# Available components for detailed logging
DETAILED_LOGGING_COMPONENTS = {
    'energy_calculation': False,  # Very verbose energy calculations
    'route_generation': False,    # Route generation details
    'charging_sessions': False,   # Charging session details
    'infrastructure': False,      # Infrastructure management
    'gps_trace': False           # GPS trace generation
}

# Enable/disable detailed logging
enable_detailed_logging('energy_calculation')
disable_detailed_logging('route_generation')

# Check if detailed logging is enabled
if is_detailed_logging_enabled('energy_calculation'):
    log_detailed("Detailed energy calculation", "energy_calculation", "vehicle_id")
```

## File Structure

```
src/
├── utils/
│   └── logger.py              # Main logging module
config/
├── logging_config.py          # Logging configuration
examples/
├── logging_example.py         # Usage examples
debug_logs/                    # Log files directory
├── ev_fleet_YYYYMMDD_HHMMSS.log
├── energy_calculation_*.log
├── route_failures.log
└── ...
```

## Advanced Usage

### Custom Configuration

```python
# Custom logger configuration
custom_config = {
    'log_level': 'DEBUG',
    'enable_console': True,
    'enable_file': True,
    'detailed_logging': True,
    'log_format': 'detailed',
    'log_dir': 'custom_logs'
}

logger = setup_logger(**custom_config)
```

### Summary Printing

```python
from src.utils.logger import print_summary

# Print formatted summary
summary_data = {
    'Total Routes': 1250,
    'Total Distance (km)': 15420.5,
    'Average Efficiency': 18.5
}

print_summary("DATASET SUMMARY", summary_data)
```

### Logger Status

```python
from src.utils.logger import get_global_logger

logger = get_global_logger()
status = logger.get_status()

print("Current logger status:")
for key, value in status.items():
    print(f"  {key}: {value}")
```

## Migration from Old Logging

### Before (Old System)
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message")
logger.warning("Warning")
logger.error("Error")
```

### After (New System)
```python
from src.utils.logger import info, warning, error
info("Message", "module_name")
warning("Warning", "module_name")
error("Error", "module_name")
```

## Best Practices

1. **Use Module Names**: Always specify the module name when logging
2. **Choose Appropriate Mode**: Use production mode for deployment, debug mode for development
3. **Enable Detailed Logging Sparingly**: Detailed logging can be very verbose
4. **Check Module Status**: Verify if module logging is enabled before extensive logging
5. **Use Summary Printing**: Use `print_summary()` for clean, formatted output

## Troubleshooting

### No Logs Appearing
- Check if logging mode is set correctly
- Verify module logging is enabled
- Check if console/file output is enabled for current mode

### Too Many Logs
- Switch to production mode
- Disable detailed logging
- Disable specific module logging

### Performance Issues
- Use silent mode for performance testing
- Disable file logging if not needed
- Use minimal format for faster processing

## Example Scripts

Run the example script to see all features in action:

```bash
python examples/logging_example.py
```

This will demonstrate:
- All logging modes
- Module-specific control
- Detailed logging
- Summary printing
- Logger status 