# EV Fleet Logging System

Centralized logging with mode switches, module-level toggles, file/console handlers, and dedicated detailed logs for specific components.

## Architecture

- Config in `config/logging_config.py` defines modes, module toggles, and detailed components
- Runtime API in `src/utils/logger.py` exposes `info/warning/error/debug`, `print_summary`, `log_detailed`, and helpers to get/update the global logger

## Modes

- Production: WARNING, console only
- Development: INFO, console+file
- Debug: DEBUG, console+file, detailed logs enabled
- Silent: CRITICAL, no output
- Testing: DEBUG, file only, detailed logs enabled

Switch with `switch_to_production()`, `switch_to_development()`, `switch_to_debug()`, `switch_to_silent()`, `switch_to_testing()`; then call `setup_logger(**get_logging_config())`.

## Usage

```python
from config.logging_config import switch_to_debug, get_logging_config
from src.utils.logger import setup_logger, info, warning, error, debug, log_detailed, print_summary

# Configure and initialize
switch_to_debug()
logger = setup_logger(**get_logging_config())

info("Dataset generated", "synthetic_ev_generator")
log_detailed("Segment 42: hvac=0.05kWh", "energy_calculation", vehicle_id="V-001")
print_summary("RUN SUMMARY", {"trips": 1250, "avg_eff": 18.5})
```

### Module toggles

```python
from config.logging_config import enable_module_logging, disable_module_logging, is_module_logging_enabled
from src.utils.logger import info

disable_module_logging('openchargemap_api')
if is_module_logging_enabled('synthetic_ev_generator'):
    info("Generator logging enabled", "synthetic_ev_generator")
```

### Detailed components

Available keys (default may vary by mode): `energy_calculation`, `route_generation`, `charging_sessions`, `infrastructure`, `gps_trace`.

```python
from config.logging_config import enable_detailed_logging, is_detailed_logging_enabled
from src.utils.logger import log_detailed

enable_detailed_logging('energy_calculation')
if is_detailed_logging_enabled('energy_calculation'):
    log_detailed("Forces: rolling=120N,aero=80N", "energy_calculation", vehicle_id="V-001")
```

## Files and directories

- Default directory: `debug_logs/`
- Main rotating file: `ev_fleet_<timestamp>.log`
- Component logs: `energy_calculation_*.log` etc.
- Route failures: `route_failures.log` (see `log_route_failure`)

## Best practices

- Prefer production mode in benchmarks
- Enable detailed logs only when diagnosing specific components
- Always tag logs with a module name for filtering

## Troubleshooting

- No logs: ensure you called `setup_logger()` after switching mode; confirm module toggles
- Too noisy: switch to PRODUCTION or disable detailed components
- File missing: set `enable_file=True` in config and ensure write permissions for `LOG_DIR`