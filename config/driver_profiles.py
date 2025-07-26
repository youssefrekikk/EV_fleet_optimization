DRIVER_PROFILES = {
    'commuter': {
        'daily_km': (65, 130),           # km per day
        'trips_per_day': (2, 4),
        'avg_speed_city': 25,            # km/h
        'avg_speed_highway': 90,         # km/h
        'peak_hours': [(7, 9), (17, 19)],
        'home_charging_prob': 0.9,
        'charging_threshold': 0.3,       # charge when battery < 30%
        'driving_style': 'normal',
        'weekend_factor': 0.3,
        'proportion': 0.35
    },
    'rideshare': {
        'daily_km': (160, 320),          # km per day
        'trips_per_day': (15, 25),
        'avg_speed_city': 22,            # km/h (more stop-and-go)
        'avg_speed_highway': 85,         # km/h
        'peak_hours': [(6, 10), (16, 20), (21, 2)],
        'home_charging_prob': 0.6,
        'charging_threshold': 0.2,
        'driving_style': 'efficient',
        'weekend_factor': 1.2,
        'proportion': 0.25
    },
    'delivery': {
        'daily_km': (240, 480),          # km per day
        'trips_per_day': (20, 40),
        'avg_speed_city': 20,            # km/h (lots of stops)
        'avg_speed_highway': 80,         # km/h
        'peak_hours': [(9, 17)],
        'home_charging_prob': 0.3,
        'charging_threshold': 0.25,
        'driving_style': 'aggressive',
        'weekend_factor': 0.8,
        'proportion': 0.20
    },
    'casual': {
        'daily_km': (30, 100),           # km per day
        'trips_per_day': (1, 3),
        'avg_speed_city': 30,            # km/h (less traffic)
        'avg_speed_highway': 95,         # km/h
        'peak_hours': [(10, 16)],
        'home_charging_prob': 0.95,
        'charging_threshold': 0.4,
        'driving_style': 'eco',
        'weekend_factor': 1.5,
        'proportion': 0.2
    }
}
