"""
Analytics Package
"""

from .kalman import apply_kalman_filter, calculate_noise_reduction, apply_multi_variable_kalman
from .zscore import calculate_zscore, analyze_time_series_anomalies, detect_outliers_iqr
from .weather import WeatherService, get_location_for_region, get_regional_weather
from .rules import evaluate_all_rules, BATTERY_ALERT_RULES

__all__ = [
    "apply_kalman_filter",
    "calculate_noise_reduction",
    "apply_multi_variable_kalman",
    "calculate_zscore",
    "analyze_time_series_anomalies",
    "detect_outliers_iqr",
    "WeatherService",
    "get_location_for_region",
    "get_regional_weather",
    "evaluate_all_rules",
    "BATTERY_ALERT_RULES"
]
