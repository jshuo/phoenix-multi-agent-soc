"""
Kalman Filter Implementation using FilterPy
Professional-grade signal processing for IoT telemetry
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List
import logging

logger = logging.getLogger(__name__)


def apply_kalman_filter(
    measurements: List[float], 
    process_noise: float = 0.01,
    measurement_noise: float = 0.1
) -> List[float]:
    """
    Apply Kalman filter to a time series of measurements.
    
    This implementation uses a 2D state space model:
    - State[0]: Current value estimate
    - State[1]: Rate of change (velocity)
    
    Args:
        measurements: List of raw measurements (e.g., voltage readings)
        process_noise: Process noise covariance (Q) - how much the system changes
        measurement_noise: Measurement noise covariance (R) - sensor accuracy
    
    Returns:
        List of filtered values with reduced noise
    
    Example:
        >>> voltages = [3.7, 3.71, 3.69, 3.72, 3.70]
        >>> filtered = apply_kalman_filter(voltages)
        >>> print(filtered)  # Smoother values
    """
    if len(measurements) == 0:
        logger.warning("Empty measurement list provided to Kalman filter")
        return []
    
    if len(measurements) == 1:
        return measurements
    
    try:
        # Initialize Kalman filter
        # dim_x=2: [value, rate_of_change]
        # dim_z=1: single measurement (voltage, capacity, etc.)
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # Initial state: [first measurement, zero velocity]
        kf.x = np.array([measurements[0], 0.])
        
        # State transition matrix: simple constant velocity model
        # x_new = x_old + velocity * dt (where dt=1)
        kf.F = np.array([
            [1., 1.],  # value_new = value_old + velocity
            [0., 1.]   # velocity_new = velocity_old
        ])
        
        # Measurement matrix: we only observe the value, not velocity
        kf.H = np.array([[1., 0.]])
        
        # Initial uncertainty (high because we don't know the system yet)
        kf.P *= 1000.
        
        # Measurement noise covariance (sensor accuracy)
        kf.R = measurement_noise
        
        # Process noise covariance (how much we trust the model)
        kf.Q = np.eye(2) * process_noise
        
        # Process all measurements
        filtered = []
        for z in measurements:
            kf.predict()  # Predict next state
            kf.update(z)  # Update with measurement
            filtered.append(float(kf.x[0]))  # Extract filtered value
        
        logger.info(f"Kalman filter applied to {len(measurements)} measurements")
        return filtered
        
    except Exception as e:
        logger.error(f"Error in Kalman filter: {e}")
        # Fall back to original measurements on error
        return measurements


def calculate_noise_reduction(original: List[float], filtered: List[float]) -> dict:
    """
    Calculate noise reduction statistics.
    
    Args:
        original: Original noisy measurements
        filtered: Kalman-filtered measurements
    
    Returns:
        Dictionary with noise reduction metrics
    """
    if len(original) != len(filtered) or len(original) < 2:
        return {
            "varianceReduction": 0.0,
            "originalVariance": 0.0,
            "filteredVariance": 0.0
        }
    
    original_var = float(np.var(original))
    filtered_var = float(np.var(filtered))
    
    reduction_percent = 0.0
    if original_var > 0:
        reduction_percent = ((original_var - filtered_var) / original_var) * 100
    
    return {
        "varianceReduction": reduction_percent,
        "originalVariance": original_var,
        "filteredVariance": filtered_var,
        "originalStdDev": float(np.std(original)),
        "filteredStdDev": float(np.std(filtered))
    }


def apply_multi_variable_kalman(
    voltage_data: List[float],
    capacity_data: List[float],
    temperature_data: List[float]
) -> dict:
    """
    Apply Kalman filtering to multiple variables simultaneously.
    
    Args:
        voltage_data: Voltage measurements
        capacity_data: Capacity measurements
        temperature_data: Temperature measurements
    
    Returns:
        Dictionary with filtered values for each variable
    """
    result = {
        "voltage": apply_kalman_filter(voltage_data, 0.01, 0.1),
        "capacity": apply_kalman_filter(capacity_data, 0.005, 0.05),
        "temperature": apply_kalman_filter(temperature_data, 0.02, 0.15)
    }
    
    logger.info(f"Multi-variable Kalman filter applied to {len(voltage_data)} samples")
    return result
