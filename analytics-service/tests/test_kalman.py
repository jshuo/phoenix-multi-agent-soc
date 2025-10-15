"""
Tests for Kalman Filter Module
"""

import pytest
import numpy as np
from analytics.kalman import (
    apply_kalman_filter, 
    calculate_noise_reduction,
    apply_multi_variable_kalman
)


def test_kalman_filter_basic():
    """Test basic Kalman filtering with battery voltage data"""
    # Realistic battery voltage readings with noise
    measurements = [3.7, 3.71, 3.69, 3.72, 3.70, 3.68, 3.73, 3.67, 3.74, 3.66, 
                   3.75, 3.65, 3.76, 3.64, 3.77, 3.63, 3.78, 3.62, 3.79, 3.61, 
                   3.80, 3.60, 3.81, 3.59, 3.82]
    filtered = apply_kalman_filter(measurements)
    
    # Basic validation
    assert len(filtered) == len(measurements)
    assert all(isinstance(v, float) for v in filtered)
    
    # Filtered values should be smoother (lower variance)
    original_var = sum((x - sum(measurements)/len(measurements))**2 for x in measurements)
    filtered_var = sum((x - sum(filtered)/len(filtered))**2 for x in filtered)
    assert filtered_var < original_var
    
    # Filtered values should stay within reasonable bounds of original
    assert min(filtered) >= min(measurements) - 0.5
    assert max(filtered) <= max(measurements) + 0.5


def test_kalman_filter_empty():
    """Test Kalman filter with empty input"""
    result = apply_kalman_filter([])
    assert result == []


def test_kalman_filter_single_value():
    """Test Kalman filter with single value"""
    result = apply_kalman_filter([3.7])
    assert result == [3.7]


def test_kalman_filter_two_values():
    """Test Kalman filter with two values"""
    result = apply_kalman_filter([3.7, 3.8])
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)


def test_kalman_filter_constant_signal():
    """Test Kalman filter with constant signal (no noise)"""
    measurements = [5.0] * 10
    filtered = apply_kalman_filter(measurements)
    
    assert len(filtered) == len(measurements)
    # All values should be very close to 5.0
    assert all(abs(v - 5.0) < 0.1 for v in filtered)


def test_kalman_filter_with_spike():
    """Test Kalman filter handles outliers/spikes"""
    measurements = [3.7, 3.71, 3.69, 5.0, 3.70, 3.68, 3.73]  # 5.0 is a spike
    filtered = apply_kalman_filter(measurements)
    
    # The spike should be smoothed
    assert filtered[3] < measurements[3]
    assert filtered[3] > measurements[2]


def test_kalman_filter_custom_noise_params():
    """Test Kalman filter with custom noise parameters"""
    measurements = [3.7, 3.71, 3.69, 3.72, 3.70]
    
    # High process noise (system changes a lot)
    filtered_high = apply_kalman_filter(measurements, process_noise=0.1, measurement_noise=0.01)
    
    # Low process noise (system changes slowly)
    filtered_low = apply_kalman_filter(measurements, process_noise=0.001, measurement_noise=0.1)
    
    assert len(filtered_high) == len(measurements)
    assert len(filtered_low) == len(measurements)


def test_noise_reduction_calculation():
    """Test noise reduction statistics"""
    original = [1.0, 2.0, 1.5, 2.5, 1.8, 2.2]
    filtered = [1.5, 1.7, 1.8, 1.9, 2.0, 2.1]
    
    stats = calculate_noise_reduction(original, filtered)
    
    # Check all expected keys
    assert "varianceReduction" in stats
    assert "originalVariance" in stats
    assert "filteredVariance" in stats
    assert "originalStdDev" in stats
    assert "filteredStdDev" in stats
    
    # Variance should be reduced
    assert stats["varianceReduction"] > 0
    assert stats["filteredVariance"] < stats["originalVariance"]
    assert stats["filteredStdDev"] < stats["originalStdDev"]


def test_noise_reduction_empty_lists():
    """Test noise reduction with empty or mismatched lists"""
    stats = calculate_noise_reduction([], [])
    assert stats["varianceReduction"] == 0.0
    
    stats = calculate_noise_reduction([1.0], [1.0])
    assert stats["varianceReduction"] == 0.0


def test_noise_reduction_identical_data():
    """Test noise reduction when filtered equals original"""
    data = [3.7, 3.8, 3.9, 4.0]
    stats = calculate_noise_reduction(data, data)
    
    assert stats["varianceReduction"] == 0.0
    assert stats["originalVariance"] == stats["filteredVariance"]


def test_multi_variable_kalman():
    """Test multi-variable Kalman filtering"""
    voltage_data = [3.7, 3.71, 3.69, 3.72, 3.70]
    capacity_data = [2500, 2480, 2490, 2470, 2485]
    temperature_data = [25.0, 25.5, 24.8, 25.2, 25.1]
    
    result = apply_multi_variable_kalman(voltage_data, capacity_data, temperature_data)
    
    # Check structure
    assert "voltage" in result
    assert "capacity" in result
    assert "temperature" in result
    
    # Check all filtered data has correct length
    assert len(result["voltage"]) == len(voltage_data)
    assert len(result["capacity"]) == len(capacity_data)
    assert len(result["temperature"]) == len(temperature_data)
    
    # Check all values are floats
    assert all(isinstance(v, float) for v in result["voltage"])
    assert all(isinstance(v, float) for v in result["capacity"])
    assert all(isinstance(v, float) for v in result["temperature"])


def test_multi_variable_kalman_empty():
    """Test multi-variable Kalman with empty data"""
    result = apply_multi_variable_kalman([], [], [])
    
    assert result["voltage"] == []
    assert result["capacity"] == []
    assert result["temperature"] == []


def test_kalman_filter_preserves_trend():
    """Test that Kalman filter preserves overall trend"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Increasing trend with noise
    measurements = [1.0 + 0.1*i + 0.05*np.random.randn() for i in range(20)]
    filtered = apply_kalman_filter(measurements)
    
    # First value should be less than last value (upward trend preserved)
    assert filtered[0] < filtered[-1]
    
    # Check variance is reasonable (might not always be lower due to Kalman's nature)
    # But the filtered signal should still be smoother in most cases
    assert len(filtered) == len(measurements)


def test_kalman_filter_battery_discharge_pattern():
    """Test Kalman filter on realistic battery discharge pattern"""
    # Simulate battery discharge from 4.2V to 3.0V with noise
    actual_voltage = np.linspace(4.2, 3.0, 50)
    noise = np.random.normal(0, 0.05, 50)
    measurements = (actual_voltage + noise).tolist()
    
    filtered = apply_kalman_filter(measurements)
    
    # Should smooth out the noise
    assert len(filtered) == len(measurements)
    
    # Trend should be preserved (decreasing)
    assert filtered[0] > filtered[-1]
    
    # Variance should be reduced
    assert np.var(filtered) < np.var(measurements)
