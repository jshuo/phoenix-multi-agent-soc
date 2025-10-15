"""
Tests for Z-Score Analysis Module
"""

import pytest
from analytics.zscore import calculate_zscore, detect_outliers_iqr


def test_zscore_calculation():
    """Test Z-score calculation"""
    historical = [3.7, 3.71, 3.69, 3.72, 3.70]
    
    # Normal value
    result = calculate_zscore(3.70, historical)
    assert result["isAnomaly"] == False
    assert result["severity"] == "normal"
    
    # Anomalous value
    result = calculate_zscore(4.2, historical)
    assert result["isAnomaly"] == True
    assert abs(result["zScore"]) > 2.0


def test_zscore_insufficient_data():
    """Test Z-score with insufficient historical data"""
    result = calculate_zscore(3.7, [])
    assert result["zScore"] == 0.0
    assert result["isAnomaly"] == False


def test_zscore_zero_variance():
    """Test Z-score with zero variance (all same values)"""
    historical = [3.7, 3.7, 3.7, 3.7]
    result = calculate_zscore(3.7, historical)
    assert result["zScore"] == 0.0


def test_outlier_detection_iqr():
    """Test IQR outlier detection"""
    values = [1, 2, 2, 3, 3, 3, 4, 4, 5, 100]  # 100 is an outlier
    
    result = detect_outliers_iqr(values)
    
    assert "outliers" in result
    assert 100 in result["outliers"]
    assert result["outlierCount"] > 0


def test_outlier_detection_insufficient_data():
    """Test IQR with insufficient data"""
    result = detect_outliers_iqr([1, 2])
    assert result["outliers"] == []
