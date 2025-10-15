"""
Z-Score Statistical Analysis
Anomaly detection using statistical deviation
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_zscore(
    value: float, 
    historical: List[float],
    threshold: float = 2.0
) -> dict:
    """
    Calculate Z-score for anomaly detection.
    
    Z-score measures how many standard deviations away a value is from the mean:
    z = (x - μ) / σ
    
    Typical thresholds:
    - |z| > 2.0: Warning (95% confidence interval)
    - |z| > 3.0: Critical (99.7% confidence interval)
    
    Args:
        value: Current measurement to analyze
        historical: Historical baseline measurements
        threshold: Z-score threshold for anomaly detection (default: 2.0)
    
    Returns:
        Dictionary with Z-score analysis results
    
    Example:
        >>> historical = [3.7, 3.71, 3.69, 3.72, 3.70]
        >>> result = calculate_zscore(4.2, historical)
        >>> print(result['isAnomaly'])  # True
    """
    if len(historical) < 2:
        logger.debug("Insufficient historical data for Z-score calculation")
        return {
            "zScore": 0.0,
            "mean": value,
            "stdDev": 0.0,
            "isAnomaly": False,
            "severity": "normal",
            "confidenceLevel": None
        }
    
    try:
        # Calculate statistical parameters
        mean = float(np.mean(historical))
        std = float(np.std(historical, ddof=1))  # Sample standard deviation
        
        # Handle zero standard deviation (all values identical)
        if std == 0:
            logger.debug("Zero standard deviation in historical data")
            return {
                "zScore": 0.0,
                "mean": mean,
                "stdDev": 0.0,
                "isAnomaly": abs(value - mean) > 0.01,  # Small tolerance
                "severity": "normal" if abs(value - mean) <= 0.01 else "warning",
                "confidenceLevel": None
            }
        
        # Calculate Z-score
        z_score = (value - mean) / std
        abs_z = abs(z_score)
        
        # Determine severity level
        if abs_z > 3.0:
            severity = "critical"
            confidence = 99.7
        elif abs_z > threshold:
            severity = "warning"
            confidence = 95.0
        else:
            severity = "normal"
            confidence = None
        
        result = {
            "zScore": float(z_score),
            "mean": mean,
            "stdDev": std,
            "isAnomaly": abs_z > threshold,
            "severity": severity,
            "confidenceLevel": confidence,
            "deviation": float(value - mean),
            "deviationPercent": float(((value - mean) / mean) * 100) if mean != 0 else 0.0
        }
        
        logger.debug(f"Z-score calculated: {z_score:.2f} (severity: {severity})")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating Z-score: {e}")
        return {
            "zScore": 0.0,
            "mean": value,
            "stdDev": 0.0,
            "isAnomaly": False,
            "severity": "error",
            "confidenceLevel": None
        }


def analyze_time_series_anomalies(
    timestamps: List[str],
    values: List[float],
    window_size: int = 10,
    threshold: float = 2.0
) -> List[dict]:
    """
    Perform rolling Z-score analysis on time series data.
    
    Args:
        timestamps: List of timestamp strings
        values: List of measurements
        window_size: Rolling window size for baseline calculation
        threshold: Z-score threshold for anomaly detection
    
    Returns:
        List of anomaly detection results for each timestamp
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    results = []
    
    for i in range(len(values)):
        # Use sliding window for historical baseline
        start_idx = max(0, i - window_size)
        historical = values[start_idx:i] if i > 0 else []
        
        if len(historical) >= 2:
            analysis = calculate_zscore(values[i], historical, threshold)
            analysis['timestamp'] = timestamps[i]
            analysis['value'] = values[i]
            results.append(analysis)
        else:
            # Not enough historical data yet
            results.append({
                'timestamp': timestamps[i],
                'value': values[i],
                'zScore': 0.0,
                'isAnomaly': False,
                'severity': 'normal'
            })
    
    anomaly_count = sum(1 for r in results if r['isAnomaly'])
    logger.info(f"Time series analysis: {anomaly_count}/{len(results)} anomalies detected")
    
    return results


def calculate_confidence_interval(
    historical: List[float],
    confidence: float = 0.95
) -> dict:
    """
    Calculate confidence interval for historical data.
    
    Args:
        historical: Historical measurements
        confidence: Confidence level (0.95 = 95%, 0.99 = 99%)
    
    Returns:
        Dictionary with confidence interval bounds
    """
    if len(historical) < 2:
        return {"lower": None, "upper": None, "mean": None}
    
    from scipy import stats
    
    mean = np.mean(historical)
    std_err = stats.sem(historical)
    
    # Calculate confidence interval
    interval = stats.t.interval(
        confidence, 
        len(historical) - 1,
        loc=mean, 
        scale=std_err
    )
    
    return {
        "mean": float(mean),
        "lower": float(interval[0]),
        "upper": float(interval[1]),
        "confidence": confidence * 100
    }


def detect_outliers_iqr(values: List[float]) -> dict:
    """
    Detect outliers using Interquartile Range (IQR) method.
    Alternative to Z-score for non-normal distributions.
    
    Args:
        values: List of measurements
    
    Returns:
        Dictionary with outlier detection results
    """
    if len(values) < 4:
        return {"outliers": [], "lowerBound": None, "upperBound": None}
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [v for v in values if v < lower_bound or v > upper_bound]
    
    return {
        "outliers": outliers,
        "outlierCount": len(outliers),
        "lowerBound": float(lower_bound),
        "upperBound": float(upper_bound),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr)
    }
