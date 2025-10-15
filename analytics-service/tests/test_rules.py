"""
Tests for Alert Rules Module
"""

import pytest
from analytics.rules import (
    VoltageThresholdRule,
    CapacityDepletionRule,
    TemperatureExtremeRule,
    evaluate_all_rules
)


def test_voltage_threshold_rule():
    """Test voltage threshold rule"""
    rule = VoltageThresholdRule()
    
    # Normal voltage
    data = {"voltage": 3.7}
    result = rule.evaluate(data)
    assert result is None
    
    # Low voltage
    data = {"voltage": 3.0}
    result = rule.evaluate(data)
    assert result is not None
    assert result["severity"] == "critical"
    
    # High voltage
    data = {"voltage": 4.5}
    result = rule.evaluate(data)
    assert result is not None
    assert result["severity"] == "critical"


def test_capacity_depletion_rule():
    """Test capacity depletion rule"""
    rule = CapacityDepletionRule()
    
    # Normal capacity
    data = {"capacity": 80}
    result = rule.evaluate(data)
    assert result is None
    
    # Low capacity
    data = {"capacity": 30}
    result = rule.evaluate(data)
    assert result is not None
    assert result["severity"] == "high"
    
    # Critical capacity
    data = {"capacity": 15}
    result = rule.evaluate(data)
    assert result is not None
    assert result["severity"] == "critical"


def test_temperature_extreme_rule():
    """Test temperature extreme rule"""
    rule = TemperatureExtremeRule()
    
    # Normal temperature
    data = {"temperature": 25}
    result = rule.evaluate(data)
    assert result is None
    
    # High temperature
    data = {"temperature": 55}
    result = rule.evaluate(data)
    assert result is not None
    assert result["severity"] == "critical"
    
    # Low temperature
    data = {"temperature": -15}
    result = rule.evaluate(data)
    assert result is not None


def test_evaluate_all_rules():
    """Test evaluating all rules"""
    device_data = {
        "deviceId": "TEST-001",
        "voltage": 3.0,  # Critical low
        "capacity": 15,  # Critical low
        "temperature": 25
    }
    
    alerts = evaluate_all_rules(device_data)
    
    # Should trigger at least voltage and capacity alerts
    assert len(alerts) >= 2
    assert all("timestamp" in alert for alert in alerts)
    assert all("deviceId" in alert for alert in alerts)
