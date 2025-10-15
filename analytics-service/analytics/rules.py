"""
Alert Rules Engine
Battery performance and weather-aware alert rules
"""

from typing import List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertRule:
    """Base class for alert rules"""
    
    def __init__(self, rule_id: str, name: str, severity: str, action: str):
        self.rule_id = rule_id
        self.name = name
        self.severity = severity
        self.action = action
    
    def evaluate(self, data: dict, weather: Optional[dict] = None) -> Optional[dict]:
        """
        Evaluate rule and return alert if triggered.
        Override this method in subclasses.
        """
        raise NotImplementedError


class VoltageThresholdRule(AlertRule):
    """Alert when voltage is outside safe range"""
    
    def __init__(self):
        super().__init__(
            rule_id="VOLTAGE_THRESHOLD",
            name="Voltage Out of Range",
            severity="critical",
            action="CHECK_DEVICE"
        )
    
    def evaluate(self, data: dict, weather: Optional[dict] = None) -> Optional[dict]:
        voltage = data.get("voltage", 0)
        
        if voltage < 3.2:
            return {
                "alertType": self.rule_id,
                "severity": "critical",
                "message": f"Critical low voltage: {voltage}V (minimum: 3.2V)",
                "action": "IMMEDIATE_ATTENTION",
                "voltage": voltage
            }
        elif voltage > 4.2:
            return {
                "alertType": self.rule_id,
                "severity": "critical",
                "message": f"Critical high voltage: {voltage}V (maximum: 4.2V)",
                "action": "SHUTDOWN_DEVICE",
                "voltage": voltage
            }
        
        return None


class CapacityDepletionRule(AlertRule):
    """Alert when battery capacity is low"""
    
    def __init__(self):
        super().__init__(
            rule_id="CAPACITY_DEPLETION",
            name="Low Battery Capacity",
            severity="high",
            action="RECHARGE_REPLACE"
        )
    
    def evaluate(self, data: dict, weather: Optional[dict] = None) -> Optional[dict]:
        capacity = data.get("capacity", 100)
        
        if capacity < 20:
            return {
                "alertType": self.rule_id,
                "severity": "critical",
                "message": f"Critical battery depletion: {capacity}%",
                "action": "IMMEDIATE_REPLACEMENT",
                "capacity": capacity
            }
        elif capacity < 40:
            return {
                "alertType": self.rule_id,
                "severity": "high",
                "message": f"Low battery capacity: {capacity}%",
                "action": "SCHEDULE_REPLACEMENT",
                "capacity": capacity
            }
        
        return None


class TemperatureExtremeRule(AlertRule):
    """Alert on extreme temperatures"""
    
    def __init__(self):
        super().__init__(
            rule_id="TEMPERATURE_EXTREME",
            name="Extreme Temperature",
            severity="high",
            action="CHECK_COOLING"
        )
    
    def evaluate(self, data: dict, weather: Optional[dict] = None) -> Optional[dict]:
        temp = data.get("temperature", 25)
        
        if temp > 50:
            return {
                "alertType": self.rule_id,
                "severity": "critical",
                "message": f"Critical overheating: {temp}°C (thermal shutdown risk)",
                "action": "IMMEDIATE_COOLING",
                "temperature": temp
            }
        elif temp < -10:
            return {
                "alertType": self.rule_id,
                "severity": "high",
                "message": f"Extreme cold: {temp}°C (performance degradation)",
                "action": "MONITOR_PERFORMANCE",
                "temperature": temp
            }
        
        return None


class ZScoreAnomalyRule(AlertRule):
    """Alert on statistical anomalies"""
    
    def __init__(self):
        super().__init__(
            rule_id="ZSCORE_ANOMALY",
            name="Statistical Anomaly Detected",
            severity="medium",
            action="INVESTIGATE"
        )
    
    def evaluate(self, data: dict, weather: Optional[dict] = None) -> Optional[dict]:
        voltage_z = data.get("voltageZScore", 0)
        capacity_z = data.get("capacityZScore", 0)
        
        alerts = []
        
        if abs(voltage_z) > 3.0:
            alerts.append({
                "alertType": f"{self.rule_id}_VOLTAGE",
                "severity": "critical",
                "message": f"Critical voltage anomaly: Z-score = {voltage_z:.2f}",
                "action": "IMMEDIATE_INVESTIGATION",
                "zScore": voltage_z
            })
        elif abs(voltage_z) > 2.0:
            alerts.append({
                "alertType": f"{self.rule_id}_VOLTAGE",
                "severity": "medium",
                "message": f"Voltage anomaly detected: Z-score = {voltage_z:.2f}",
                "action": "INVESTIGATE",
                "zScore": voltage_z
            })
        
        if abs(capacity_z) > 3.0:
            alerts.append({
                "alertType": f"{self.rule_id}_CAPACITY",
                "severity": "critical",
                "message": f"Critical capacity anomaly: Z-score = {capacity_z:.2f}",
                "action": "IMMEDIATE_INVESTIGATION",
                "zScore": capacity_z
            })
        
        return alerts if alerts else None


class WeatherCorrelationRule(AlertRule):
    """Weather-aware alert rules"""
    
    def __init__(self):
        super().__init__(
            rule_id="WEATHER_CORRELATION",
            name="Weather Impact Alert",
            severity="medium",
            action="MONITOR_ENVIRONMENT"
        )
    
    def evaluate(self, data: dict, weather: Optional[dict] = None) -> Optional[dict]:
        if not weather:
            return None
        
        alerts = []
        device_temp = data.get("temperature", 25)
        ambient_temp = weather.get("temperature", 25)
        capacity = data.get("capacity", 100)
        
        # High ambient temperature + low capacity
        if ambient_temp > 35 and capacity < 50:
            alerts.append({
                "alertType": "WEATHER_HEAT_STRESS",
                "severity": "high",
                "message": f"Heat stress: {ambient_temp}°C ambient, {capacity}% capacity",
                "action": "MOVE_TO_COOLER_LOCATION",
                "weather": {"temperature": ambient_temp}
            })
        
        # Device much hotter than ambient
        temp_delta = device_temp - ambient_temp
        if temp_delta > 15:
            alerts.append({
                "alertType": "WEATHER_INTERNAL_HEAT",
                "severity": "high",
                "message": f"Abnormal heat: device {device_temp}°C, ambient {ambient_temp}°C (Δ{temp_delta:.1f}°C)",
                "action": "CHECK_THERMAL_MANAGEMENT",
                "weather": {"temperature": ambient_temp}
            })
        
        # High humidity + voltage issues
        humidity = weather.get("humidity")
        voltage = data.get("voltage", 3.7)
        if humidity and humidity > 80 and voltage < 3.5:
            alerts.append({
                "alertType": "WEATHER_HUMIDITY_CORROSION",
                "severity": "medium",
                "message": f"High humidity ({humidity}%) + low voltage ({voltage}V): corrosion risk",
                "action": "INSPECT_DEVICE",
                "weather": {"humidity": humidity}
            })
        
        return alerts if alerts else None


# Master list of all rules
BATTERY_ALERT_RULES: List[AlertRule] = [
    VoltageThresholdRule(),
    CapacityDepletionRule(),
    TemperatureExtremeRule(),
    ZScoreAnomalyRule(),
    WeatherCorrelationRule()
]


def evaluate_all_rules(device_data: dict, weather_data: Optional[dict] = None) -> List[dict]:
    """
    Evaluate all rules against device data.
    
    Args:
        device_data: Dictionary with device telemetry
        weather_data: Optional weather data for correlation
    
    Returns:
        List of triggered alerts
    """
    all_alerts = []
    
    for rule in BATTERY_ALERT_RULES:
        try:
            result = rule.evaluate(device_data, weather_data)
            
            if result:
                # Handle both single alert and list of alerts
                if isinstance(result, list):
                    all_alerts.extend(result)
                else:
                    all_alerts.append(result)
                    
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    # Add timestamp to all alerts
    for alert in all_alerts:
        alert['timestamp'] = datetime.now().isoformat()
        alert['deviceId'] = device_data.get('deviceId', 'unknown')
    
    logger.info(f"Evaluated {len(BATTERY_ALERT_RULES)} rules, triggered {len(all_alerts)} alerts")
    return all_alerts
