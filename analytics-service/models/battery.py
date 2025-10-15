"""
Pydantic Models for Battery Analytics API
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class BatteryQuery(BaseModel):
    """Request model for battery performance analysis"""
    region: Optional[str] = Field(None, description="Filter by region (e.g., 'Asia-Pacific')")
    deviceId: Optional[str] = Field(None, description="Filter by specific device ID")
    health: Optional[str] = Field(None, description="Filter by health status")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of records to return")
    applyKalman: bool = Field(True, description="Apply Kalman filter for noise reduction")
    applyZScore: bool = Field(True, description="Apply Z-score anomaly detection")
    applyRules: bool = Field(True, description="Apply alert rules")
    includeWeather: bool = Field(False, description="Include weather data correlation")


class WeatherData(BaseModel):
    """Weather data model"""
    location: str
    timestamp: str
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    pressure: Optional[float] = Field(None, description="Atmospheric pressure in hPa")
    precipitation: Optional[float] = Field(None, description="Precipitation in mm")
    windSpeed: Optional[float] = Field(None, description="Wind speed in km/h")
    conditions: str = Field(..., description="Weather conditions description")


class BatteryAlert(BaseModel):
    """Alert model"""
    alertType: str
    severity: str
    message: str
    action: str
    timestamp: datetime
    voltage: Optional[float] = None
    capacity: Optional[float] = None
    temperature: Optional[float] = None
    zScore: Optional[float] = None


class DeviceAnalytics(BaseModel):
    """Device analytics result"""
    device: str
    voltage: float
    capacity: float
    temperature: float
    cycles: int
    region: str
    health: str
    predictedLife: str
    filteredVoltage: Optional[float] = None
    filteredCapacity: Optional[float] = None
    voltageZScore: Optional[float] = None
    capacityZScore: Optional[float] = None
    temperatureZScore: Optional[float] = None
    alerts: List[BatteryAlert] = []
    weather: Optional[WeatherData] = None


class AnalyticsSummary(BaseModel):
    """Summary statistics"""
    totalDevices: int
    healthyDevices: int
    warningDevices: int
    criticalDevices: int
    avgCapacity: float
    totalAlerts: int
    criticalAlerts: int


class AnalyticsMetadata(BaseModel):
    """Analytics processing metadata"""
    kalmanFilterApplied: bool
    zScoreAnalysisApplied: bool
    rulesEvaluated: int
    anomaliesDetected: int
    weatherEnriched: bool


class BatteryPerformanceResponse(BaseModel):
    """Complete battery performance response"""
    devices: List[DeviceAnalytics]
    summary: AnalyticsSummary
    analytics: AnalyticsMetadata
    regionalWeather: Optional[dict] = None
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: str
