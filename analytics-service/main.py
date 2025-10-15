"""
Battery Analytics Microservice
FastAPI service for compute-intensive IoT analytics operations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
from typing import Optional

# Import local modules
from models import (
    BatteryQuery,
    BatteryPerformanceResponse,
    DeviceAnalytics,
    AnalyticsSummary,
    AnalyticsMetadata,
    HealthCheckResponse
)
from analytics import (
    apply_kalman_filter,
    calculate_zscore,
    evaluate_all_rules,
    get_regional_weather,
    BATTERY_ALERT_RULES
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection pool
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    # Startup
    logger.info("Starting Battery Analytics Service...")
    global db_pool
    
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # min and max connections
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "supply_chain_iot"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Service will use mock data.")
        db_pool = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Battery Analytics Service...")
    if db_pool:
        db_pool.closeall()


# Create FastAPI app
app = FastAPI(
    title="Battery Analytics Service",
    description="High-performance analytics for IoT battery monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_battery_telemetry(query: BatteryQuery) -> list:
    """
    Fetch battery telemetry from PostgreSQL database
    """
    if not db_pool:
        logger.warning("Database not available, using mock data")
        return get_mock_telemetry()
    
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build SQL query
        sql = """
            SELECT 
                device_id,
                timestamp,
                voltage,
                capacity_percent as capacity,
                temperature_celsius as temperature,
                charge_cycles as cycles,
                region
            FROM battery_telemetry
            WHERE timestamp > NOW() - INTERVAL '7 days'
        """
        
        conditions = []
        params = []
        
        if query.region:
            conditions.append("region = %s")
            params.append(query.region)
        
        if query.deviceId:
            conditions.append("device_id = %s")
            params.append(query.deviceId)
        
        if conditions:
            sql += " AND " + " AND ".join(conditions)
        
        sql += " ORDER BY device_id, timestamp DESC LIMIT %s"
        params.append(query.limit)
        
        # Execute query
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        telemetry = [dict(row) for row in rows]
        
        cursor.close()
        db_pool.putconn(conn)
        
        logger.info(f"Fetched {len(telemetry)} telemetry records from database")
        return telemetry
        
    except Exception as e:
        logger.error(f"Database query error: {e}")
        if conn:
            db_pool.putconn(conn)
        return get_mock_telemetry()


def get_mock_telemetry() -> list:
    """Generate mock telemetry data for testing"""
    import random
    from datetime import timedelta
    
    devices = ["GPS-TRACKER-A1", "GPS-TRACKER-B2", "SENSOR-NODE-C3"]
    regions = ["Asia-Pacific", "Europe", "North America"]
    
    telemetry = []
    now = datetime.now()
    
    for device_id in devices:
        region = random.choice(regions)
        
        # Generate 20 data points over last 7 days
        for i in range(20):
            timestamp = now - timedelta(days=7 * i / 20)
            
            telemetry.append({
                "deviceId": device_id,
                "timestamp": timestamp,
                "voltage": round(3.7 + random.uniform(-0.2, 0.2), 2),
                "capacity": round(max(20, 100 - i * 3 + random.uniform(-5, 5)), 1),
                "temperature": round(25 + random.uniform(-5, 10), 1),
                "cycles": 100 + i * 2,
                "region": region
            })
    
    logger.info(f"Generated {len(telemetry)} mock telemetry records")
    return telemetry


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Battery Analytics Service",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "analytics": "/api/analytics/battery"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        service="battery-analytics",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/analytics/battery", response_model=BatteryPerformanceResponse)
async def analyze_battery_performance(query: BatteryQuery):
    """
    Main battery analytics endpoint.
    
    Performs:
    - Kalman filtering for noise reduction
    - Z-score statistical analysis
    - Alert rule evaluation
    - Weather correlation (optional)
    
    Example request:
    ```json
    {
        "region": "Asia-Pacific",
        "limit": 100,
        "applyKalman": true,
        "applyZScore": true,
        "includeWeather": true
    }
    ```
    """
    try:
        logger.info(f"Processing battery analytics query: {query}")
        
        # Fetch telemetry data
        telemetry = get_battery_telemetry(query)
        
        if not telemetry:
            return BatteryPerformanceResponse(
                devices=[],
                summary=AnalyticsSummary(
                    totalDevices=0,
                    healthyDevices=0,
                    warningDevices=0,
                    criticalDevices=0,
                    avgCapacity=0.0,
                    totalAlerts=0,
                    criticalAlerts=0
                ),
                analytics=AnalyticsMetadata(
                    kalmanFilterApplied=False,
                    zScoreAnalysisApplied=False,
                    rulesEvaluated=0,
                    anomaliesDetected=0,
                    weatherEnriched=False
                ),
                timestamp=datetime.now().isoformat()
            )
        
        # Group telemetry by device
        device_map = {}
        for item in telemetry:
            device_id = item["deviceId"]
            if device_id not in device_map:
                device_map[device_id] = []
            device_map[device_id].append(item)
        
        # Fetch weather data if requested
        regional_weather = None
        if query.includeWeather:
            regions = list(set(item["region"] for item in telemetry))
            regional_weather = await get_regional_weather(regions)
        
        # Process each device
        processed_devices = []
        total_anomalies = 0
        
        for device_id, data in device_map.items():
            # Sort by timestamp (oldest first for time-series processing)
            data.sort(key=lambda x: x["timestamp"])
            
            latest = data[-1]
            region = latest["region"]
            
            # Apply Kalman filter if requested
            filtered_voltage = latest["voltage"]
            filtered_capacity = latest["capacity"]
            
            if query.applyKalman and len(data) > 1:
                voltages = [d["voltage"] for d in data]
                capacities = [d["capacity"] for d in data]
                
                filtered_voltages = apply_kalman_filter(voltages, 0.01, 0.1)
                filtered_capacities = apply_kalman_filter(capacities, 0.005, 0.05)
                
                filtered_voltage = filtered_voltages[-1]
                filtered_capacity = filtered_capacities[-1]
            
            # Z-score analysis if requested
            voltage_zscore = None
            capacity_zscore = None
            
            if query.applyZScore and len(data) > 5:
                historical_voltages = [d["voltage"] for d in data[:-1]]
                historical_capacities = [d["capacity"] for d in data[:-1]]
                
                voltage_analysis = calculate_zscore(latest["voltage"], historical_voltages)
                capacity_analysis = calculate_zscore(latest["capacity"], historical_capacities)
                
                voltage_zscore = voltage_analysis["zScore"]
                capacity_zscore = capacity_analysis["zScore"]
                
                if voltage_analysis["isAnomaly"] or capacity_analysis["isAnomaly"]:
                    total_anomalies += 1
            
            # Determine health status
            health = "Good"
            if filtered_capacity < 20 or (voltage_zscore and abs(voltage_zscore) > 3):
                health = "Critical"
            elif filtered_capacity < 40 or (voltage_zscore and abs(voltage_zscore) > 2):
                health = "Warning"
            elif filtered_capacity > 80:
                health = "Excellent"
            
            # Predict remaining life (simplified)
            cycles = latest["cycles"]
            predicted_life = "Unknown"
            if cycles > 0:
                remaining_cycles = max(0, 500 - cycles)
                predicted_life = f"{remaining_cycles} cycles (~{remaining_cycles // 30} months)"
            
            # Build device data for rule evaluation
            device_data = {
                "deviceId": device_id,
                "voltage": latest["voltage"],
                "capacity": latest["capacity"],
                "temperature": latest["temperature"],
                "filteredVoltage": filtered_voltage,
                "filteredCapacity": filtered_capacity,
                "voltageZScore": voltage_zscore,
                "capacityZScore": capacity_zscore
            }
            
            # Evaluate alert rules
            alerts = []
            if query.applyRules:
                weather_data = regional_weather.get(region) if regional_weather else None
                alerts = evaluate_all_rules(device_data, weather_data)
            
            # Build device result
            device_result = DeviceAnalytics(
                device=device_id,
                voltage=latest["voltage"],
                capacity=latest["capacity"],
                temperature=latest["temperature"],
                cycles=cycles,
                region=region,
                health=health,
                predictedLife=predicted_life,
                filteredVoltage=filtered_voltage,
                filteredCapacity=filtered_capacity,
                voltageZScore=voltage_zscore,
                capacityZScore=capacity_zscore,
                alerts=alerts,
                weather=regional_weather.get(region) if regional_weather else None
            )
            
            processed_devices.append(device_result)
        
        # Calculate summary statistics
        total_devices = len(processed_devices)
        healthy_count = sum(1 for d in processed_devices if d.health in ["Good", "Excellent"])
        warning_count = sum(1 for d in processed_devices if d.health == "Warning")
        critical_count = sum(1 for d in processed_devices if d.health == "Critical")
        avg_capacity = sum(d.capacity for d in processed_devices) / total_devices if total_devices > 0 else 0
        total_alerts = sum(len(d.alerts) for d in processed_devices)
        critical_alerts = sum(1 for d in processed_devices for a in d.alerts if a.get("severity") == "critical")
        
        # Build response
        response = BatteryPerformanceResponse(
            devices=processed_devices,
            summary=AnalyticsSummary(
                totalDevices=total_devices,
                healthyDevices=healthy_count,
                warningDevices=warning_count,
                criticalDevices=critical_count,
                avgCapacity=round(avg_capacity, 1),
                totalAlerts=total_alerts,
                criticalAlerts=critical_alerts
            ),
            analytics=AnalyticsMetadata(
                kalmanFilterApplied=query.applyKalman,
                zScoreAnalysisApplied=query.applyZScore,
                rulesEvaluated=len(BATTERY_ALERT_RULES),
                anomaliesDetected=total_anomalies,
                weatherEnriched=query.includeWeather
            ),
            regionalWeather=regional_weather,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Analytics complete: {total_devices} devices, {total_alerts} alerts")
        return response
        
    except Exception as e:
        logger.error(f"Error processing analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/rules")
async def list_alert_rules():
    """List all available alert rules"""
    return {
        "rules": [
            {
                "id": rule.rule_id,
                "name": rule.name,
                "severity": rule.severity,
                "action": rule.action
            }
            for rule in BATTERY_ALERT_RULES
        ],
        "count": len(BATTERY_ALERT_RULES)
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("SERVICE_PORT", "8000"))
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
