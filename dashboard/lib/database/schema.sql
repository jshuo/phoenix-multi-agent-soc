-- ============================================================================
-- Battery Telemetry Database Schema
-- Supply Chain IoT Battery Monitoring System
-- ============================================================================

-- Enable TimescaleDB extension (optional, for time-series optimization)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- Main battery telemetry table
-- ============================================================================

CREATE TABLE IF NOT EXISTS battery_telemetry (
    id BIGSERIAL PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Battery metrics
    voltage DECIMAL(5, 3) NOT NULL,              -- Voltage in volts (e.g., 3.725)
    capacity_percent DECIMAL(5, 2) NOT NULL,     -- Capacity percentage (0-100)
    temperature_celsius DECIMAL(5, 2) NOT NULL,  -- Temperature in Celsius
    charge_cycles INTEGER NOT NULL DEFAULT 0,     -- Number of charge cycles
    
    -- Location and metadata
    region VARCHAR(100),                          -- Geographic region
    supplier_id VARCHAR(100),                     -- Supplier identifier
    asset_type VARCHAR(50),                       -- Type of asset (GPS tracker, sensor, etc.)
    
    -- Additional metrics (optional)
    current_ma DECIMAL(8, 2),                     -- Current in milliamps
    power_mw DECIMAL(10, 2),                      -- Power in milliwatts
    resistance_mohm DECIMAL(8, 2),                -- Internal resistance in milliohms
    
    -- Data quality flags
    signal_quality INTEGER DEFAULT 100,           -- Signal quality 0-100
    data_source VARCHAR(50) DEFAULT 'sensor',     -- Data source identifier
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes for common queries
    CONSTRAINT voltage_range CHECK (voltage BETWEEN 0 AND 5),
    CONSTRAINT capacity_range CHECK (capacity_percent BETWEEN 0 AND 100),
    CONSTRAINT temperature_range CHECK (temperature_celsius BETWEEN -40 AND 85)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_battery_telemetry_device_timestamp 
    ON battery_telemetry(device_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_battery_telemetry_timestamp 
    ON battery_telemetry(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_battery_telemetry_region 
    ON battery_telemetry(region);

CREATE INDEX IF NOT EXISTS idx_battery_telemetry_device_id 
    ON battery_telemetry(device_id);

-- Convert to TimescaleDB hypertable for better time-series performance (optional)
-- SELECT create_hypertable('battery_telemetry', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- Battery alerts table
-- ============================================================================

CREATE TABLE IF NOT EXISTS battery_alerts (
    id BIGSERIAL PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,              -- low, medium, high, critical
    message TEXT NOT NULL,
    action VARCHAR(100),
    
    -- Metrics at time of alert
    voltage DECIMAL(5, 3),
    capacity_percent DECIMAL(5, 2),
    temperature_celsius DECIMAL(5, 2),
    z_score DECIMAL(8, 4),
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'active',         -- active, acknowledged, resolved
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT severity_values CHECK (severity IN ('low', 'medium', 'high', 'critical'))
);

CREATE INDEX IF NOT EXISTS idx_battery_alerts_device 
    ON battery_alerts(device_id);

CREATE INDEX IF NOT EXISTS idx_battery_alerts_severity 
    ON battery_alerts(severity);

CREATE INDEX IF NOT EXISTS idx_battery_alerts_status 
    ON battery_alerts(status);

CREATE INDEX IF NOT EXISTS idx_battery_alerts_created 
    ON battery_alerts(created_at DESC);

-- ============================================================================
-- Device metadata table
-- ============================================================================

CREATE TABLE IF NOT EXISTS battery_devices (
    device_id VARCHAR(100) PRIMARY KEY,
    device_type VARCHAR(50) NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    battery_type VARCHAR(50),
    nominal_capacity_mah INTEGER,
    nominal_voltage DECIMAL(4, 2),
    
    -- Installation info
    installed_date DATE,
    region VARCHAR(100),
    location_lat DECIMAL(10, 8),
    location_lon DECIMAL(11, 8),
    supplier_id VARCHAR(100),
    
    -- Current status
    status VARCHAR(50) DEFAULT 'active',         -- active, maintenance, retired
    last_maintenance_date DATE,
    next_maintenance_date DATE,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_battery_devices_region 
    ON battery_devices(region);

CREATE INDEX IF NOT EXISTS idx_battery_devices_status 
    ON battery_devices(status);

-- ============================================================================
-- Statistical baselines table (for Z-score analysis)
-- ============================================================================

CREATE TABLE IF NOT EXISTS battery_statistics (
    id BIGSERIAL PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,           -- voltage, capacity, temperature
    
    -- Statistical measures
    mean_value DECIMAL(10, 4),
    std_dev DECIMAL(10, 4),
    min_value DECIMAL(10, 4),
    max_value DECIMAL(10, 4),
    
    -- Time window
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    sample_count INTEGER NOT NULL,
    
    -- Timestamps
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_device_metric_window 
        UNIQUE(device_id, metric_name, window_start, window_end)
);

CREATE INDEX IF NOT EXISTS idx_battery_statistics_device_metric 
    ON battery_statistics(device_id, metric_name);

-- ============================================================================
-- Sample data insertion
-- ============================================================================

-- Insert sample devices
INSERT INTO battery_devices (device_id, device_type, manufacturer, model, battery_type, 
                             nominal_capacity_mah, nominal_voltage, region, status)
VALUES 
    ('GPS-TRACKER-B2', 'GPS Tracker', 'TechCorp', 'GT-2000', 'Li-ion', 2500, 3.7, 'Asia-Pacific', 'active'),
    ('TEMP-SENSOR-A1', 'Temperature Sensor', 'SensorPro', 'TS-100', 'Li-ion', 1500, 3.7, 'North America', 'active'),
    ('HUMIDITY-SENSOR-C3', 'Humidity Sensor', 'SensorPro', 'HS-200', 'Li-ion', 1800, 3.7, 'Asia-Pacific', 'active'),
    ('PRESSURE-MONITOR-D4', 'Pressure Monitor', 'IndustrialTech', 'PM-500', 'Li-ion', 3000, 3.7, 'Europe', 'active')
ON CONFLICT (device_id) DO NOTHING;

-- Insert sample telemetry data (last 30 days)
INSERT INTO battery_telemetry (device_id, timestamp, voltage, capacity_percent, 
                                temperature_celsius, charge_cycles, region)
SELECT 
    device_id,
    NOW() - (random() * INTERVAL '30 days'),
    3.7 + (random() * 0.5 - 0.25) + 
        CASE 
            WHEN device_id = 'GPS-TRACKER-B2' THEN -0.8
            WHEN device_id = 'PRESSURE-MONITOR-D4' THEN -1.0
            ELSE 0
        END,
    CASE 
        WHEN device_id = 'GPS-TRACKER-B2' THEN 62 + (random() * 10 - 5)
        WHEN device_id = 'PRESSURE-MONITOR-D4' THEN 45 + (random() * 10 - 5)
        WHEN device_id = 'TEMP-SENSOR-A1' THEN 85 + (random() * 10 - 5)
        ELSE 91 + (random() * 5 - 2.5)
    END,
    CASE 
        WHEN device_id = 'GPS-TRACKER-B2' THEN 31 + (random() * 8 - 4)
        WHEN device_id = 'PRESSURE-MONITOR-D4' THEN 38 + (random() * 10 - 5)
        ELSE 20 + (random() * 10 - 5)
    END,
    CASE 
        WHEN device_id = 'GPS-TRACKER-B2' THEN 2890 + floor(random() * 100)::int
        WHEN device_id = 'PRESSURE-MONITOR-D4' THEN 3200 + floor(random() * 100)::int
        WHEN device_id = 'TEMP-SENSOR-A1' THEN 1250 + floor(random() * 50)::int
        ELSE 850 + floor(random() * 50)::int
    END,
    CASE device_id
        WHEN 'GPS-TRACKER-B2' THEN 'Asia-Pacific'
        WHEN 'PRESSURE-MONITOR-D4' THEN 'Europe'
        WHEN 'TEMP-SENSOR-A1' THEN 'North America'
        ELSE 'Asia-Pacific'
    END
FROM 
    generate_series(1, 100) AS i,
    UNNEST(ARRAY['GPS-TRACKER-B2', 'TEMP-SENSOR-A1', 'HUMIDITY-SENSOR-C3', 'PRESSURE-MONITOR-D4']) AS device_id;

-- ============================================================================
-- Useful queries
-- ============================================================================

-- Get latest telemetry for all devices
-- SELECT DISTINCT ON (device_id)
--     device_id,
--     timestamp,
--     voltage,
--     capacity_percent,
--     temperature_celsius,
--     charge_cycles,
--     region
-- FROM battery_telemetry
-- ORDER BY device_id, timestamp DESC;

-- Calculate statistics for a device
-- SELECT 
--     device_id,
--     AVG(voltage) as avg_voltage,
--     STDDEV(voltage) as stddev_voltage,
--     AVG(capacity_percent) as avg_capacity,
--     STDDEV(capacity_percent) as stddev_capacity
-- FROM battery_telemetry
-- WHERE device_id = 'GPS-TRACKER-B2'
--   AND timestamp > NOW() - INTERVAL '30 days'
-- GROUP BY device_id;

-- Find devices with critical capacity
-- SELECT DISTINCT ON (device_id)
--     device_id,
--     capacity_percent,
--     timestamp
-- FROM battery_telemetry
-- WHERE capacity_percent < 50
-- ORDER BY device_id, timestamp DESC;
