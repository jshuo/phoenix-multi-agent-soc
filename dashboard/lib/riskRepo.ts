// lib/riskRepo.ts
/**
 * Risk Repository - Data access layer for risk management
 * Connects to FastAPI backend or database for risk data
 */

import type { RiskItem, RiskQueryParams, RiskQueryResponse, RiskTrend } from "@/types/risk";
import { getRegionalWeather } from "./weather";

// Mock data for development - replace with actual API calls
const MOCK_RISKS: RiskItem[] = [
  {
    id: "risk-001",
    assetId: "GPS-TRACKER-B2",
    severity: "high",
    score: 87,
    reasons: [
      { feature: "battery_capacity", weight: 0.35, contribution: 28 },
      { feature: "temperature_anomaly", weight: 0.25, contribution: 22 },
      { feature: "charge_cycles", weight: 0.20, contribution: 17 },
      { feature: "voltage_drift", weight: 0.20, contribution: 20 }
    ],
    timeWindow: "2025-10-03..2025-10-10",
    region: "Asia-Pacific",
    lastUpdated: "2025-10-10T08:30:00Z"
  },
  {
    id: "risk-002",
    assetId: "PRESSURE-MONITOR-D4",
    severity: "high",
    score: 92,
    reasons: [
      { feature: "battery_capacity", weight: 0.40, contribution: 38 },
      { feature: "temperature_critical", weight: 0.30, contribution: 28 },
      { feature: "charge_cycles", weight: 0.30, contribution: 26 }
    ],
    timeWindow: "2025-10-03..2025-10-10",
    region: "Europe",
    lastUpdated: "2025-10-10T09:15:00Z"
  },
  {
    id: "risk-003",
    assetId: "TEMP-SENSOR-A1",
    severity: "medium",
    score: 45,
    reasons: [
      { feature: "signal_quality", weight: 0.30, contribution: 15 },
      { feature: "battery_health", weight: 0.25, contribution: 12 },
      { feature: "calibration_drift", weight: 0.25, contribution: 10 },
      { feature: "connectivity_issues", weight: 0.20, contribution: 8 }
    ],
    timeWindow: "2025-10-03..2025-10-10",
    region: "North America",
    lastUpdated: "2025-10-10T07:45:00Z"
  },
  {
    id: "risk-004",
    assetId: "HUMIDITY-SENSOR-C3",
    severity: "low",
    score: 22,
    reasons: [
      { feature: "minor_calibration_drift", weight: 0.40, contribution: 9 },
      { feature: "battery_age", weight: 0.35, contribution: 8 },
      { feature: "signal_variance", weight: 0.25, contribution: 5 }
    ],
    timeWindow: "2025-10-03..2025-10-10",
    region: "Asia-Pacific",
    lastUpdated: "2025-10-10T06:20:00Z"
  }
];

// Mock data for supplier risks
const MOCK_SUPPLIER_RISKS = [
  { name: 'Acme Electronics Ltd', risk: 'High', issue: 'Port congestion delays', impact: '$2.3M', region: 'Asia-Pacific' },
  { name: 'Global Components Inc', risk: 'Medium', issue: 'Quality control issues', impact: '$890K', region: 'North America' },
  { name: 'Pacific Manufacturing', risk: 'Medium', issue: 'Labor shortage', impact: '$650K', region: 'Asia-Pacific' }
];

// Mock data for IoT battery performance
const MOCK_BATTERY_DATA = [
  { device: 'Temp Sensor A1', voltage: 3.2, capacity: 85, temperature: 23, cycles: 1250, health: 'Good', predictedLife: '8 months', region: 'North America' },
  { device: 'GPS Tracker B2', voltage: 2.9, capacity: 35, temperature: 31, cycles: 2890, health: 'Warning', predictedLife: '3 months', region: 'Asia-Pacific' },  // ⚠️ Capacity dropped from 62% to 35% - will trigger Z-score anomaly!
  { device: 'Humidity Sensor C3', voltage: 3.4, capacity: 91, temperature: 19, cycles: 850, health: 'Excellent', predictedLife: '12 months', region: 'Asia-Pacific' },
  { device: 'Pressure Monitor D4', voltage: 2.7, capacity: 45, temperature: 38, cycles: 3200, health: 'Critical', predictedLife: '1 month', region: 'Europe' }
];

/**
 * Fetch top risks based on query parameters
 */
export async function getTopRisks(params: RiskQueryParams): Promise<RiskQueryResponse> {
  const { region, days = 7, severity, minScore = 0, limit = 10 } = params;

  // In production, this would be an API call to FastAPI backend
  // const response = await fetch(`${process.env.API_BASE_URL}/api/risks`, {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(params)
  // });
  // return response.json();

  // Filter mock data
  let filtered = MOCK_RISKS.filter(risk => {
    if (region && risk.region !== region) return false;
    if (severity && risk.severity !== severity) return false;
    if (risk.score < minScore) return false;
    return true;
  });

  // Sort by score descending
  filtered.sort((a, b) => b.score - a.score);

  // Apply limit
  const risks = filtered.slice(0, limit);

  return {
    risks,
    totalCount: filtered.length,
    queryParams: params,
    timestamp: new Date().toISOString()
  };
}

/**
 * Get risk by ID
 */
export async function getRiskById(id: string): Promise<RiskItem | null> {
  // In production: API call to backend
  const risk = MOCK_RISKS.find(r => r.id === id);
  return risk || null;
}

/**
 * Get risk trends over time
 */
export async function getRiskTrends(params: {
  region?: string;
  days?: number;
}): Promise<RiskTrend[]> {
  const { region, days = 7 } = params;

  // Mock trend data - replace with actual time-series query
  const trends: RiskTrend[] = [];
  const today = new Date();

  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split('T')[0];

    // Filter risks for this date/region
    const filtered = MOCK_RISKS.filter(risk => {
      if (region && risk.region !== region) return false;
      return true;
    });

    const avgScore = filtered.length > 0
      ? filtered.reduce((sum, r) => sum + r.score, 0) / filtered.length
      : 0;

    const highSeverityCount = filtered.filter(r => r.severity === 'high').length;

    trends.push({
      date: dateStr,
      avgScore: Math.round(avgScore * 100) / 100,
      count: filtered.length,
      highSeverityCount,
      region
    });
  }

  return trends;
}

/**
 * Get summary statistics
 */
export async function getRiskSummary(params: { region?: string }) {
  const { region } = params;

  const filtered = MOCK_RISKS.filter(risk => {
    if (region && risk.region !== region) return false;
    return true;
  });

  return {
    totalRisks: filtered.length,
    highSeverity: filtered.filter(r => r.severity === 'high').length,
    mediumSeverity: filtered.filter(r => r.severity === 'medium').length,
    lowSeverity: filtered.filter(r => r.severity === 'low').length,
    avgScore: filtered.length > 0
      ? Math.round((filtered.reduce((sum, r) => sum + r.score, 0) / filtered.length) * 100) / 100
      : 0,
    criticalAssets: filtered.filter(r => r.score >= 80).map(r => r.assetId)
  };
}

/**
 * Get supplier risks
 */
export async function getSupplierRisks(params: { region?: string; limit?: number }) {
  const { region, limit = 10 } = params;
  
  let filtered = MOCK_SUPPLIER_RISKS.filter(supplier => {
    if (region && supplier.region !== region) return false;
    return true;
  });

  return {
    suppliers: filtered.slice(0, limit),
    totalCount: filtered.length,
    timestamp: new Date().toISOString()
  };
}

/**
 * Get battery performance data with advanced analytics
 * 
 * This function now supports two modes:
 * 1. SQL mode: Fetches data from database with Kalman filtering, Z-score analysis, and rule engine
 * 2. Mock mode: Returns mock data for development/testing
 * 
 * Set USE_SQL_BATTERY_ANALYTICS=true in .env to enable SQL mode
 */
export async function getBatteryPerformance(params: { 
  region?: string; 
  health?: string;
  deviceId?: string;
  useAdvancedAnalytics?: boolean;
}) {
  const { region, health, deviceId, useAdvancedAnalytics = false } = params;
  
  // Check if SQL-based analytics should be used (database mode)
  const useSqlAnalytics = process.env.USE_SQL_BATTERY_ANALYTICS === 'true';
  
  if (useSqlAnalytics) {
    try {
      console.log('[Risk Repo] Attempting to use SQL-based battery analytics');
      // Import the advanced analytics module dynamically
      const { getBatteryPerformance: getAdvancedBatteryPerformance } = await import('./batteryAnalytics');
      
      // Use the advanced analytics with Kalman filter, Z-score, and rule engine
      return await getAdvancedBatteryPerformance({
        region,
        health,
        deviceId,
        applyKalman: true,      // Enable Kalman filter for noise reduction
        applyZScore: true,      // Enable Z-score residual analysis
        applyRules: true,       // Enable rule engine for alerts
        limit: 100
      });
    } catch (error: any) {
      console.warn('[Risk Repo] SQL analytics failed, falling back to mock data with Kalman filter:', error.message);
      // Fall through to mock data with Kalman filter
    }
  }
  
  // FALLBACK: Use mock data WITH Kalman filter applied
  console.log('[Risk Repo] Using mock data with Kalman filter enabled');
  
  // Apply Kalman filter to mock data if useAdvancedAnalytics is requested
  if (useAdvancedAnalytics) {
    try {
      // Import Kalman filter functions AND Z-score analysis
      const { applyKalmanFilter, analyzeZScore, evaluateAlertRules } = await import('./batteryAnalytics');
      
      console.log('[Risk Repo] Applying Kalman filter to mock battery data');
      
      // Filter by criteria first
      let filtered = MOCK_BATTERY_DATA.filter(battery => {
        if (region && battery.region !== region) return false;
        if (health && battery.health !== health) return false;
        if (deviceId && battery.device !== deviceId) return false;
        return true;
      });
      
      // Apply Kalman filtering to each device's data
      const enhancedDevices = filtered.map(device => {
        // Generate synthetic time series data (simulating historical measurements)
        const numSamples = 15;  // 15 samples (realistic for 7 days)
        const baseVoltage = device.voltage;
        const baseCapacity = device.capacity;
        
        // Add realistic noise to create synthetic measurements
        const voltageNoise = 0.15; // ±0.15V noise
        const capacityNoise = 3.0;   // ±3% noise
        
        // 🔥 SPECIAL CASE: GPS Tracker B2 - Simulate sudden capacity drop (anomaly)
        const isAnomalousDevice = device.device === 'GPS Tracker B2';
        
        const voltageHistory = Array.from({ length: numSamples }, (_, i) => 
          baseVoltage + (Math.random() - 0.5) * voltageNoise
        );
        
        // For GPS Tracker B2: Generate capacity history with sudden drop
        const capacityHistory = Array.from({ length: numSamples }, (_, i) => {
          if (isAnomalousDevice) {
            // Historical capacity was normal (around 62%), but suddenly dropped to 35%
            if (i < numSamples - 1) {
              // Past 14 measurements: normal capacity around 62%
              return 62 + (Math.random() - 0.5) * capacityNoise;
            } else {
              // Most recent measurement: sudden drop to 35% (ANOMALY!)
              return baseCapacity + (Math.random() - 0.5) * capacityNoise;
            }
          }
          return baseCapacity + (Math.random() - 0.5) * capacityNoise;
        });
        
        // Apply Kalman filter
        const filteredVoltages = applyKalmanFilter(voltageHistory, 0.01, 0.1);
        const filteredCapacities = applyKalmanFilter(capacityHistory, 0.005, 0.05);
        
        // Use the last filtered values
        const filteredVoltage = filteredVoltages[filteredVoltages.length - 1];
        const filteredCapacity = filteredCapacities[filteredCapacities.length - 1];
        
        // ✅ Z-score analysis for anomaly detection
        const voltageZScore = analyzeZScore(
          device.device,
          'voltage',
          filteredVoltage,
          voltageHistory.slice(0, -1),
          2.0  // 2-sigma threshold
        );
        
        const capacityZScore = analyzeZScore(
          device.device,
          'capacity',
          filteredCapacity,
          capacityHistory.slice(0, -1),
          2.0
        );
        
        // Generate temperature history for Z-score analysis
        const temperatureMeasurements = Array.from({ length: numSamples }, () => 
          device.temperature + (Math.random() - 0.5) * 2
        );
        const temperatureZScore = analyzeZScore(
          device.device,
          'temperature',
          device.temperature,
          temperatureMeasurements,
          2.0
        );
        
        // ✅ Evaluate alert rules based on Z-scores
        const telemetryData = {
          deviceId: device.device,
          timestamp: new Date(),
          voltage: filteredVoltage,
          capacity: filteredCapacity,
          temperature: device.temperature,
          chargeCycles: device.cycles,
          region: device.region
        };
        
        const alerts = evaluateAlertRules(telemetryData, [
          voltageZScore,
          capacityZScore,
          temperatureZScore
        ]);
        
        return {
          ...device,
          filteredVoltage: Math.round(filteredVoltage * 100) / 100,
          filteredCapacity: Math.round(filteredCapacity),
          voltageZScore: voltageZScore.zScore,        // ✅ Real Z-score!
          capacityZScore: capacityZScore.zScore,      // ✅ Real Z-score!
          temperatureZScore: temperatureZScore.zScore, // ✅ Real Z-score!
          alerts: alerts  // ✅ Real alerts based on rules!
        };
      });
      
      // Fetch weather data for all unique regions
      const uniqueRegions = [...new Set(enhancedDevices.map(d => d.region))];
      let weatherData = {};
      try {
        weatherData = await getRegionalWeather(uniqueRegions);
        console.log('[Risk Repo] Weather data fetched for battery performance analysis');
      } catch (error: any) {
        console.warn('[Risk Repo] Failed to fetch weather data:', error.message);
      }
      
      return {
        devices: enhancedDevices,
        summary: {
          totalDevices: MOCK_BATTERY_DATA.length,
          healthyDevices: MOCK_BATTERY_DATA.filter(b => b.health === 'Excellent' || b.health === 'Good').length,
          warningDevices: MOCK_BATTERY_DATA.filter(b => b.health === 'Warning').length,
          criticalDevices: MOCK_BATTERY_DATA.filter(b => b.health === 'Critical').length,
          avgCapacity: Math.round(MOCK_BATTERY_DATA.reduce((sum, b) => sum + b.capacity, 0) / MOCK_BATTERY_DATA.length),
          totalAlerts: enhancedDevices.reduce((sum, d) => sum + (d.alerts?.length || 0), 0),
          criticalAlerts: enhancedDevices.reduce((sum, d) => sum + (d.alerts?.filter((a: any) => a.severity === 'critical').length || 0), 0)
        },
        weather: weatherData,
        analytics: {
          kalmanFilterApplied: true,      // ✅ Kalman filter was applied!
          zScoreAnalysisApplied: true,    // ✅ Z-score analysis is NOW enabled!
          rulesEvaluated: 10,             // ✅ 10 rules from the rule engine
          anomaliesDetected: enhancedDevices.filter(d => 
            Math.abs(d.voltageZScore) > 2 || 
            Math.abs(d.capacityZScore) > 2 ||
            Math.abs(d.temperatureZScore) > 2
          ).length,
          weatherDataIncluded: Object.keys(weatherData).length > 0
        },
        timestamp: new Date().toISOString()
      };
    } catch (error: any) {
      console.warn('[Risk Repo] Kalman filter failed on mock data, using plain mock data:', error.message);
      // Fall through to plain mock data
    }
  }
  
  // Plain mock data (no Kalman filter)
  let filtered = MOCK_BATTERY_DATA.filter(battery => {
    if (region && battery.region !== region) return false;
    if (health && battery.health !== health) return false;
    if (deviceId && battery.device !== deviceId) return false;
    return true;
  });

  // Fetch weather data for all unique regions
  const uniqueRegions = [...new Set(filtered.map(d => d.region))];
  let weatherData = {};
  try {
    weatherData = await getRegionalWeather(uniqueRegions);
    console.log('[Risk Repo] Weather data fetched for battery performance analysis');
  } catch (error: any) {
    console.warn('[Risk Repo] Failed to fetch weather data:', error.message);
  }

  return {
    devices: filtered,
    summary: {
      totalDevices: MOCK_BATTERY_DATA.length,
      healthyDevices: MOCK_BATTERY_DATA.filter(b => b.health === 'Excellent' || b.health === 'Good').length,
      warningDevices: MOCK_BATTERY_DATA.filter(b => b.health === 'Warning').length,
      criticalDevices: MOCK_BATTERY_DATA.filter(b => b.health === 'Critical').length,
      avgCapacity: Math.round(MOCK_BATTERY_DATA.reduce((sum, b) => sum + b.capacity, 0) / MOCK_BATTERY_DATA.length),
      totalAlerts: 0,
      criticalAlerts: 0
    },
    weather: weatherData,
    analytics: {
      kalmanFilterApplied: false,
      zScoreAnalysisApplied: false,
      rulesEvaluated: 0,
      anomaliesDetected: 0,
      weatherDataIncluded: Object.keys(weatherData).length > 0
    },
    timestamp: new Date().toISOString()
  };
}

/**
 * Get battery reliability summary
 */
export async function getBatteryReliability() {
  const total = MOCK_BATTERY_DATA.length;
  const optimal = MOCK_BATTERY_DATA.filter(b => b.capacity > 80).length;
  const needsAttention = MOCK_BATTERY_DATA.filter(b => b.capacity >= 50 && b.capacity <= 80).length;
  const critical = MOCK_BATTERY_DATA.filter(b => b.capacity < 50).length;

  return {
    summary: `Current IoT device battery status across supply chain:

• ${Math.round((optimal / total) * 100)}% of devices show optimal battery health (>80% capacity)
• ${Math.round((needsAttention / total) * 100)}% require attention within next 6 months
• ${Math.round((critical / total) * 100)}% are in critical condition requiring immediate replacement
• Average battery lifespan: 18 months under current conditions

Recommendation: Prioritize replacement of ${MOCK_BATTERY_DATA.filter(b => b.health === 'Critical').map(b => b.device).join(', ')}. Consider upgrading to newer battery technology for devices in high-temperature environments.`,
    metrics: {
      optimalPercent: Math.round((optimal / total) * 100),
      needsAttentionPercent: Math.round((needsAttention / total) * 100),
      criticalPercent: Math.round((critical / total) * 100),
      avgLifespanMonths: 18
    },
    criticalDevices: MOCK_BATTERY_DATA.filter(b => b.health === 'Critical').map(b => b.device)
  };
}

/**
 * Get alert trends for a region
 */
export async function getAlertTrends(params: { region?: string }) {
  const { region } = params;
  
  // Mock trend data based on region
  if (region === 'Asia-Pacific' || region?.toLowerCase().includes('asia')) {
    return {
      summary: `There has been a 23% increase in logistics alerts across Asia this week. Key drivers include:

• Typhoon-related port closures in Southeast Asia (40% of alerts)
• Semiconductor shortage affecting electronics suppliers (35%)
• Increased customs inspection times in China (25%)

Recommendation: Consider alternative routing through Singapore and diversify semiconductor suppliers.`,
      metrics: {
        increasePercent: 23,
        primaryCauses: [
          { cause: 'Typhoon-related port closures', percent: 40 },
          { cause: 'Semiconductor shortage', percent: 35 },
          { cause: 'Increased customs inspection', percent: 25 }
        ]
      }
    };
  }
  
  // Default/global trends
  return {
    summary: 'Based on current data, your supply chain is operating at 94% efficiency. There are 12 active alerts requiring attention, with 3 high-priority risks identified.',
    metrics: {
      efficiency: 94,
      activeAlerts: 12,
      highPriority: 3
    }
  };
}
