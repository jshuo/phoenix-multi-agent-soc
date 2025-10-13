// lib/batteryAnalytics.ts
/**
 * Battery Analytics Module
 * Implements advanced signal processing and anomaly detection for IoT battery monitoring
 * 
 * Features:
 * - Kalman Filter for noise reduction in battery telemetry
 * - Z-score residual analysis for deviation scoring
 * - Rule Engine for threshold-based alerting
 */

import { Pool } from 'pg';
import { KalmanFilter } from 'kalman-filter';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface BatteryTelemetry {
  deviceId: string;
  timestamp: Date;
  voltage: number;
  capacity: number;
  temperature: number;
  chargeCycles: number;
  region: string;
  rawVoltage?: number;  // Original voltage before filtering
  rawCapacity?: number; // Original capacity before filtering
}

export interface KalmanState {
  x: number;      // State estimate
  P: number;      // Error covariance
  Q: number;      // Process noise covariance
  R: number;      // Measurement noise covariance
  K: number;      // Kalman gain
}

export interface ZScoreAnalysis {
  deviceId: string;
  metric: string;
  value: number;
  mean: number;
  stdDev: number;
  zScore: number;
  isAnomaly: boolean;
  severity: 'normal' | 'warning' | 'critical';
}

export interface AlertRule {
  name: string;
  condition: (data: BatteryTelemetry, zscore?: ZScoreAnalysis) => boolean;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: (data: BatteryTelemetry) => string;
  action?: string;
}

export interface BatteryAlert {
  deviceId: string;
  alertType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  action?: string;
  metrics: {
    voltage?: number;
    capacity?: number;
    temperature?: number;
    zScore?: number;
  };
}

export interface BatteryPerformanceResult {
  devices: Array<{
    device: string;
    voltage: number;
    capacity: number;
    temperature: number;
    cycles: number;
    health: string;
    predictedLife: string;
    region: string;
    filteredVoltage: number;
    filteredCapacity: number;
    voltageZScore: number;
    capacityZScore: number;
    temperatureZScore: number;
    alerts: BatteryAlert[];
  }>;
  summary: {
    totalDevices: number;
    healthyDevices: number;
    warningDevices: number;
    criticalDevices: number;
    avgCapacity: number;
    totalAlerts: number;
    criticalAlerts: number;
  };
  analytics: {
    kalmanFilterApplied: boolean;
    zScoreAnalysisApplied: boolean;
    rulesEvaluated: number;
    anomaliesDetected: number;
  };
  timestamp: string;
}

// ============================================================================
// DATABASE CONNECTION
// ============================================================================

const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME || 'supply_chain_iot',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// ============================================================================
// KALMAN FILTER IMPLEMENTATION (using kalman-filter library)
// ============================================================================

/**
 * Initialize Kalman filter state
 * 
 * @param initialValue - Initial measurement value
 * @param processNoise - Q: How much the system changes (default: 0.01)
 * @param measurementNoise - R: How noisy the measurements are (default: 0.1)
 * @returns Initial Kalman state
 */
export function initKalmanFilter(
  initialValue: number,
  processNoise: number = 0.01,
  measurementNoise: number = 0.1
): KalmanState {
  return {
    x: initialValue,           // Initial state estimate
    P: 1.0,                     // Initial error covariance
    Q: processNoise,            // Process noise covariance
    R: measurementNoise,        // Measurement noise covariance
    K: 0                        // Kalman gain (will be calculated)
  };
}

/**
 * Update Kalman filter with new measurement
 * 
 * @param state - Current Kalman filter state
 * @param measurement - New measurement value
 * @returns Updated Kalman state with filtered value
 */
export function updateKalmanFilter(
  state: KalmanState,
  measurement: number
): KalmanState {
  // Prediction step
  const x_pred = state.x;                    // State prediction (assuming constant model)
  const P_pred = state.P + state.Q;          // Error covariance prediction
  
  // Update step
  const K = P_pred / (P_pred + state.R);     // Kalman gain
  const x = x_pred + K * (measurement - x_pred); // State update
  const P = (1 - K) * P_pred;                // Error covariance update
  
  return {
    x,
    P,
    Q: state.Q,
    R: state.R,
    K
  };
}

/**
 * Apply Kalman filter to a time series of measurements using kalman-filter library
 * 
 * @param measurements - Array of measurement values
 * @param processNoise - Q parameter (process noise covariance)
 * @param measurementNoise - R parameter (measurement noise covariance)
 * @returns Array of filtered values
 */
export function applyKalmanFilter(
  measurements: number[],
  processNoise: number = 0.01,
  measurementNoise: number = 0.1
): number[] {
  if (measurements.length === 0) {
    console.log('[Kalman Filter] Empty measurements array, returning empty result');
    return [];
  }
  
  console.log('[Kalman Filter] Starting filtering process');
  console.log(`[Kalman Filter] Input: ${measurements.length} measurements`);
  console.log(`[Kalman Filter] Parameters: Q=${processNoise}, R=${measurementNoise}`);
  console.log(`[Kalman Filter] First measurement: ${measurements[0]}, Last: ${measurements[measurements.length - 1]}`);
  
  // Configure Kalman filter for 1D scalar tracking
  // The library expects observations as arrays of arrays for multi-dimensional support
  const kf = new KalmanFilter({
    observation: {
      name: 'sensor',
      sensorDimension: 1,  // 1D measurement
      dimension: 1          // 1D state
    },
    dynamic: {
      name: 'constant-position',  // Constant position model (no velocity/acceleration)
      dimension: 1                 // 1D state
    },
    observationCovariance: [[measurementNoise]], // R: measurement noise
    dynamicCovariance: [[processNoise]]          // Q: process noise
  });
  
  console.log('[Kalman Filter] Filter configuration created');
  
  // Convert measurements to the format expected by the library
  const observationsArray = measurements.map(m => [m]);
  
  // Apply the filter to all measurements
  const filtered = kf.filterAll(observationsArray);
  
  console.log('[Kalman Filter] Filtering complete');
  
  // Convert back to simple array of numbers
  const result = filtered.map(f => f[0]);
  
  // Calculate and log statistics
  const originalMean = measurements.reduce((sum, v) => sum + v, 0) / measurements.length;
  const filteredMean = result.reduce((sum, v) => sum + v, 0) / result.length;
  const noiseReduction = Math.abs(originalMean - filteredMean);
  
  console.log(`[Kalman Filter] Original mean: ${originalMean.toFixed(4)}`);
  console.log(`[Kalman Filter] Filtered mean: ${filteredMean.toFixed(4)}`);
  console.log(`[Kalman Filter] Noise reduction: ${noiseReduction.toFixed(4)}`);
  console.log(`[Kalman Filter] Output: ${result.length} filtered values`);
  console.log(`[Kalman Filter] First filtered: ${result[0].toFixed(4)}, Last: ${result[result.length - 1].toFixed(4)}`);
  
  return result;
}

// ============================================================================
// Z-SCORE RESIDUAL ANALYSIS
// ============================================================================

/**
 * Calculate mean of an array
 */
function calculateMean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate standard deviation of an array
 */
function calculateStdDev(values: number[], mean?: number): number {
  if (values.length === 0) return 0;
  
  const avg = mean ?? calculateMean(values);
  const squaredDiffs = values.map(val => Math.pow(val - avg, 2));
  const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
  
  return Math.sqrt(variance);
}

/**
 * Calculate Z-score for a value
 * 
 * Z-score = (value - mean) / standard_deviation
 * 
 * @param value - The value to score
 * @param mean - Mean of the distribution
 * @param stdDev - Standard deviation of the distribution
 * @returns Z-score
 */
export function calculateZScore(
  value: number,
  mean: number,
  stdDev: number
): number {
  if (stdDev === 0) return 0;
  return (value - mean) / stdDev;
}

/**
 * Perform Z-score analysis on battery metrics
 * 
 * @param deviceId - Device identifier
 * @param metric - Metric name (e.g., 'voltage', 'capacity')
 * @param value - Current value
 * @param historicalValues - Array of historical values for comparison
 * @param threshold - Z-score threshold for anomaly detection (default: 2.0)
 * @returns Z-score analysis result
 */
export function analyzeZScore(
  deviceId: string,
  metric: string,
  value: number,
  historicalValues: number[],
  threshold: number = 2.0
): ZScoreAnalysis {
  const mean = calculateMean(historicalValues);
  const stdDev = calculateStdDev(historicalValues, mean);
  const zScore = calculateZScore(value, mean, stdDev);
  const absZScore = Math.abs(zScore);
  
  // Determine severity based on Z-score
  let severity: 'normal' | 'warning' | 'critical' = 'normal';
  if (absZScore > 3.0) {
    severity = 'critical';  // >3 std devs (99.7% confidence)
  } else if (absZScore > threshold) {
    severity = 'warning';   // >2 std devs (95% confidence)
  }
  
  // 📊 Log Z-score analysis
  console.log(`[Z-Score Analysis] Device: ${deviceId} | Metric: ${metric}`);
  console.log(`  Current value: ${value.toFixed(2)}`);
  console.log(`  Historical mean: ${mean.toFixed(2)} (±${stdDev.toFixed(2)} std dev)`);
  console.log(`  Z-Score: ${zScore.toFixed(3)} | Severity: ${severity.toUpperCase()}`);
  if (absZScore > threshold) {
    console.log(`  ⚠️  ANOMALY DETECTED! |Z| = ${absZScore.toFixed(3)} > ${threshold}`);
  }
  
  return {
    deviceId,
    metric,
    value,
    mean,
    stdDev,
    zScore,
    isAnomaly: absZScore > threshold,
    severity
  };
}

// ============================================================================
// RULE ENGINE
// ============================================================================

/**
 * Battery health classification based on capacity
 */
function classifyHealth(capacity: number): string {
  if (capacity >= 90) return 'Excellent';
  if (capacity >= 70) return 'Good';
  if (capacity >= 50) return 'Warning';
  return 'Critical';
}

/**
 * Predict remaining battery life based on capacity and cycles
 */
function predictRemainingLife(capacity: number, cycles: number): string {
  // Simple heuristic: life expectancy decreases with cycles and low capacity
  const maxCycles = 3500;
  const cyclesFactor = 1 - (cycles / maxCycles);
  const capacityFactor = capacity / 100;
  
  const estimatedMonths = Math.max(1, Math.round(24 * cyclesFactor * capacityFactor));
  
  if (estimatedMonths >= 12) return `${estimatedMonths} months`;
  if (estimatedMonths > 1) return `${estimatedMonths} months`;
  return '1 month';
}

/**
 * Define alert rules for battery monitoring
 */
export const BATTERY_ALERT_RULES: AlertRule[] = [
  // Critical voltage rules
  {
    name: 'VOLTAGE_CRITICAL_LOW',
    condition: (data) => data.voltage < 2.8,
    severity: 'critical',
    message: (data) => 
      `Critical low voltage detected: ${data.voltage.toFixed(2)}V (threshold: 2.8V)`,
    action: 'IMMEDIATE_REPLACEMENT_REQUIRED'
  },
  {
    name: 'VOLTAGE_CRITICAL_HIGH',
    condition: (data) => data.voltage > 4.2,
    severity: 'critical',
    message: (data) => 
      `Critical high voltage detected: ${data.voltage.toFixed(2)}V (threshold: 4.2V)`,
    action: 'DISCONNECT_IMMEDIATELY'
  },
  
  // High severity rules
  {
    name: 'CAPACITY_CRITICAL',
    condition: (data) => data.capacity < 50,
    severity: 'high',
    message: (data) => 
      `Battery capacity critically low: ${data.capacity}% - Replacement recommended`,
    action: 'SCHEDULE_REPLACEMENT'
  },
  {
    name: 'TEMPERATURE_HIGH',
    condition: (data) => data.temperature > 35,
    severity: 'high',
    message: (data) => 
      `High temperature detected: ${data.temperature}°C - Risk of accelerated degradation`,
    action: 'MONITOR_CLOSELY'
  },
  {
    name: 'TEMPERATURE_CRITICAL',
    condition: (data) => data.temperature > 45,
    severity: 'critical',
    message: (data) => 
      `Critical temperature: ${data.temperature}°C - Thermal runaway risk`,
    action: 'SHUTDOWN_DEVICE'
  },
  
  // Medium severity rules
  {
    name: 'CAPACITY_LOW',
    condition: (data) => data.capacity >= 50 && data.capacity < 70,
    severity: 'medium',
    message: (data) => 
      `Battery capacity degraded: ${data.capacity}% - Plan replacement within 3 months`,
    action: 'PLAN_REPLACEMENT'
  },
  {
    name: 'VOLTAGE_LOW',
    condition: (data) => data.voltage >= 2.8 && data.voltage < 3.0,
    severity: 'medium',
    message: (data) => 
      `Low voltage warning: ${data.voltage.toFixed(2)}V - Monitor closely`,
    action: 'MONITOR'
  },
  {
    name: 'HIGH_CYCLE_COUNT',
    condition: (data) => data.chargeCycles > 3000,
    severity: 'medium',
    message: (data) => 
      `High charge cycle count: ${data.chargeCycles} cycles - Battery aging`,
    action: 'SCHEDULE_INSPECTION'
  },
  
  // Z-score based anomaly rules
  {
    name: 'VOLTAGE_ANOMALY',
    condition: (data, zscore) => 
      zscore?.metric === 'voltage' && zscore.severity === 'critical',
    severity: 'high',
    message: (data) => 
      `Voltage anomaly detected: ${data.voltage.toFixed(2)}V - Unusual deviation from normal pattern`,
    action: 'INVESTIGATE'
  },
  {
    name: 'CAPACITY_ANOMALY',
    condition: (data, zscore) => 
      zscore?.metric === 'capacity' && zscore.severity === 'critical',
    severity: 'high',
    message: (data) => 
      `Capacity anomaly detected: ${data.capacity}% - Unexpected degradation pattern`,
    action: 'INVESTIGATE'
  }
];

/**
 * Evaluate battery data against alert rules
 * 
 * @param data - Battery telemetry data
 * @param zscores - Array of Z-score analyses
 * @returns Array of triggered alerts
 */
export function evaluateAlertRules(
  data: BatteryTelemetry,
  zscores: ZScoreAnalysis[] = []
): BatteryAlert[] {
  const alerts: BatteryAlert[] = [];
  
  for (const rule of BATTERY_ALERT_RULES) {
    // Find relevant Z-score for this rule
    const relevantZScore = zscores.find(z => 
      rule.name.toLowerCase().includes(z.metric.toLowerCase())
    );
    
    // Evaluate rule condition
    if (rule.condition(data, relevantZScore)) {
      alerts.push({
        deviceId: data.deviceId,
        alertType: rule.name,
        severity: rule.severity,
        message: rule.message(data),
        timestamp: new Date(),
        action: rule.action,
        metrics: {
          voltage: data.voltage,
          capacity: data.capacity,
          temperature: data.temperature,
          zScore: relevantZScore?.zScore
        }
      });
    }
  }
  
  return alerts;
}

// ============================================================================
// MAIN BATTERY PERFORMANCE FUNCTION
// ============================================================================

/**
 * Get battery performance data from SQL database with advanced analytics
 * 
 * @param params - Query parameters
 * @returns Battery performance analysis results
 */
export async function getBatteryPerformance(params: {
  region?: string;
  health?: string;
  deviceId?: string;
  limit?: number;
  applyKalman?: boolean;
  applyZScore?: boolean;
  applyRules?: boolean;
}): Promise<BatteryPerformanceResult> {
  const {
    region,
    health,
    deviceId,
    limit = 100,
    applyKalman = true,
    applyZScore = true,
    applyRules = true
  } = params;

  try {
    // ========================================================================
    // STEP 1: FETCH DATA FROM SQL DATABASE
    // ========================================================================
    
    let query = `
      SELECT 
        device_id,
        timestamp,
        voltage,
        capacity_percent as capacity,
        temperature_celsius as temperature,
        charge_cycles,
        region
      FROM battery_telemetry
      WHERE 1=1
    `;
    
    const queryParams: any[] = [];
    let paramIndex = 1;
    
    if (region) {
      query += ` AND region = $${paramIndex}`;
      queryParams.push(region);
      paramIndex++;
    }
    
    if (deviceId) {
      query += ` AND device_id = $${paramIndex}`;
      queryParams.push(deviceId);
      paramIndex++;
    }
    
    // Get recent data (last 7 days for current state)
    query += ` AND timestamp > NOW() - INTERVAL '7 days'`;
    query += ` ORDER BY device_id, timestamp DESC`;
    query += ` LIMIT $${paramIndex}`;
    queryParams.push(limit);
    
    const result = await pool.query(query, queryParams);
    
    if (result.rows.length === 0) {
      return {
        devices: [],
        summary: {
          totalDevices: 0,
          healthyDevices: 0,
          warningDevices: 0,
          criticalDevices: 0,
          avgCapacity: 0,
          totalAlerts: 0,
          criticalAlerts: 0
        },
        analytics: {
          kalmanFilterApplied: false,
          zScoreAnalysisApplied: false,
          rulesEvaluated: 0,
          anomaliesDetected: 0
        },
        timestamp: new Date().toISOString()
      };
    }
    
    // ========================================================================
    // STEP 2: ORGANIZE DATA BY DEVICE
    // ========================================================================
    
    const deviceMap = new Map<string, BatteryTelemetry[]>();
    
    for (const row of result.rows) {
      const telemetry: BatteryTelemetry = {
        deviceId: row.device_id,
        timestamp: row.timestamp,
        voltage: parseFloat(row.voltage),
        capacity: parseFloat(row.capacity),
        temperature: parseFloat(row.temperature),
        chargeCycles: parseInt(row.charge_cycles),
        region: row.region
      };
      
      if (!deviceMap.has(telemetry.deviceId)) {
        deviceMap.set(telemetry.deviceId, []);
      }
      deviceMap.get(telemetry.deviceId)!.push(telemetry);
    }
    
    // ========================================================================
    // STEP 3: APPLY KALMAN FILTER FOR NOISE REDUCTION
    // ========================================================================
    
    const processedDevices = [];
    let totalAnomalies = 0;
    let totalAlertCount = 0;
    let criticalAlertCount = 0;
    
    for (const [deviceId, telemetryArray] of deviceMap.entries()) {
      // Sort by timestamp (oldest first for time series processing)
      telemetryArray.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
      
      const latestData = telemetryArray[telemetryArray.length - 1];
      let processedData = { ...latestData };
      
      // Store raw values
      processedData.rawVoltage = latestData.voltage;
      processedData.rawCapacity = latestData.capacity;
      
      if (applyKalman && telemetryArray.length > 1) {
        console.log(`\n[Battery Analytics] Processing device: ${deviceId}`);
        console.log(`[Battery Analytics] Time series length: ${telemetryArray.length} samples`);
        
        // Apply Kalman filter to voltage
        console.log(`[Battery Analytics] Applying Kalman filter to VOLTAGE data`);
        const voltages = telemetryArray.map(t => t.voltage);
        const filteredVoltages = applyKalmanFilter(voltages, 0.01, 0.1);
        processedData.voltage = filteredVoltages[filteredVoltages.length - 1];
        console.log(`[Battery Analytics] Voltage - Raw: ${latestData.voltage.toFixed(3)}V → Filtered: ${processedData.voltage.toFixed(3)}V`);
        
        // Apply Kalman filter to capacity
        console.log(`[Battery Analytics] Applying Kalman filter to CAPACITY data`);
        const capacities = telemetryArray.map(t => t.capacity);
        const filteredCapacities = applyKalmanFilter(capacities, 0.005, 0.05);
        processedData.capacity = filteredCapacities[filteredCapacities.length - 1];
        console.log(`[Battery Analytics] Capacity - Raw: ${latestData.capacity.toFixed(1)}% → Filtered: ${processedData.capacity.toFixed(1)}%`);
      } else if (applyKalman) {
        console.log(`[Battery Analytics] Skipping Kalman filter for device ${deviceId} - insufficient data (only ${telemetryArray.length} sample)`);
      }
      
      // ======================================================================
      // STEP 4: Z-SCORE RESIDUAL ANALYSIS
      // ======================================================================
      
      const zscores: ZScoreAnalysis[] = [];
      
      if (applyZScore && telemetryArray.length > 5) {
        // Voltage Z-score analysis
        const voltages = telemetryArray.slice(0, -1).map(t => t.voltage);
        const voltageZScore = analyzeZScore(
          deviceId,
          'voltage',
          processedData.voltage,
          voltages,
          2.0
        );
        zscores.push(voltageZScore);
        
        // Capacity Z-score analysis
        const capacities = telemetryArray.slice(0, -1).map(t => t.capacity);
        const capacityZScore = analyzeZScore(
          deviceId,
          'capacity',
          processedData.capacity,
          capacities,
          2.0
        );
        zscores.push(capacityZScore);
        
        // Temperature Z-score analysis
        const temperatures = telemetryArray.slice(0, -1).map(t => t.temperature);
        const temperatureZScore = analyzeZScore(
          deviceId,
          'temperature',
          processedData.temperature,
          temperatures,
          2.0
        );
        zscores.push(temperatureZScore);
        
        // Count anomalies
        totalAnomalies += zscores.filter(z => z.isAnomaly).length;
      }
      
      // ======================================================================
      // STEP 5: RULE ENGINE EVALUATION
      // ======================================================================
      
      let alerts: BatteryAlert[] = [];
      
      if (applyRules) {
        alerts = evaluateAlertRules(processedData, zscores);
        totalAlertCount += alerts.length;
        criticalAlertCount += alerts.filter(a => a.severity === 'critical').length;
      }
      
      // ======================================================================
      // STEP 6: CLASSIFY HEALTH AND PREDICT LIFE
      // ======================================================================
      
      const healthStatus = classifyHealth(processedData.capacity);
      const predictedLife = predictRemainingLife(
        processedData.capacity,
        processedData.chargeCycles
      );
      
      // Filter by health if specified
      if (health && healthStatus !== health) {
        continue;
      }
      
      processedDevices.push({
        device: deviceId,
        voltage: Math.round(processedData.voltage * 100) / 100,
        capacity: Math.round(processedData.capacity),
        temperature: Math.round(processedData.temperature),
        cycles: processedData.chargeCycles,
        health: healthStatus,
        predictedLife,
        region: processedData.region,
        filteredVoltage: Math.round(processedData.voltage * 100) / 100,
        filteredCapacity: Math.round(processedData.capacity),
        voltageZScore: zscores.find(z => z.metric === 'voltage')?.zScore || 0,
        capacityZScore: zscores.find(z => z.metric === 'capacity')?.zScore || 0,
        temperatureZScore: zscores.find(z => z.metric === 'temperature')?.zScore || 0,
        alerts
      });
    }
    
    // ========================================================================
    // STEP 7: CALCULATE SUMMARY STATISTICS
    // ========================================================================
    
    const totalDevices = processedDevices.length;
    const healthyDevices = processedDevices.filter(
      d => d.health === 'Excellent' || d.health === 'Good'
    ).length;
    const warningDevices = processedDevices.filter(d => d.health === 'Warning').length;
    const criticalDevices = processedDevices.filter(d => d.health === 'Critical').length;
    const avgCapacity = totalDevices > 0
      ? Math.round(
          processedDevices.reduce((sum, d) => sum + d.capacity, 0) / totalDevices
        )
      : 0;
    
    // Log summary
    console.log('\n[Battery Analytics] ========================================');
    console.log('[Battery Analytics] PROCESSING COMPLETE');
    console.log('[Battery Analytics] ========================================');
    console.log(`[Battery Analytics] Total devices processed: ${totalDevices}`);
    console.log(`[Battery Analytics] Healthy devices: ${healthyDevices}`);
    console.log(`[Battery Analytics] Warning devices: ${warningDevices}`);
    console.log(`[Battery Analytics] Critical devices: ${criticalDevices}`);
    console.log(`[Battery Analytics] Average capacity: ${avgCapacity}%`);
    console.log(`[Battery Analytics] Total alerts: ${totalAlertCount}`);
    console.log(`[Battery Analytics] Critical alerts: ${criticalAlertCount}`);
    console.log(`[Battery Analytics] Anomalies detected: ${totalAnomalies}`);
    console.log(`[Battery Analytics] Kalman filter applied: ${applyKalman ? 'YES' : 'NO'}`);
    console.log(`[Battery Analytics] Z-score analysis applied: ${applyZScore ? 'YES' : 'NO'}`);
    console.log(`[Battery Analytics] Rules evaluated: ${BATTERY_ALERT_RULES.length}`);
    console.log('[Battery Analytics] ========================================\n');
    
    return {
      devices: processedDevices,
      summary: {
        totalDevices,
        healthyDevices,
        warningDevices,
        criticalDevices,
        avgCapacity,
        totalAlerts: totalAlertCount,
        criticalAlerts: criticalAlertCount
      },
      analytics: {
        kalmanFilterApplied: applyKalman,
        zScoreAnalysisApplied: applyZScore,
        rulesEvaluated: BATTERY_ALERT_RULES.length,
        anomaliesDetected: totalAnomalies
      },
      timestamp: new Date().toISOString()
    };
    
  } catch (error) {
    console.error('Error fetching battery performance data:', error);
    throw error;
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Close database connection pool
 */
export async function closePool(): Promise<void> {
  await pool.end();
}

/**
 * Test database connection
 */
export async function testConnection(): Promise<boolean> {
  try {
    const result = await pool.query('SELECT NOW()');
    return result.rows.length > 0;
  } catch (error) {
    console.error('Database connection test failed:', error);
    return false;
  }
}
