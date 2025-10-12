// lib/__tests__/batteryAnalytics.test.ts
/**
 * Battery Analytics Test Suite
 * 
 * Tests for Kalman filtering, Z-score analysis, and rule engine
 */

import {
  initKalmanFilter,
  updateKalmanFilter,
  applyKalmanFilter,
  calculateZScore,
  analyzeZScore,
  evaluateAlertRules,
  BATTERY_ALERT_RULES,
  type BatteryTelemetry,
  type KalmanState,
  type ZScoreAnalysis
} from '../batteryAnalytics';

// ============================================================================
// KALMAN FILTER TESTS
// ============================================================================

describe('Kalman Filter', () => {
  describe('initKalmanFilter', () => {
    it('should initialize with correct default values', () => {
      const state = initKalmanFilter(3.7, 0.01, 0.1);
      
      expect(state.x).toBe(3.7);
      expect(state.P).toBe(1.0);
      expect(state.Q).toBe(0.01);
      expect(state.R).toBe(0.1);
      expect(state.K).toBe(0);
    });
    
    it('should accept custom noise parameters', () => {
      const state = initKalmanFilter(100, 0.05, 0.5);
      
      expect(state.Q).toBe(0.05);
      expect(state.R).toBe(0.5);
    });
  });
  
  describe('updateKalmanFilter', () => {
    it('should update state with new measurement', () => {
      let state = initKalmanFilter(3.7, 0.01, 0.1);
      state = updateKalmanFilter(state, 3.75);
      
      expect(state.x).toBeGreaterThan(3.7);
      expect(state.x).toBeLessThan(3.75);
      expect(state.K).toBeGreaterThan(0);
      expect(state.K).toBeLessThan(1);
    });
    
    it('should reduce error covariance over time', () => {
      let state = initKalmanFilter(3.7, 0.01, 0.1);
      const initialP = state.P;
      
      for (let i = 0; i < 10; i++) {
        state = updateKalmanFilter(state, 3.7 + (Math.random() - 0.5) * 0.1);
      }
      
      expect(state.P).toBeLessThan(initialP);
    });
  });
  
  describe('applyKalmanFilter', () => {
    it('should filter a time series of measurements', () => {
      const measurements = [3.7, 3.68, 3.75, 3.69, 3.71, 3.73];
      const filtered = applyKalmanFilter(measurements, 0.01, 0.1);
      
      expect(filtered.length).toBe(measurements.length);
      expect(filtered[0]).toBe(measurements[0]); // First value unchanged
    });
    
    it('should reduce noise variance', () => {
      // Generate noisy signal
      const trueValue = 3.7;
      const noise = 0.2;
      const measurements = Array.from({ length: 30 }, () => 
        trueValue + (Math.random() - 0.5) * noise
      );
      
      const filtered = applyKalmanFilter(measurements, 0.01, 0.1);
      
      // Calculate variance
      const originalVariance = variance(measurements);
      const filteredVariance = variance(filtered);
      
      expect(filteredVariance).toBeLessThan(originalVariance);
    });
    
    it('should handle empty array', () => {
      const filtered = applyKalmanFilter([], 0.01, 0.1);
      expect(filtered).toEqual([]);
    });
    
    it('should handle single measurement', () => {
      const filtered = applyKalmanFilter([3.7], 0.01, 0.1);
      expect(filtered).toEqual([3.7]);
    });
  });
});

// ============================================================================
// Z-SCORE ANALYSIS TESTS
// ============================================================================

describe('Z-Score Analysis', () => {
  describe('calculateZScore', () => {
    it('should calculate correct Z-score', () => {
      const zscore = calculateZScore(85, 80, 5);
      expect(zscore).toBe(1.0);
    });
    
    it('should handle negative Z-scores', () => {
      const zscore = calculateZScore(75, 80, 5);
      expect(zscore).toBe(-1.0);
    });
    
    it('should return 0 for zero standard deviation', () => {
      const zscore = calculateZScore(100, 80, 0);
      expect(zscore).toBe(0);
    });
    
    it('should handle value equal to mean', () => {
      const zscore = calculateZScore(80, 80, 5);
      expect(zscore).toBe(0);
    });
  });
  
  describe('analyzeZScore', () => {
    const historicalValues = [85, 84, 85, 86, 84, 85, 85, 86, 84, 85];
    
    it('should detect normal values', () => {
      const analysis = analyzeZScore(
        'DEVICE-001',
        'capacity',
        85,
        historicalValues,
        2.0
      );
      
      expect(analysis.isAnomaly).toBe(false);
      expect(analysis.severity).toBe('normal');
    });
    
    it('should detect warning level anomalies', () => {
      const analysis = analyzeZScore(
        'DEVICE-001',
        'capacity',
        75, // Significantly below mean
        historicalValues,
        2.0
      );
      
      expect(analysis.isAnomaly).toBe(true);
      expect(analysis.severity).toMatch(/warning|critical/);
    });
    
    it('should detect critical level anomalies', () => {
      const analysis = analyzeZScore(
        'DEVICE-001',
        'capacity',
        60, // Very far from mean
        historicalValues,
        2.0
      );
      
      expect(analysis.isAnomaly).toBe(true);
      expect(analysis.severity).toBe('critical');
    });
    
    it('should calculate correct statistics', () => {
      const analysis = analyzeZScore(
        'DEVICE-001',
        'capacity',
        85,
        historicalValues,
        2.0
      );
      
      expect(analysis.mean).toBeCloseTo(84.9, 1);
      expect(analysis.stdDev).toBeGreaterThan(0);
      expect(analysis.value).toBe(85);
    });
    
    it('should respect custom threshold', () => {
      // With threshold 3.0, need more deviation to trigger
      const analysis = analyzeZScore(
        'DEVICE-001',
        'capacity',
        80,
        historicalValues,
        3.0 // Higher threshold
      );
      
      expect(analysis.isAnomaly).toBe(false);
    });
  });
});

// ============================================================================
// RULE ENGINE TESTS
// ============================================================================

describe('Rule Engine', () => {
  describe('evaluateAlertRules', () => {
    it('should trigger VOLTAGE_CRITICAL_LOW', () => {
      const data: BatteryTelemetry = {
        deviceId: 'TEST-001',
        timestamp: new Date(),
        voltage: 2.7, // Below 2.8V threshold
        capacity: 80,
        temperature: 25,
        chargeCycles: 1000,
        region: 'Test'
      };
      
      const alerts = evaluateAlertRules(data);
      const criticalVoltage = alerts.find(a => a.alertType === 'VOLTAGE_CRITICAL_LOW');
      
      expect(criticalVoltage).toBeDefined();
      expect(criticalVoltage?.severity).toBe('critical');
    });
    
    it('should trigger CAPACITY_CRITICAL', () => {
      const data: BatteryTelemetry = {
        deviceId: 'TEST-002',
        timestamp: new Date(),
        voltage: 3.7,
        capacity: 45, // Below 50% threshold
        temperature: 25,
        chargeCycles: 1000,
        region: 'Test'
      };
      
      const alerts = evaluateAlertRules(data);
      const criticalCapacity = alerts.find(a => a.alertType === 'CAPACITY_CRITICAL');
      
      expect(criticalCapacity).toBeDefined();
      expect(criticalCapacity?.severity).toBe('high');
    });
    
    it('should trigger TEMPERATURE_CRITICAL', () => {
      const data: BatteryTelemetry = {
        deviceId: 'TEST-003',
        timestamp: new Date(),
        voltage: 3.7,
        capacity: 80,
        temperature: 50, // Above 45°C threshold
        chargeCycles: 1000,
        region: 'Test'
      };
      
      const alerts = evaluateAlertRules(data);
      const criticalTemp = alerts.find(a => a.alertType === 'TEMPERATURE_CRITICAL');
      
      expect(criticalTemp).toBeDefined();
      expect(criticalTemp?.severity).toBe('critical');
    });
    
    it('should trigger multiple alerts for severely degraded battery', () => {
      const data: BatteryTelemetry = {
        deviceId: 'TEST-004',
        timestamp: new Date(),
        voltage: 2.7,  // Critical low
        capacity: 40,   // Critical low
        temperature: 46, // Critical high
        chargeCycles: 3500, // Very high
        region: 'Test'
      };
      
      const alerts = evaluateAlertRules(data);
      
      expect(alerts.length).toBeGreaterThan(2);
      expect(alerts.filter(a => a.severity === 'critical').length).toBeGreaterThan(0);
    });
    
    it('should not trigger alerts for healthy battery', () => {
      const data: BatteryTelemetry = {
        deviceId: 'TEST-005',
        timestamp: new Date(),
        voltage: 3.7,
        capacity: 90,
        temperature: 25,
        chargeCycles: 500,
        region: 'Test'
      };
      
      const alerts = evaluateAlertRules(data);
      
      expect(alerts.length).toBe(0);
    });
    
    it('should include action recommendations', () => {
      const data: BatteryTelemetry = {
        deviceId: 'TEST-006',
        timestamp: new Date(),
        voltage: 2.7,
        capacity: 80,
        temperature: 25,
        chargeCycles: 1000,
        region: 'Test'
      };
      
      const alerts = evaluateAlertRules(data);
      const withAction = alerts.find(a => a.action);
      
      expect(withAction).toBeDefined();
      expect(withAction?.action).toBeTruthy();
    });
    
    it('should evaluate Z-score based rules', () => {
      const data: BatteryTelemetry = {
        deviceId: 'TEST-007',
        timestamp: new Date(),
        voltage: 3.2,
        capacity: 80,
        temperature: 25,
        chargeCycles: 1000,
        region: 'Test'
      };
      
      // Create a critical voltage Z-score
      const zscores: ZScoreAnalysis[] = [{
        deviceId: 'TEST-007',
        metric: 'voltage',
        value: 3.2,
        mean: 3.7,
        stdDev: 0.1,
        zScore: -5.0, // Critical deviation
        isAnomaly: true,
        severity: 'critical'
      }];
      
      const alerts = evaluateAlertRules(data, zscores);
      const anomalyAlert = alerts.find(a => a.alertType === 'VOLTAGE_ANOMALY');
      
      expect(anomalyAlert).toBeDefined();
    });
  });
  
  describe('BATTERY_ALERT_RULES', () => {
    it('should have multiple rules defined', () => {
      expect(BATTERY_ALERT_RULES.length).toBeGreaterThan(5);
    });
    
    it('should have rules for all severity levels', () => {
      const severities = BATTERY_ALERT_RULES.map(r => r.severity);
      
      expect(severities).toContain('low');
      expect(severities).toContain('medium');
      expect(severities).toContain('high');
      expect(severities).toContain('critical');
    });
    
    it('should have unique rule names', () => {
      const names = BATTERY_ALERT_RULES.map(r => r.name);
      const uniqueNames = new Set(names);
      
      expect(names.length).toBe(uniqueNames.size);
    });
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration Tests', () => {
  it('should process complete analytics pipeline', () => {
    // 1. Generate simulated measurements with noise
    const trueVoltage = 3.7;
    const measurements = Array.from({ length: 30 }, () => 
      trueVoltage + (Math.random() - 0.5) * 0.2
    );
    
    // 2. Apply Kalman filter
    const filtered = applyKalmanFilter(measurements, 0.01, 0.1);
    expect(filtered.length).toBe(measurements.length);
    
    // 3. Analyze current value with Z-score
    const currentValue = filtered[filtered.length - 1];
    const historical = filtered.slice(0, -1);
    const analysis = analyzeZScore(
      'INTEGRATION-TEST',
      'voltage',
      currentValue,
      historical,
      2.0
    );
    
    expect(analysis).toBeDefined();
    expect(analysis.zScore).toBeDefined();
    
    // 4. Create telemetry data
    const telemetry: BatteryTelemetry = {
      deviceId: 'INTEGRATION-TEST',
      timestamp: new Date(),
      voltage: currentValue,
      capacity: 75,
      temperature: 28,
      chargeCycles: 2000,
      region: 'Test'
    };
    
    // 5. Evaluate rules
    const alerts = evaluateAlertRules(telemetry, [analysis]);
    
    expect(Array.isArray(alerts)).toBe(true);
  });
  
  it('should detect degradation pattern', () => {
    // Simulate capacity degradation over time
    const capacityReadings = [
      90, 89, 88, 89, 87, 86, 85, 84, 83, 82, // Normal degradation
      75, 70, 65, 60 // Sudden drop
    ];
    
    // Analyze last value
    const historical = capacityReadings.slice(0, -1);
    const current = capacityReadings[capacityReadings.length - 1];
    
    const analysis = analyzeZScore(
      'DEGRADATION-TEST',
      'capacity',
      current,
      historical,
      2.0
    );
    
    expect(analysis.isAnomaly).toBe(true);
    expect(analysis.severity).toMatch(/warning|critical/);
  });
});

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function mean(values: number[]): number {
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

function variance(values: number[]): number {
  const avg = mean(values);
  const squaredDiffs = values.map(val => Math.pow(val - avg, 2));
  return mean(squaredDiffs);
}

// ============================================================================
// MOCK DATA TESTS
// ============================================================================

describe('Edge Cases', () => {
  it('should handle extreme voltage values', () => {
    const extremeData: BatteryTelemetry = {
      deviceId: 'EXTREME-001',
      timestamp: new Date(),
      voltage: 0.5, // Very low
      capacity: 0,
      temperature: -40,
      chargeCycles: 10000,
      region: 'Test'
    };
    
    const alerts = evaluateAlertRules(extremeData);
    expect(alerts.length).toBeGreaterThan(0);
  });
  
  it('should handle NaN gracefully in Z-score', () => {
    const zscore = calculateZScore(NaN, 80, 5);
    expect(isNaN(zscore)).toBe(true);
  });
  
  it('should handle empty historical data', () => {
    const analysis = analyzeZScore(
      'EMPTY-TEST',
      'voltage',
      3.7,
      [],
      2.0
    );
    
    expect(analysis.mean).toBe(0);
    expect(analysis.stdDev).toBe(0);
  });
});

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

describe('Performance', () => {
  it('should process Kalman filter efficiently', () => {
    const measurements = Array.from({ length: 1000 }, () => Math.random() * 4);
    
    const startTime = Date.now();
    applyKalmanFilter(measurements, 0.01, 0.1);
    const endTime = Date.now();
    
    expect(endTime - startTime).toBeLessThan(100); // Should be fast
  });
  
  it('should evaluate rules efficiently', () => {
    const data: BatteryTelemetry = {
      deviceId: 'PERF-TEST',
      timestamp: new Date(),
      voltage: 3.5,
      capacity: 70,
      temperature: 30,
      chargeCycles: 2000,
      region: 'Test'
    };
    
    const iterations = 1000;
    const startTime = Date.now();
    
    for (let i = 0; i < iterations; i++) {
      evaluateAlertRules(data);
    }
    
    const endTime = Date.now();
    const avgTime = (endTime - startTime) / iterations;
    
    expect(avgTime).toBeLessThan(1); // Should be < 1ms per evaluation
  });
});
