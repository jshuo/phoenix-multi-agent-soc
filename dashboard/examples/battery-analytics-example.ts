// examples/battery-analytics-example.ts
/**
 * Battery Analytics Example
 * 
 * This file demonstrates how to use the advanced battery analytics system
 * with Kalman filtering, Z-score analysis, and rule-based alerting.
 */

import {
  getBatteryPerformance,
  initKalmanFilter,
  updateKalmanFilter,
  applyKalmanFilter,
  analyzeZScore,
  evaluateAlertRules,
  BATTERY_ALERT_RULES,
  testConnection,
  closePool
} from '../lib/batteryAnalytics';

// ============================================================================
// EXAMPLE 1: Basic Usage - Get Battery Performance with All Analytics
// ============================================================================

export async function example1_basicUsage() {
  console.log('\n=== Example 1: Basic Battery Performance Query ===\n');
  
  try {
    // Fetch battery performance with all analytics enabled
    const result = await getBatteryPerformance({
      region: 'Asia-Pacific',
      applyKalman: true,
      applyZScore: true,
      applyRules: true,
      limit: 10
    });
    
    console.log(`Found ${result.devices.length} devices in Asia-Pacific region`);
    console.log(`\nSummary:`);
    console.log(`  Total Devices: ${result.summary.totalDevices}`);
    console.log(`  Healthy: ${result.summary.healthyDevices}`);
    console.log(`  Warning: ${result.summary.warningDevices}`);
    console.log(`  Critical: ${result.summary.criticalDevices}`);
    console.log(`  Average Capacity: ${result.summary.avgCapacity}%`);
    console.log(`  Total Alerts: ${result.summary.totalAlerts}`);
    console.log(`  Critical Alerts: ${result.summary.criticalAlerts}`);
    
    console.log(`\nAnalytics Applied:`);
    console.log(`  Kalman Filter: ${result.analytics.kalmanFilterApplied}`);
    console.log(`  Z-Score Analysis: ${result.analytics.zScoreAnalysisApplied}`);
    console.log(`  Rules Evaluated: ${result.analytics.rulesEvaluated}`);
    console.log(`  Anomalies Detected: ${result.analytics.anomaliesDetected}`);
    
    // Show details for devices with alerts
    console.log(`\n--- Devices with Alerts ---`);
    result.devices
      .filter(d => d.alerts.length > 0)
      .forEach(device => {
        console.log(`\n${device.device} (${device.region}):`);
        console.log(`  Health: ${device.health}`);
        console.log(`  Voltage: ${device.voltage}V (filtered: ${device.filteredVoltage}V)`);
        console.log(`  Capacity: ${device.capacity}%`);
        console.log(`  Temperature: ${device.temperature}°C`);
        console.log(`  Z-Scores: V=${device.voltageZScore.toFixed(2)}, C=${device.capacityZScore.toFixed(2)}, T=${device.temperatureZScore.toFixed(2)}`);
        console.log(`  Alerts (${device.alerts.length}):`);
        device.alerts.forEach(alert => {
          console.log(`    [${alert.severity.toUpperCase()}] ${alert.alertType}`);
          console.log(`      ${alert.message}`);
          if (alert.action) {
            console.log(`      Action: ${alert.action}`);
          }
        });
      });
    
    return result;
  } catch (error) {
    console.error('Error in example 1:', error);
    throw error;
  }
}

// ============================================================================
// EXAMPLE 2: Kalman Filter Demonstration
// ============================================================================

export function example2_kalmanFilter() {
  console.log('\n=== Example 2: Kalman Filter Demonstration ===\n');
  
  // Simulated noisy voltage measurements
  const noisyVoltages = [
    3.72, 3.68, 3.75, 3.69, 3.71,
    3.73, 3.67, 3.74, 3.70, 3.72,
    3.68, 3.71, 3.73, 3.69, 3.70
  ];
  
  console.log('Original noisy measurements:', noisyVoltages);
  
  // Apply Kalman filter
  const filteredVoltages = applyKalmanFilter(
    noisyVoltages,
    0.01,  // Process noise (Q)
    0.1    // Measurement noise (R)
  );
  
  console.log('Filtered measurements:', filteredVoltages.map(v => v.toFixed(3)));
  
  // Calculate noise reduction
  const originalVariance = calculateVariance(noisyVoltages);
  const filteredVariance = calculateVariance(filteredVoltages);
  const noiseReduction = ((originalVariance - filteredVariance) / originalVariance * 100).toFixed(1);
  
  console.log(`\nNoise reduction: ${noiseReduction}%`);
  console.log(`Original variance: ${originalVariance.toFixed(6)}`);
  console.log(`Filtered variance: ${filteredVariance.toFixed(6)}`);
  
  return { original: noisyVoltages, filtered: filteredVoltages };
}

// ============================================================================
// EXAMPLE 3: Z-Score Analysis Demonstration
// ============================================================================

export function example3_zScoreAnalysis() {
  console.log('\n=== Example 3: Z-Score Analysis Demonstration ===\n');
  
  // Historical capacity measurements for a device (30 days)
  const historicalCapacities = [
    85, 84, 85, 86, 84, 85, 85, 86, 84, 85,
    84, 85, 86, 85, 84, 85, 85, 84, 86, 85,
    84, 85, 85, 84, 85, 86, 84, 85, 85, 84
  ];
  
  // Sudden drop in capacity (anomaly)
  const currentCapacity = 72;
  
  console.log('Historical capacity range:', 
    Math.min(...historicalCapacities), '-', Math.max(...historicalCapacities), '%');
  console.log('Current capacity:', currentCapacity, '%');
  
  // Analyze Z-score
  const analysis = analyzeZScore(
    'GPS-TRACKER-001',
    'capacity',
    currentCapacity,
    historicalCapacities,
    2.0  // Threshold
  );
  
  console.log('\nZ-Score Analysis Results:');
  console.log(`  Mean: ${analysis.mean.toFixed(2)}%`);
  console.log(`  Std Dev: ${analysis.stdDev.toFixed(2)}%`);
  console.log(`  Z-Score: ${analysis.zScore.toFixed(2)}`);
  console.log(`  Is Anomaly: ${analysis.isAnomaly}`);
  console.log(`  Severity: ${analysis.severity}`);
  
  if (analysis.isAnomaly) {
    console.log(`\n⚠️  ALERT: Capacity drop detected!`);
    console.log(`  The device capacity is ${Math.abs(analysis.zScore).toFixed(1)} standard deviations below normal.`);
    console.log(`  This indicates ${analysis.severity === 'critical' ? 'CRITICAL' : 'WARNING'} degradation.`);
  }
  
  return analysis;
}

// ============================================================================
// EXAMPLE 4: Rule Engine Demonstration
// ============================================================================

export function example4_ruleEngine() {
  console.log('\n=== Example 4: Rule Engine Demonstration ===\n');
  
  // Test scenarios
  const testScenarios = [
    {
      name: 'Healthy Device',
      data: {
        deviceId: 'TEMP-SENSOR-001',
        timestamp: new Date(),
        voltage: 3.7,
        capacity: 85,
        temperature: 25,
        chargeCycles: 1200,
        region: 'North America'
      }
    },
    {
      name: 'Low Capacity Warning',
      data: {
        deviceId: 'GPS-TRACKER-002',
        timestamp: new Date(),
        voltage: 3.5,
        capacity: 65,
        temperature: 28,
        chargeCycles: 2500,
        region: 'Asia-Pacific'
      }
    },
    {
      name: 'Critical Battery',
      data: {
        deviceId: 'PRESSURE-MONITOR-003',
        timestamp: new Date(),
        voltage: 2.9,
        capacity: 42,
        temperature: 38,
        chargeCycles: 3200,
        region: 'Europe'
      }
    },
    {
      name: 'Thermal Runaway Risk',
      data: {
        deviceId: 'HUMIDITY-SENSOR-004',
        timestamp: new Date(),
        voltage: 3.8,
        capacity: 75,
        temperature: 48,
        chargeCycles: 1800,
        region: 'Asia-Pacific'
      }
    }
  ];
  
  console.log(`Evaluating ${BATTERY_ALERT_RULES.length} rules against ${testScenarios.length} scenarios...\n`);
  
  testScenarios.forEach(scenario => {
    console.log(`\n--- ${scenario.name} ---`);
    console.log(`Device: ${scenario.data.deviceId}`);
    console.log(`Metrics: V=${scenario.data.voltage}V, C=${scenario.data.capacity}%, T=${scenario.data.temperature}°C, Cycles=${scenario.data.chargeCycles}`);
    
    const alerts = evaluateAlertRules(scenario.data);
    
    if (alerts.length === 0) {
      console.log('✅ No alerts - device is healthy');
    } else {
      console.log(`⚠️  ${alerts.length} alert(s) triggered:`);
      alerts.forEach(alert => {
        console.log(`  [${alert.severity.toUpperCase()}] ${alert.alertType}`);
        console.log(`    ${alert.message}`);
        if (alert.action) {
          console.log(`    → Action: ${alert.action}`);
        }
      });
    }
  });
}

// ============================================================================
// EXAMPLE 5: Real-time Monitoring Simulation
// ============================================================================

export async function example5_realtimeMonitoring() {
  console.log('\n=== Example 5: Real-time Monitoring Simulation ===\n');
  
  try {
    // Monitor critical devices only
    const result = await getBatteryPerformance({
      health: 'Critical',
      applyKalman: true,
      applyZScore: true,
      applyRules: true
    });
    
    console.log(`🔴 Critical Devices Report`);
    console.log(`Found ${result.devices.length} critical devices requiring immediate attention\n`);
    
    if (result.devices.length === 0) {
      console.log('✅ No critical devices at this time');
      return result;
    }
    
    // Prioritize by number of critical alerts
    const prioritized = result.devices
      .sort((a, b) => {
        const aCritical = a.alerts.filter(alert => alert.severity === 'critical').length;
        const bCritical = b.alerts.filter(alert => alert.severity === 'critical').length;
        return bCritical - aCritical;
      });
    
    prioritized.forEach((device, index) => {
      const criticalAlerts = device.alerts.filter(a => a.severity === 'critical');
      const highAlerts = device.alerts.filter(a => a.severity === 'high');
      
      console.log(`${index + 1}. ${device.device} - ${device.region}`);
      console.log(`   Priority: ${criticalAlerts.length > 0 ? '🔴 CRITICAL' : '🟠 HIGH'}`);
      console.log(`   Capacity: ${device.capacity}%`);
      console.log(`   Voltage: ${device.voltage}V`);
      console.log(`   Temperature: ${device.temperature}°C`);
      console.log(`   Predicted Life: ${device.predictedLife}`);
      console.log(`   Critical Alerts: ${criticalAlerts.length}`);
      console.log(`   High Alerts: ${highAlerts.length}`);
      
      if (criticalAlerts.length > 0) {
        console.log(`   Immediate Actions Required:`);
        criticalAlerts.forEach(alert => {
          console.log(`     • ${alert.action || 'Review required'}`);
        });
      }
      console.log('');
    });
    
    return result;
  } catch (error) {
    console.error('Error in example 5:', error);
    throw error;
  }
}

// ============================================================================
// EXAMPLE 6: Regional Comparison
// ============================================================================

export async function example6_regionalComparison() {
  console.log('\n=== Example 6: Regional Comparison ===\n');
  
  const regions = ['Asia-Pacific', 'North America', 'Europe'];
  
  try {
    const regionalData = await Promise.all(
      regions.map(region =>
        getBatteryPerformance({
          region,
          applyKalman: true,
          applyZScore: true,
          applyRules: true
        })
      )
    );
    
    console.log('Battery Health Comparison by Region:\n');
    console.log('Region'.padEnd(20), 
                'Devices', 
                'Healthy', 
                'Warning', 
                'Critical', 
                'Avg Cap', 
                'Alerts');
    console.log('-'.repeat(80));
    
    regions.forEach((region, index) => {
      const data = regionalData[index];
      console.log(
        region.padEnd(20),
        data.summary.totalDevices.toString().padEnd(8),
        data.summary.healthyDevices.toString().padEnd(8),
        data.summary.warningDevices.toString().padEnd(8),
        data.summary.criticalDevices.toString().padEnd(9),
        `${data.summary.avgCapacity}%`.padEnd(8),
        data.summary.totalAlerts.toString()
      );
    });
    
    // Find region with most issues
    const mostIssues = regionalData.reduce((prev, curr, index) => {
      const prevScore = prev.data.summary.criticalDevices * 3 + 
                       prev.data.summary.warningDevices +
                       prev.data.summary.criticalAlerts * 2;
      const currScore = curr.summary.criticalDevices * 3 + 
                       curr.summary.warningDevices +
                       curr.summary.criticalAlerts * 2;
      return currScore > prevScore ? { data: curr, region: regions[index] } : prev;
    }, { data: regionalData[0], region: regions[0] });
    
    console.log(`\n⚠️  ${mostIssues.region} requires the most attention`);
    console.log(`   Critical devices: ${mostIssues.data.summary.criticalDevices}`);
    console.log(`   Critical alerts: ${mostIssues.data.summary.criticalAlerts}`);
    
    return regionalData;
  } catch (error) {
    console.error('Error in example 6:', error);
    throw error;
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function calculateVariance(values: number[]): number {
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
  return squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
}

// ============================================================================
// RUN ALL EXAMPLES
// ============================================================================

export async function runAllExamples() {
  console.log('\n');
  console.log('═'.repeat(80));
  console.log('  BATTERY ANALYTICS DEMONSTRATION');
  console.log('  Advanced IoT Battery Monitoring with Kalman Filtering,');
  console.log('  Z-Score Analysis, and Rule-Based Alerting');
  console.log('═'.repeat(80));
  
  try {
    // Test database connection first
    console.log('\nTesting database connection...');
    const isConnected = await testConnection();
    
    if (!isConnected) {
      console.log('⚠️  Database connection failed. Using demonstration mode with simulated data.\n');
    } else {
      console.log('✅ Database connection successful\n');
    }
    
    // Run examples that don't require database
    example2_kalmanFilter();
    example3_zScoreAnalysis();
    example4_ruleEngine();
    
    // Run database-dependent examples if connected
    if (isConnected) {
      await example1_basicUsage();
      await example5_realtimeMonitoring();
      await example6_regionalComparison();
    }
    
    console.log('\n');
    console.log('═'.repeat(80));
    console.log('  ALL EXAMPLES COMPLETED');
    console.log('═'.repeat(80));
    console.log('\n');
    
  } catch (error) {
    console.error('\n❌ Error running examples:', error);
  } finally {
    // Clean up
    await closePool();
  }
}

// Run if called directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}
