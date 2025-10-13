// Test script for Kalman filter logging
import { applyKalmanFilter } from './lib/batteryAnalytics';

console.log('=== Testing Kalman Filter Logging ===\n');

// Test 1: Normal voltage data
console.log('\n--- Test 1: Voltage Filtering ---');
const voltageData = [3.7, 3.75, 3.68, 3.72, 3.69, 3.71, 3.73, 3.70, 3.72, 3.71];
const filteredVoltage = applyKalmanFilter(voltageData, 0.01, 0.1);

// Test 2: Capacity data with different parameters
console.log('\n--- Test 2: Capacity Filtering ---');
const capacityData = [85, 84, 86, 83, 85, 84, 85, 83, 84, 85, 84];
const filteredCapacity = applyKalmanFilter(capacityData, 0.005, 0.05);

// Test 3: Empty array
console.log('\n--- Test 3: Empty Array ---');
const emptyResult = applyKalmanFilter([], 0.01, 0.1);

// Test 4: Single value
console.log('\n--- Test 4: Single Value ---');
const singleResult = applyKalmanFilter([3.7], 0.01, 0.1);

console.log('\n=== All Tests Complete ===');
