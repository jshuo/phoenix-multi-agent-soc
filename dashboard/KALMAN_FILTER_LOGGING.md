# Kalman Filter Logging Documentation

## Overview
Comprehensive console logging has been added to the Kalman filter implementation to help with debugging, monitoring, and understanding the filtering process in the battery analytics module.

## Logging Levels

### 1. Kalman Filter Function (`applyKalmanFilter`)

The low-level Kalman filter function logs detailed information about each filtering operation:

#### Input Phase
```
[Kalman Filter] Starting filtering process
[Kalman Filter] Input: {N} measurements
[Kalman Filter] Parameters: Q={processNoise}, R={measurementNoise}
[Kalman Filter] First measurement: {value}, Last: {value}
```

#### Processing Phase
```
[Kalman Filter] Filter configuration created
[Kalman Filter] Filtering complete
```

#### Output Phase with Statistics
```
[Kalman Filter] Original mean: {mean}
[Kalman Filter] Filtered mean: {mean}
[Kalman Filter] Noise reduction: {difference}
[Kalman Filter] Output: {N} filtered values
[Kalman Filter] First filtered: {value}, Last: {value}
```

#### Edge Cases
```
[Kalman Filter] Empty measurements array, returning empty result
```

### 2. Battery Analytics Function (`getBatteryPerformance`)

The high-level battery analytics function logs device-specific filtering operations:

#### Per-Device Processing
```
[Battery Analytics] Processing device: {deviceId}
[Battery Analytics] Time series length: {N} samples

[Battery Analytics] Applying Kalman filter to VOLTAGE data
[Kalman Filter] ... (detailed filter logs)
[Battery Analytics] Voltage - Raw: {value}V → Filtered: {value}V

[Battery Analytics] Applying Kalman filter to CAPACITY data
[Kalman Filter] ... (detailed filter logs)
[Battery Analytics] Capacity - Raw: {value}% → Filtered: {value}%
```

#### Insufficient Data Warning
```
[Battery Analytics] Skipping Kalman filter for device {deviceId} - insufficient data (only {N} sample)
```

#### Summary Statistics
```
[Battery Analytics] ========================================
[Battery Analytics] PROCESSING COMPLETE
[Battery Analytics] ========================================
[Battery Analytics] Total devices processed: {N}
[Battery Analytics] Healthy devices: {N}
[Battery Analytics] Warning devices: {N}
[Battery Analytics] Critical devices: {N}
[Battery Analytics] Average capacity: {N}%
[Battery Analytics] Total alerts: {N}
[Battery Analytics] Critical alerts: {N}
[Battery Analytics] Anomalies detected: {N}
[Battery Analytics] Kalman filter applied: YES/NO
[Battery Analytics] Z-score analysis applied: YES/NO
[Battery Analytics] Rules evaluated: {N}
[Battery Analytics] ========================================
```

## Example Log Output

### Complete Processing Example
```
[Battery Analytics] Processing device: GPS-TRACKER-B2
[Battery Analytics] Time series length: 15 samples
[Battery Analytics] Applying Kalman filter to VOLTAGE data
[Kalman Filter] Starting filtering process
[Kalman Filter] Input: 15 measurements
[Kalman Filter] Parameters: Q=0.01, R=0.1
[Kalman Filter] First measurement: 3.68, Last: 3.71
[Kalman Filter] Filter configuration created
[Kalman Filter] Filtering complete
[Kalman Filter] Original mean: 3.7053
[Kalman Filter] Filtered mean: 3.7049
[Kalman Filter] Noise reduction: 0.0004
[Kalman Filter] Output: 15 filtered values
[Kalman Filter] First filtered: 3.6800, Last: 3.7098
[Battery Analytics] Voltage - Raw: 3.710V → Filtered: 3.710V

[Battery Analytics] Applying Kalman filter to CAPACITY data
[Kalman Filter] Starting filtering process
[Kalman Filter] Input: 15 measurements
[Kalman Filter] Parameters: Q=0.005, R=0.05
[Kalman Filter] First measurement: 83, Last: 84
[Kalman Filter] Filter configuration created
[Kalman Filter] Filtering complete
[Kalman Filter] Original mean: 83.8667
[Kalman Filter] Filtered mean: 83.9124
[Kalman Filter] Noise reduction: 0.0457
[Kalman Filter] Output: 15 filtered values
[Kalman Filter] First filtered: 82.9999, Last: 84.0789
[Battery Analytics] Capacity - Raw: 84.0% → Filtered: 84.1%

[Battery Analytics] ========================================
[Battery Analytics] PROCESSING COMPLETE
[Battery Analytics] ========================================
[Battery Analytics] Total devices processed: 25
[Battery Analytics] Healthy devices: 18
[Battery Analytics] Warning devices: 5
[Battery Analytics] Critical devices: 2
[Battery Analytics] Average capacity: 78%
[Battery Analytics] Total alerts: 12
[Battery Analytics] Critical alerts: 3
[Battery Analytics] Anomalies detected: 7
[Battery Analytics] Kalman filter applied: YES
[Battery Analytics] Z-score analysis applied: YES
[Battery Analytics] Rules evaluated: 10
[Battery Analytics] ========================================
```

## Monitoring and Debugging

### What to Look For

#### 1. **Noise Reduction Effectiveness**
- Check the "Noise reduction" value in the logs
- Higher values indicate more noise was filtered out
- Typical values: 0.001 - 0.1 for voltage, 0.01 - 1.0 for capacity

#### 2. **Parameter Tuning**
- **Q (Process Noise):** How much the system changes
  - Lower Q = smoother but slower response
  - Higher Q = faster response but less smoothing
- **R (Measurement Noise):** How noisy the measurements are
  - Lower R = trust measurements more
  - Higher R = trust predictions more

#### 3. **Data Quality Issues**
- Watch for "insufficient data" warnings
- Check if first and last measurements are reasonable
- Verify mean values are in expected ranges

#### 4. **Performance Metrics**
- Monitor total devices processed
- Track healthy vs. warning vs. critical device counts
- Review anomaly detection rates

## Configuration

### Voltage Filtering Parameters
```typescript
processNoise: 0.01      // Q parameter
measurementNoise: 0.1   // R parameter
```

### Capacity Filtering Parameters
```typescript
processNoise: 0.005     // Q parameter (lower = more stable)
measurementNoise: 0.05  // R parameter
```

## Disabling Logs

To disable logging in production, you can:

### Option 1: Environment Variable
Set `NODE_ENV=production` and modify logs:
```typescript
if (process.env.NODE_ENV !== 'production') {
  console.log('[Kalman Filter] ...');
}
```

### Option 2: Logging Level
Add a logging configuration:
```typescript
const DEBUG_KALMAN = process.env.DEBUG_KALMAN === 'true';

if (DEBUG_KALMAN) {
  console.log('[Kalman Filter] ...');
}
```

### Option 3: Replace with Logger
Use a proper logging library:
```typescript
import logger from './logger';
logger.debug('[Kalman Filter] ...');
```

## Testing Logs

To test the logging functionality:
```bash
npx tsx test-kalman-logging.ts
```

This runs comprehensive tests showing:
- ✅ Voltage filtering with 10 samples
- ✅ Capacity filtering with 11 samples  
- ✅ Empty array handling
- ✅ Single value handling

## Benefits

1. **Debugging:** Quickly identify issues in the filtering pipeline
2. **Monitoring:** Track filter performance in real-time
3. **Optimization:** Tune Q and R parameters based on logged statistics
4. **Validation:** Verify filtering is working correctly
5. **Transparency:** Understand what the Kalman filter is doing

## Next Steps

Consider adding:
- Log aggregation (e.g., Winston, Bunyan)
- Performance metrics (filtering time)
- Error rate tracking
- Filter convergence monitoring
- Automated parameter tuning based on logged statistics
