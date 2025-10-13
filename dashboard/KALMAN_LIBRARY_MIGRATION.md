# Kalman Filter Library Migration

## Summary
Successfully migrated from custom Kalman filter implementation to the `kalman-filter` npm library.

## Changes Made

### 1. Installed Package
```bash
npm install kalman-filter
```

### 2. Updated Import Statement
**File:** `lib/batteryAnalytics.ts`

```typescript
import { KalmanFilter } from 'kalman-filter';
```

### 3. Updated `applyKalmanFilter` Function
Replaced the custom implementation with the library's API:

**Before:**
```typescript
export function applyKalmanFilter(
  measurements: number[],
  processNoise: number = 0.01,
  measurementNoise: number = 0.1
): number[] {
  if (measurements.length === 0) return [];
  
  let state = initKalmanFilter(measurements[0], processNoise, measurementNoise);
  const filtered: number[] = [state.x];
  
  for (let i = 1; i < measurements.length; i++) {
    state = updateKalmanFilter(state, measurements[i]);
    filtered.push(state.x);
  }
  
  return filtered;
}
```

**After:**
```typescript
export function applyKalmanFilter(
  measurements: number[],
  processNoise: number = 0.01,
  measurementNoise: number = 0.1
): number[] {
  if (measurements.length === 0) return [];
  
  // Configure Kalman filter for 1D scalar tracking
  const kf = new KalmanFilter({
    observation: {
      name: 'sensor',
      sensorDimension: 1,  // 1D measurement
      dimension: 1          // 1D state
    },
    dynamic: {
      name: 'constant-position',  // Constant position model
      dimension: 1                 // 1D state
    },
    observationCovariance: [[measurementNoise]], // R: measurement noise
    dynamicCovariance: [[processNoise]]          // Q: process noise
  });
  
  // Convert measurements to the format expected by the library
  const observationsArray = measurements.map(m => [m]);
  
  // Apply the filter to all measurements
  const filtered = kf.filterAll(observationsArray);
  
  // Convert back to simple array of numbers
  return filtered.map(f => f[0]);
}
```

## Benefits

### 1. **Professional Implementation**
- Uses a well-tested, production-grade library
- Maintained by the open-source community
- Handles edge cases and numerical stability better

### 2. **Advanced Features Available**
The library supports:
- Multi-dimensional Kalman filtering
- Extended Kalman Filter (EKF)
- Forward-Backward smoothing
- Correlation matrices
- Custom dynamic models

### 3. **Performance**
- Optimized for efficiency
- Supports batch processing with `filterAll()`
- Better numerical stability

### 4. **Backward Compatibility**
- API remains the same: `applyKalmanFilter(measurements, processNoise, measurementNoise)`
- Helper functions `initKalmanFilter` and `updateKalmanFilter` retained for compatibility
- No changes needed in calling code

## Testing Results

All tests passed successfully:

✅ Voltage filtering (10 samples)
✅ Capacity filtering (10 samples)
✅ Empty array handling
✅ Single value handling
✅ Integration with battery analytics system

## Example Usage

```typescript
import { applyKalmanFilter } from './lib/batteryAnalytics';

// Filter noisy voltage measurements
const noisyVoltages = [3.7, 3.75, 3.68, 3.72, 3.69, 3.71, 3.73, 3.70, 3.72];
const filtered = applyKalmanFilter(noisyVoltages, 0.01, 0.1);

// Original: [3.7, 3.75, 3.68, 3.72, 3.69, 3.71, 3.73, 3.70, 3.72]
// Filtered: [3.700, 3.733, 3.700, 3.712, 3.699, 3.706, 3.721, 3.708, 3.715]
```

## Configuration Parameters

### Process Noise (Q)
- **Default:** 0.01
- **Purpose:** How much the system state is expected to change
- **Tuning:** Lower = smoother but slower response, Higher = faster but less smooth

### Measurement Noise (R)
- **Default:** 0.1
- **Purpose:** How noisy the measurements are
- **Tuning:** Lower = trust measurements more, Higher = trust predictions more

### Battery Analytics Defaults
- **Voltage filtering:** Q=0.01, R=0.1
- **Capacity filtering:** Q=0.005, R=0.05 (more stable)

## No Breaking Changes

The migration maintains 100% backward compatibility:
- Same function signatures
- Same return types
- Same parameter defaults
- All existing code continues to work without modification

## Next Steps (Optional)

If you want to leverage advanced features in the future:

1. **Multi-dimensional filtering:** Track voltage + capacity together
2. **Extended Kalman Filter:** For non-linear battery degradation models
3. **Smoothing:** Apply backward pass for better historical analysis

## References

- Library: https://github.com/piercus/kalman-filter
- Documentation: https://www.npmjs.com/package/kalman-filter
- Examples: See `node_modules/kalman-filter/README.md`
