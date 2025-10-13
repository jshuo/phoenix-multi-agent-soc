# Kalman Filter Logging - Quick Reference

## 📊 What Was Added

### 1. Core Filter Function Logging
**File:** `lib/batteryAnalytics.ts` - `applyKalmanFilter()`

```typescript
✅ Input validation logging
✅ Parameter display (Q, R values)
✅ Statistical analysis (mean, noise reduction)
✅ Output verification
✅ Edge case handling (empty arrays)
```

### 2. Battery Analytics Logging
**File:** `lib/batteryAnalytics.ts` - `getBatteryPerformance()`

```typescript
✅ Per-device processing logs
✅ Raw vs. Filtered value comparison
✅ Voltage filtering logs
✅ Capacity filtering logs
✅ Summary statistics banner
✅ Data sufficiency warnings
```

## 🎯 Log Tags

| Tag | Purpose |
|-----|---------|
| `[Kalman Filter]` | Low-level filter operations |
| `[Battery Analytics]` | High-level device processing |

## 📈 Key Metrics Logged

### Per-Filter Metrics
- 📥 Input measurements count
- ⚙️ Q and R parameters
- 📊 Original vs. Filtered mean
- 🔇 Noise reduction amount
- 📤 Output values count

### Per-Device Metrics
- 🔋 Raw voltage → Filtered voltage
- ⚡ Raw capacity → Filtered capacity
- 📏 Time series length

### Summary Metrics
- 🎯 Total devices processed
- 💚 Healthy / ⚠️ Warning / 🔴 Critical counts
- 🚨 Alert counts
- 🔍 Anomaly detection results

## 🧪 Testing

Run the test script:
```bash
npx tsx test-kalman-logging.ts
```

**Tests cover:**
- ✅ Voltage data (10 samples, Q=0.01, R=0.1)
- ✅ Capacity data (11 samples, Q=0.005, R=0.05)
- ✅ Empty array edge case
- ✅ Single value edge case

## 📖 Example Output

```
[Kalman Filter] Starting filtering process
[Kalman Filter] Input: 10 measurements
[Kalman Filter] Parameters: Q=0.01, R=0.1
[Kalman Filter] First measurement: 3.7, Last: 3.71
[Kalman Filter] Filter configuration created
[Kalman Filter] Filtering complete
[Kalman Filter] Original mean: 3.7110
[Kalman Filter] Filtered mean: 3.7106
[Kalman Filter] Noise reduction: 0.0004
[Kalman Filter] Output: 10 filtered values
[Kalman Filter] First filtered: 3.7000, Last: 3.7121
```

## 🎛️ Configuration

### Default Parameters

**Voltage Filtering:**
```typescript
Q = 0.01   // Process noise
R = 0.1    // Measurement noise
```

**Capacity Filtering:**
```typescript
Q = 0.005  // Process noise (more stable)
R = 0.05   // Measurement noise
```

## 📚 Documentation

- **Detailed Guide:** `KALMAN_FILTER_LOGGING.md`
- **Migration Info:** `KALMAN_LIBRARY_MIGRATION.md`
- **Test Script:** `test-kalman-logging.ts`

## 🔧 Tuning Guide

### When to Adjust Q (Process Noise)
- **Increase Q** if filtered values lag behind measurements
- **Decrease Q** if filtered values are too noisy

### When to Adjust R (Measurement Noise)  
- **Increase R** if you don't trust the measurements
- **Decrease R** if measurements are accurate

### Monitoring
Watch these logged values:
- **Noise reduction:** Should be small but consistent
- **Mean difference:** Should be minimal
- **First/Last values:** Should make physical sense

## 🚀 Benefits

1. **🐛 Debugging** - See exactly what's happening
2. **📊 Monitoring** - Track filter performance
3. **⚡ Optimization** - Tune parameters effectively
4. **✅ Validation** - Verify correct operation
5. **🔍 Transparency** - Understand the filtering process

## ⚠️ Production Notes

For production deployments, consider:
- Using a proper logging library (Winston, Pino)
- Adding log levels (DEBUG, INFO, WARN, ERROR)
- Implementing log rotation
- Adding performance timing
- Using structured logging (JSON format)

## 📞 Support

If you see unexpected values in the logs:
1. Check the raw input data quality
2. Verify Q and R parameter settings
3. Review the time series length (need 2+ samples)
4. Check for sensor calibration issues
5. Review the KALMAN_FILTER_LOGGING.md for detailed diagnostics
