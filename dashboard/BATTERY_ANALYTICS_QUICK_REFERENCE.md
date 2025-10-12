# Battery Analytics - Quick Reference

## 🚀 Quick Start (3 Steps)

### 1. Setup Database
```bash
createdb supply_chain_iot
psql supply_chain_iot < lib/database/schema.sql
```

### 2. Configure Environment
```bash
# .env
USE_SQL_BATTERY_ANALYTICS=true
DB_HOST=localhost
DB_NAME=supply_chain_iot
```

### 3. Install & Run
```bash
npm install pg @types/pg
npx tsx examples/battery-analytics-example.ts
```

## 📊 Key Functions

### Main Function
```typescript
getBatteryPerformance({
  region?: string,
  health?: string,
  deviceId?: string,
  applyKalman?: boolean,    // default: true
  applyZScore?: boolean,    // default: true
  applyRules?: boolean,     // default: true
  limit?: number            // default: 100
})
```

### Kalman Filter
```typescript
// Single update
let state = initKalmanFilter(3.7, 0.01, 0.1);
state = updateKalmanFilter(state, 3.75);

// Time series
const filtered = applyKalmanFilter(
  [3.72, 3.68, 3.75, 3.69],
  0.01,  // Q (process noise)
  0.1    // R (measurement noise)
);
```

### Z-Score Analysis
```typescript
const analysis = analyzeZScore(
  'device-id',
  'capacity',
  72,                        // current value
  [85, 84, 85, 86, 84],     // historical values
  2.0                        // threshold
);

console.log(analysis.isAnomaly);  // true/false
console.log(analysis.severity);   // 'normal' | 'warning' | 'critical'
```

### Rule Engine
```typescript
const alerts = evaluateAlertRules(batteryData, zscores);
alerts.forEach(alert => {
  console.log(`[${alert.severity}] ${alert.message}`);
  console.log(`Action: ${alert.action}`);
});
```

## 🎯 Alert Rules Reference

| Rule | Threshold | Severity | Action |
|------|-----------|----------|--------|
| VOLTAGE_CRITICAL_LOW | < 2.8V | Critical | Immediate replacement |
| VOLTAGE_CRITICAL_HIGH | > 4.2V | Critical | Disconnect immediately |
| CAPACITY_CRITICAL | < 50% | High | Schedule replacement |
| TEMPERATURE_CRITICAL | > 45°C | Critical | Shutdown device |
| TEMPERATURE_HIGH | > 35°C | High | Monitor closely |
| CAPACITY_LOW | 50-70% | Medium | Plan replacement |
| VOLTAGE_LOW | 2.8-3.0V | Medium | Monitor |
| HIGH_CYCLE_COUNT | > 3000 | Medium | Schedule inspection |
| VOLTAGE_ANOMALY | Z > 3σ | High | Investigate |
| CAPACITY_ANOMALY | Z > 3σ | High | Investigate |

## 📈 Configuration Cheat Sheet

### Kalman Filter Tuning
```
Smooth & Stable:  Q=0.001, R=0.5
Balanced:         Q=0.01,  R=0.1  ← Recommended
Responsive:       Q=0.1,   R=0.01
```

### Z-Score Thresholds
```
Sensitive:   1.0σ (68% confidence)
Balanced:    2.0σ (95% confidence) ← Recommended
Conservative: 3.0σ (99% confidence)
```

## 💾 SQL Quick Queries

### Latest telemetry
```sql
SELECT DISTINCT ON (device_id)
  device_id, voltage, capacity_percent, temperature_celsius
FROM battery_telemetry
ORDER BY device_id, timestamp DESC;
```

### Critical devices
```sql
SELECT device_id, capacity_percent, voltage
FROM battery_telemetry
WHERE timestamp > NOW() - INTERVAL '1 day'
  AND (capacity_percent < 50 OR voltage < 2.8)
ORDER BY capacity_percent ASC;
```

### Statistics
```sql
SELECT 
  device_id,
  AVG(voltage) as avg_voltage,
  STDDEV(voltage) as stddev_voltage
FROM battery_telemetry
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY device_id;
```

## 🔧 Common Tasks

### Monitor Critical Devices
```typescript
const result = await getBatteryPerformance({
  health: 'Critical',
  applyRules: true
});

result.devices
  .filter(d => d.alerts.some(a => a.severity === 'critical'))
  .forEach(d => console.log(`⚠️  ${d.device}: ${d.alerts.length} critical alerts`));
```

### Regional Comparison
```typescript
const regions = ['Asia-Pacific', 'Europe', 'North America'];
const results = await Promise.all(
  regions.map(r => getBatteryPerformance({ region: r }))
);

results.forEach((r, i) => {
  console.log(`${regions[i]}: ${r.summary.criticalDevices} critical`);
});
```

### Detect Anomalies
```typescript
const result = await getBatteryPerformance({ applyZScore: true });

result.devices
  .filter(d => Math.abs(d.capacityZScore) > 2)
  .forEach(d => console.log(`📉 ${d.device}: Capacity anomaly (Z=${d.capacityZScore})`));
```

## 🎓 Formula Reference

### Kalman Filter
```
Prediction:
  x̂ₖ⁻ = x̂ₖ₋₁
  Pₖ⁻ = Pₖ₋₁ + Q

Update:
  Kₖ = Pₖ⁻ / (Pₖ⁻ + R)
  x̂ₖ = x̂ₖ⁻ + Kₖ(zₖ - x̂ₖ⁻)
  Pₖ = (1 - Kₖ)Pₖ⁻
```

### Z-Score
```
Z = (x - μ) / σ

where:
  x = current value
  μ = mean
  σ = standard deviation
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot find module 'pg'" | `npm install pg @types/pg` |
| Too much smoothing | Increase Q or decrease R |
| Too many anomalies | Increase Z-score threshold |
| No data returned | Check DB connection, verify data exists |
| Missing alerts | Review rule thresholds in code |

## 📚 File Locations

```
dashboard/
├── lib/
│   ├── batteryAnalytics.ts          ← Main implementation
│   ├── riskRepo.ts                  ← Integration point
│   └── database/schema.sql          ← Database setup
├── examples/
│   └── battery-analytics-example.ts ← Working examples
├── .env.battery-analytics           ← Config template
└── BATTERY_ANALYTICS_README.md      ← Full documentation
```

## 🔗 Integration with MCP

Add to `mcp/server.ts`:
```typescript
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === 'getBatteryPerformance') {
    const { getBatteryPerformance } = await import('./lib/batteryAnalytics');
    const result = await getBatteryPerformance(request.params.arguments);
    return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
  }
});
```

## ⚡ Performance Tips

1. **Use indexes**: Already configured in schema.sql
2. **Limit results**: Use `limit` parameter
3. **Cache statistics**: Pre-calculate in `battery_statistics` table
4. **Batch processing**: Process multiple devices in parallel
5. **TimescaleDB**: For large-scale time-series (optional)

## 📞 Support

- Full docs: `BATTERY_ANALYTICS_README.md`
- Examples: `examples/battery-analytics-example.ts`
- Tests: `lib/__tests__/batteryAnalytics.test.ts`

---

**Quick Tip**: Run `npx tsx examples/battery-analytics-example.ts` to see everything in action!
