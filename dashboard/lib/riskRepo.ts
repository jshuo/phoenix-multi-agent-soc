// lib/riskRepo.ts
/**
 * Risk Repository - Data access layer for risk management
 * Connects to FastAPI backend or database for risk data
 */

import type { RiskItem, RiskQueryParams, RiskQueryResponse, RiskTrend } from "@/types/risk";

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
