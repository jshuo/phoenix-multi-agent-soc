// types/risk.ts
/**
 * Risk Item structure for supply chain monitoring
 * Represents anomaly detection results with explainability
 */
export type RiskItem = {
  id: string;
  assetId: string;
  severity: "low" | "medium" | "high";
  score: number; // 0..100
  reasons: Array<{
    feature: string;
    weight: number;
    contribution: number;
  }>;
  timeWindow: string; // "2025-10-01..2025-10-07"
  region: string;
  city?: string; // Display city name instead of region
  lastUpdated: string;
};

/**
 * Query parameters for filtering risks
 */
export type RiskQueryParams = {
  region?: string;
  days?: number;
  severity?: "low" | "medium" | "high";
  minScore?: number;
  limit?: number;
};

/**
 * Response structure for risk queries
 */
export type RiskQueryResponse = {
  risks: RiskItem[];
  totalCount: number;
  queryParams: RiskQueryParams;
  timestamp: string;
};

/**
 * Trend data for time-series analysis
 */
export type RiskTrend = {
  date: string;
  avgScore: number;
  count: number;
  highSeverityCount: number;
  region?: string;
};

/**
 * Executive summary structure
 */
export type ExecutiveSummary = {
  summary: string;
  data: RiskItem[];
  batteryData?: any[];
  weatherData?: Record<string, any>;
  trends?: RiskTrend[];
  sources?: string[];
  recommendations?: string[];
  detailedSummary?: string;
};
