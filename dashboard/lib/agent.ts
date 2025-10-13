// lib/agent.ts
/**
 * LangChain Agent with MCP Tools Integration
 * Handles natural language queries about supply chain risks
 */

import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import type { ExecutiveSummary, RiskItem } from "@/types/risk";
import { getTopRisks, getRiskTrends, getRiskSummary, getSupplierRisks, getBatteryPerformance, getBatteryReliability, getAlertTrends } from "./riskRepo";

// Structured output schema for executive responses
const ExecutiveAnswerSchema = z.object({
  summary: z.string().describe("Human-readable executive summary of the findings"),
  data: z.array(
    z.object({
      id: z.string(),
      assetId: z.string(),
      score: z.number(),
      severity: z.string(),
      region: z.string(),
      lastUpdated: z.string(),
    })
  ).describe("Array of risk items ranked by priority"),
  trends: z.array(
    z.object({
      date: z.string(),
      avgScore: z.number(),
      count: z.number(),
      highSeverityCount: z.number(),
    })
  ).optional().describe("Optional trend data over time"),
  sources: z.array(z.string()).optional().describe("Data sources used for the analysis"),
  recommendations: z.array(z.string()).optional().describe("Actionable recommendations"),
});

export type ExecutiveContext = {
  region?: string;
  days?: number;
  userRole?: string;
};

/**
 * Main entry point for natural language risk queries
 */
export async function askExecutive(
  question: string,
  execContext: ExecutiveContext = {}
): Promise<ExecutiveSummary> {
  const llm = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY,
  });

  // Analyze intent and fetch relevant data
  const { intent, params } = await analyzeIntent(question, execContext);

  // Log extracted parameters for debugging
  console.log('[Agent] Intent analysis:', { intent, params });

  let toolResults: any = {};

  // Call appropriate tools based on intent
  if (intent === "topRisks" || intent === "priorityRisks") {
    const riskResponse = await getTopRisks({
      region: params.region || execContext.region,
      days: params.days || execContext.days || 7,
      severity: params.severity,
      minScore: params.minScore || 0,
      limit: params.limit || 10,
    });
    toolResults.risks = riskResponse.risks;
    toolResults.queryParams = riskResponse.queryParams;
  }

  if (intent === "trends" || intent === "analysis" || intent === "alertTrends") {
    const trends = await getRiskTrends({
      region: params.region || execContext.region,
      days: params.days || execContext.days || 7,
    });
    toolResults.trends = trends;
    
    // Also get alert trends
    const alertTrends = await getAlertTrends({
      region: params.region || execContext.region,
    });
    toolResults.alertTrends = alertTrends;
  }

  if (intent === "summary" || intent === "overview") {
    const summary = await getRiskSummary({
      region: params.region || execContext.region,
    });
    toolResults.summary = summary;
  }

  // Battery-specific intents
  if (intent === "batteryPerformance") {
    console.log('[Agent] Fetching battery performance with Kalman filter enabled');
    const batteryData = await getBatteryPerformance({
      region: params.region || execContext.region,
      health: params.health,
      useAdvancedAnalytics: true,  // Enable Kalman filter and advanced analytics
    });
    toolResults.batteryData = batteryData;
  }

  if (intent === "batteryReliability") {
    const reliability = await getBatteryReliability();
    toolResults.batteryReliability = reliability;
  }

  // Supplier risks
  if (intent === "supplierRisks") {
    const suppliers = await getSupplierRisks({
      region: params.region || execContext.region,
      limit: params.limit || 10,
    });
    toolResults.suppliers = suppliers;
  }

  // If no specific intent, get comprehensive data
  if (!intent || intent === "general") {
    console.log('[Agent] Fetching comprehensive data with Kalman filter enabled');
    const [riskResponse, trends, summary, batteryData, suppliers] = await Promise.all([
      getTopRisks({
        region: execContext.region,
        days: execContext.days || 7,
        limit: 5,
      }),
      getRiskTrends({
        region: execContext.region,
        days: 7,
      }),
      getRiskSummary({
        region: execContext.region,
      }),
      getBatteryPerformance({
        region: execContext.region,
        useAdvancedAnalytics: true,  // Enable Kalman filter and advanced analytics
      }),
      getSupplierRisks({
        region: execContext.region,
        limit: 5,
      }),
    ]);
    toolResults = { risks: riskResponse.risks, trends, summary, batteryData, suppliers };
  }

  // Generate structured response using LLM
  const systemPrompt = `You are an Executive Risk Assistant for supply chain monitoring.

Your role:
- Analyze IoT device battery performance and reliability
- Identify critical risks requiring immediate attention
- Provide actionable recommendations based on data
- Explain risk scores and contributing factors clearly
- Prioritize by severity and business impact

Context:
${execContext.region ? `- Focus region: ${execContext.region}` : "- Global analysis"}
${execContext.days ? `- Time window: ${execContext.days} days` : ""}
${execContext.userRole ? `- User role: ${execContext.userRole}` : ""}

Guidelines:
- Be concise but comprehensive
- Highlight critical issues (score >= 80) first
- Explain technical metrics in business terms
- Always include specific asset IDs and regions
- use Markdown formatting for clarity
- Provide timeline estimates for predicted failures`;

  const userPrompt = `Question: ${question}

Available Data:
${JSON.stringify(toolResults, null, 2)}

Provide a structured executive summary addressing this question.`;

  try {
    const structuredLlm = llm.withStructuredOutput(ExecutiveAnswerSchema, {
      name: "executive_answer",
    });

    const response = await structuredLlm.invoke([
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ]);

    return {
      summary: response.summary,
      data: response.data as RiskItem[],
      trends: response.trends,
      sources: response.sources || ["Risk Repository", "IoT Device Monitoring"],
      recommendations: response.recommendations,
    };
  } catch (error) {
    console.error("Error generating executive response:", error);
    
    // Fallback response
    return {
      summary: `Analysis for: ${question}. ${
        toolResults.risks?.length || 0
      } risks identified. Please review the data below.`,
      data: toolResults.risks || [],
      trends: toolResults.trends,
      sources: ["Risk Repository"],
    };
  }
}

/**
 * Analyze user intent and extract parameters
 */
async function analyzeIntent(
  question: string,
  context: ExecutiveContext
): Promise<{
  intent: string;
  params: any;
}> {
  const lowerQuestion = question.toLowerCase();

  // Simple intent classification
  let intent = "general";
  const params: any = {};

  // Top risks / priority
  if (/top|highest|priority|critical|urgent/i.test(question)) {
    intent = "topRisks";
    
    // Extract number if specified
    const numberMatch = question.match(/top\s+(\d+)/i);
    if (numberMatch) {
      params.limit = parseInt(numberMatch[1]);
    }
    
    // Extract "this week" or "week" for top risks
    if (/this\s+week|week/i.test(question) && !params.days) {
      params.days = 7;
    }
  }

  // Trends / analysis
  if (/trend|over time|history|pattern|analysis|alert/i.test(question)) {
    intent = "trends";
    if (/alert/i.test(question)) {
      intent = "alertTrends";
    }
  }

  // Summary / overview
  if (/summary|overview|status|how many/i.test(question)) {
    intent = "summary";
  }

  // Battery performance
  if (/battery.*performance|battery.*status|battery.*metric|battery.*analysis|iot.*battery/i.test(question)) {
    intent = "batteryPerformance";
  }

  // Battery reliability / health
  if (/battery.*(reliability|health|condition)/i.test(question)) {
    intent = "batteryReliability";
  }

  // Supplier risks
  if (/supplier.*risk|vendor.*risk|supplier.*issue/i.test(question)) {
    intent = "supplierRisks";
  }

  // Extract region if mentioned (check all variations)
  if (/asia[-\s]?pacific|apac|asia/i.test(question)) {
    params.region = "Asia-Pacific";
  } else if (/europe|eu|emea/i.test(question)) {
    params.region = "Europe";
  } else if (/north\s+america|america|americas|us/i.test(question)) {
    params.region = "North America";
  }
  
  // Log region extraction for debugging
  if (params.region) {
    console.log('[Agent] Extracted region:', params.region, 'from question:', question);
  }

  // Extract severity
  if (/high|critical|severe/i.test(question)) {
    params.severity = "high";
  } else if (/medium|moderate/i.test(question)) {
    params.severity = "medium";
  } else if (/low|minor/i.test(question)) {
    params.severity = "low";
  }

  // Extract time period (check multiple patterns)
  const daysMatch = question.match(/(\d+)\s*days?/i);
  if (daysMatch) {
    params.days = parseInt(daysMatch[1]);
  } else if (/this\s+week|week/i.test(question) && !params.days) {
    params.days = 7;
  } else if (/this\s+month|month/i.test(question)) {
    params.days = 30;
  } else if (/today/i.test(question)) {
    params.days = 1;
  }
  
  // Log time period extraction for debugging
  if (params.days) {
    console.log('[Agent] Extracted days:', params.days, 'from question:', question);
  }

  return { intent, params };
}

/**
 * Generate recommendations based on risk data
 */
export function generateRecommendations(risks: RiskItem[]): string[] {
  const recommendations: string[] = [];

  const criticalRisks = risks.filter(r => r.score >= 80);
  const highTempRisks = risks.filter(r => 
    r.reasons.some(reason => 
      reason.feature.toLowerCase().includes('temperature') && reason.contribution > 20
    )
  );
  const batteryRisks = risks.filter(r =>
    r.reasons.some(reason =>
      reason.feature.toLowerCase().includes('battery') && reason.contribution > 25
    )
  );

  if (criticalRisks.length > 0) {
    recommendations.push(
      `URGENT: ${criticalRisks.length} critical risk(s) detected. Immediate replacement recommended for: ${criticalRisks.map(r => r.assetId).join(", ")}`
    );
  }

  if (highTempRisks.length > 0) {
    recommendations.push(
      `Temperature-related issues detected in ${highTempRisks.length} device(s). Consider environmental controls or device relocation.`
    );
  }

  if (batteryRisks.length > 0) {
    recommendations.push(
      `Battery degradation is a primary concern. Schedule preventive maintenance for: ${batteryRisks.map(r => r.assetId).join(", ")}`
    );
  }

  if (recommendations.length === 0) {
    recommendations.push("All monitored assets are within acceptable parameters. Continue regular monitoring.");
  }

  return recommendations;
}
