// mcp/server.ts
/**
 * MCP (Model Context Protocol) Server
 * Exposes tools for the LangChain agent to query risk data
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { getTopRisks, getRiskById, getRiskTrends, getRiskSummary, getSupplierRisks, getBatteryPerformance, getBatteryReliability, getAlertTrends } from "../lib/riskRepo.js";
import { getWeatherForRegion, getRegionalWeather } from "../lib/weather.js";

/**
 * Define available tools
 */
const TOOLS: Tool[] = [
  {
    name: "getTopRisks",
    description: "Return top risks filtered by region, days, severity, or minimum score. Returns detailed risk items with explainability features.",
    inputSchema: {
      type: "object",
      properties: {
        region: {
          type: "string",
          description: "Filter by region (e.g., 'Asia-Pacific', 'Europe', 'North America')",
        },
        days: {
          type: "number",
          description: "Number of days to look back (default: 7)",
          default: 7,
        },
        severity: {
          type: "string",
          enum: ["low", "medium", "high"],
          description: "Filter by severity level",
        },
        minScore: {
          type: "number",
          description: "Minimum risk score (0-100, default: 0)",
          default: 0,
        },
        limit: {
          type: "number",
          description: "Maximum number of results (default: 10)",
          default: 10,
        },
      },
    },
  },
  {
    name: "getRiskById",
    description: "Get detailed information about a specific risk by its ID",
    inputSchema: {
      type: "object",
      properties: {
        id: {
          type: "string",
          description: "The risk ID to retrieve",
        },
      },
      required: ["id"],
    },
  },
  {
    name: "getRiskTrends",
    description: "Get time-series trend data for risks over a specified period",
    inputSchema: {
      type: "object",
      properties: {
        region: {
          type: "string",
          description: "Filter trends by region",
        },
        days: {
          type: "number",
          description: "Number of days for trend analysis (default: 7)",
          default: 7,
        },
      },
    },
  },
  {
    name: "getRiskSummary",
    description: "Get summary statistics about risks including counts by severity and critical assets",
    inputSchema: {
      type: "object",
      properties: {
        region: {
          type: "string",
          description: "Filter summary by region",
        },
      },
    },
  },
  {
    name: "getSupplierRisks",
    description: "Get supplier risk data including port delays, quality issues, and labor shortages",
    inputSchema: {
      type: "object",
      properties: {
        region: {
          type: "string",
          description: "Filter by region (e.g., 'Asia-Pacific', 'Europe', 'North America')",
        },
        limit: {
          type: "number",
          description: "Maximum number of suppliers to return (default: 10)",
          default: 10,
        },
      },
    },
  },
  {
    name: "getBatteryPerformance",
    description: "Get IoT device battery performance metrics including voltage, capacity, temperature, and health status. Automatically includes weather data (temperature, humidity, conditions) for each region to help correlate environmental factors with battery performance.",
    inputSchema: {
      type: "object",
      properties: {
        region: {
          type: "string",
          description: "Filter by region",
        },
        health: {
          type: "string",
          enum: ["Excellent", "Good", "Warning", "Critical"],
          description: "Filter by battery health status",
        },
      },
    },
  },
  {
    name: "getBatteryReliability",
    description: "Get battery reliability summary with health percentages and recommendations",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "getAlertTrends",
    description: "Get alert trends and logistics insights for specific regions",
    inputSchema: {
      type: "object",
      properties: {
        region: {
          type: "string",
          description: "Region to analyze (e.g., 'Asia-Pacific', 'Europe')",
        },
      },
    },
  },
  {
    name: "getWeatherForRegion",
    description: "Get current weather data for a specific region. Returns temperature, humidity, precipitation, wind speed, and weather conditions.",
    inputSchema: {
      type: "object",
      properties: {
        region: {
          type: "string",
          description: "Region name (e.g., 'Asia-Pacific', 'Europe', 'North America', 'South America', 'Africa', 'Middle East')",
        },
      },
      required: ["region"],
    },
  },
  {
    name: "getRegionalWeather",
    description: "Get current weather data for multiple regions at once",
    inputSchema: {
      type: "object",
      properties: {
        regions: {
          type: "array",
          items: {
            type: "string",
          },
          description: "Array of region names to fetch weather for",
        },
      },
      required: ["regions"],
    },
  },
];

/**
 * Create and configure MCP server
 */
export function createMCPServer() {
  const server = new Server(
    {
      name: "risk-analysis-server",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Handler for listing available tools
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: TOOLS,
    };
  });

  // Handler for executing tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      switch (name) {
        case "getTopRisks": {
          const result = await getTopRisks(args || {});
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getRiskById": {
          if (!args?.id) {
            throw new Error("Risk ID is required");
          }
          const result = await getRiskById(args.id);
          return {
            content: [
              {
                type: "text",
                text: result ? JSON.stringify(result, null, 2) : "Risk not found",
              },
            ],
          };
        }

        case "getRiskTrends": {
          const result = await getRiskTrends(args || {});
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getRiskSummary": {
          const result = await getRiskSummary(args || {});
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getSupplierRisks": {
          const result = await getSupplierRisks(args || {});
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getBatteryPerformance": {
          const result = await getBatteryPerformance(args || {});
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getBatteryReliability": {
          const result = await getBatteryReliability();
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getAlertTrends": {
          const result = await getAlertTrends(args || {});
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getWeatherForRegion": {
          if (!args?.region) {
            throw new Error("Region is required");
          }
          const result = await getWeatherForRegion(args.region);
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "getRegionalWeather": {
          if (!args?.regions || !Array.isArray(args.regions)) {
            throw new Error("Regions array is required");
          }
          const result = await getRegionalWeather(args.regions);
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ error: errorMessage }),
          },
        ],
        isError: true,
      };
    }
  });

  return server;
}

/**
 * Start the MCP server
 */
async function main() {
  const server = createMCPServer();
  const transport = new StdioServerTransport();
  
  await server.connect(transport);
  
  console.error("Risk Analysis MCP Server running on stdio");
}

// Run server if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error("Server error:", error);
    process.exit(1);
  });
}
