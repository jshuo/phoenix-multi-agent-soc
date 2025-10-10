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
import { getTopRisks, getRiskById, getRiskTrends, getRiskSummary } from "../lib/riskRepo.js";

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
