# Natural Language Querying Setup

This document explains the natural language querying implementation for the Supply Chain Risk Dashboard using LangChain, Next.js, MCP (Model Context Protocol), and OpenAI SDK.

## Architecture Overview

```
User Query → Next.js API Route → LangChain Agent → MCP Tools → Risk Repository → Response
```

## Components

### 1. Type Definitions (`types/risk.ts`)
- **RiskItem**: Core data structure for supply chain risks
- **RiskQueryParams**: Query filtering parameters
- **ExecutiveSummary**: Structured response format

### 2. Risk Repository (`lib/riskRepo.ts`)
Data access layer providing:
- `getTopRisks()`: Fetch and filter risks by region, severity, time window
- `getRiskById()`: Get specific risk details
- `getRiskTrends()`: Time-series analysis
- `getRiskSummary()`: Aggregate statistics

**Note**: Currently uses mock data. Replace with actual FastAPI calls:
```typescript
const response = await fetch(`${process.env.API_BASE_URL}/api/risks`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(params)
});
```

### 3. MCP Server (`mcp/server.ts`)
Model Context Protocol server exposing tools:
- **getTopRisks**: Filter and return top risks
- **getRiskById**: Fetch specific risk details
- **getRiskTrends**: Get time-series trends
- **getRiskSummary**: Summary statistics

The MCP server uses stdio transport for communication with the agent.

### 4. LangChain Agent (`lib/agent.ts`)
Main query processing logic:
- **Intent Analysis**: Classifies user intent (topRisks, trends, summary, etc.)
- **Parameter Extraction**: Extracts regions, time periods, severity levels
- **Tool Orchestration**: Calls appropriate data fetching functions
- **Structured Output**: Uses GPT-4o with structured output for consistent responses

Key function:
```typescript
askExecutive(question: string, context: ExecutiveContext): Promise<ExecutiveSummary>
```

### 5. API Route (`app/api/query/route.ts`)
Next.js API endpoints:
- **POST /api/query**: JSON body with question and context
- **GET /api/query?q=...**: Simple query parameter interface

## Installation

### 1. Install Dependencies

```bash
cd dashboard
npm install @langchain/openai @langchain/core @modelcontextprotocol/sdk zod
```

Or use the provided package list:
```bash
npm install $(cat package-dependencies.json | jq -r '.dependencies | keys[]')
```

### 2. Environment Variables

Create `.env.local`:
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
API_BASE_URL=http://localhost:8000  # Your FastAPI backend
NODE_ENV=development
```

### 3. TypeScript Configuration

Ensure `tsconfig.json` includes:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "paths": {
      "@/*": ["./*"]
    }
  }
}
```

## Usage

### Starting the Application

```bash
# Start Next.js development server
npm run dev

# In a separate terminal, start MCP server (optional, for standalone testing)
npm run mcp:server
```

### API Examples

#### Example 1: Top Risks Query
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top 3 risks in Asia-Pacific this week?",
    "context": {
      "region": "Asia-Pacific",
      "days": 7
    }
  }'
```

Response:
```json
{
  "success": true,
  "result": {
    "summary": "Analysis of Asia-Pacific region shows 2 high-priority risks requiring immediate attention...",
    "data": [
      {
        "id": "risk-001",
        "assetId": "GPS-TRACKER-B2",
        "score": 87,
        "severity": "high",
        "region": "Asia-Pacific",
        "lastUpdated": "2025-10-10T08:30:00Z"
      }
    ],
    "recommendations": [
      "URGENT: Immediate replacement recommended for GPS-TRACKER-B2",
      "Battery degradation detected - schedule preventive maintenance"
    ],
    "sources": ["Risk Repository", "IoT Device Monitoring"]
  },
  "timestamp": "2025-10-10T10:15:00Z"
}
```

#### Example 2: Battery Performance Query
```bash
curl "http://localhost:3000/api/query?q=Show%20me%20battery%20performance%20issues&days=7"
```

#### Example 3: Trend Analysis
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the risk trends over the past month?",
    "context": {
      "days": 30
    }
  }'
```

### Integration with Frontend

In your React component:
```typescript
const handleQuery = async (question: string) => {
  const response = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      context: {
        region: selectedRegion,
        days: 7
      }
    })
  });
  
  const { result } = await response.json();
  setMessages(prev => [...prev, {
    type: 'assistant',
    content: result.summary,
    data: result.data,
    recommendations: result.recommendations
  }]);
};
```

## Supported Query Types

### 1. Priority/Top Risks
- "What are the top 5 risks?"
- "Show me critical issues in Europe"
- "Highest priority alerts this week"

### 2. Trend Analysis
- "Show risk trends over the past month"
- "How have risks changed in Asia?"
- "What's the pattern of battery failures?"

### 3. Summary/Overview
- "Give me an overview of current risks"
- "How many high-severity alerts do we have?"
- "Summarize the supply chain status"

### 4. Specific Assets
- "What's the status of GPS-TRACKER-B2?"
- "Show me all pressure monitor risks"

## Customization

### Adding New Tools to MCP Server

Edit `mcp/server.ts`:
```typescript
const TOOLS: Tool[] = [
  // ... existing tools
  {
    name: "predictMaintenance",
    description: "Predict maintenance windows for assets",
    inputSchema: {
      type: "object",
      properties: {
        assetId: { type: "string" }
      }
    }
  }
];

// Add handler in CallToolRequestSchema
case "predictMaintenance": {
  const result = await predictMaintenance(args.assetId);
  return { content: [{ type: "text", text: JSON.stringify(result) }] };
}
```

### Extending Intent Analysis

Edit `lib/agent.ts`:
```typescript
async function analyzeIntent(question: string, context: ExecutiveContext) {
  // Add custom intent detection
  if (/predict|forecast|future/i.test(question)) {
    intent = "prediction";
  }
  
  // Add custom parameter extraction
  const assetMatch = question.match(/([A-Z]+-[A-Z]+-[A-Z0-9]+)/);
  if (assetMatch) {
    params.assetId = assetMatch[1];
  }
  
  return { intent, params };
}
```

## Production Deployment

### 1. Replace Mock Data
Update `lib/riskRepo.ts` to connect to your FastAPI backend:
```typescript
export async function getTopRisks(params: RiskQueryParams) {
  const response = await fetch(`${process.env.API_BASE_URL}/api/v1/risks/query`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.API_TOKEN}`
    },
    body: JSON.stringify(params),
    cache: 'no-store'
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  
  return response.json();
}
```

### 2. Add Authentication
Implement user authentication in `app/api/query/route.ts`:
```typescript
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";

export async function POST(request: NextRequest) {
  const session = await getServerSession(authOptions);
  
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  
  // Continue with query processing...
}
```

### 3. Rate Limiting
Add rate limiting to prevent abuse:
```typescript
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(10, "1 m"),
});

export async function POST(request: NextRequest) {
  const identifier = request.ip ?? "anonymous";
  const { success } = await ratelimit.limit(identifier);
  
  if (!success) {
    return NextResponse.json({ error: "Too many requests" }, { status: 429 });
  }
  
  // Continue...
}
```

### 4. Monitoring & Logging
Integrate with observability tools:
```typescript
import * as Sentry from "@sentry/nextjs";

try {
  const result = await askExecutive(question, execContext);
  
  // Log successful query
  console.log({
    type: "query_success",
    question,
    resultCount: result.data.length,
    timestamp: new Date().toISOString()
  });
  
  return NextResponse.json({ success: true, result });
} catch (error) {
  Sentry.captureException(error);
  // Error handling...
}
```

## Testing

### Unit Tests
```typescript
// __tests__/agent.test.ts
import { askExecutive } from "@/lib/agent";

describe("askExecutive", () => {
  it("should identify top risks query", async () => {
    const result = await askExecutive("What are the top 3 risks?", {});
    expect(result.data.length).toBeLessThanOrEqual(3);
    expect(result.summary).toBeTruthy();
  });
  
  it("should filter by region", async () => {
    const result = await askExecutive("Show risks in Europe", { region: "Europe" });
    expect(result.data.every(r => r.region === "Europe")).toBe(true);
  });
});
```

### Integration Tests
```bash
# Test API endpoint
npm run dev &
sleep 5
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Top risks?"}' | jq
```

## Troubleshooting

### Issue: OpenAI API Rate Limits
**Solution**: Implement caching and request throttling
```typescript
import { LRUCache } from "lru-cache";

const queryCache = new LRUCache({
  max: 100,
  ttl: 1000 * 60 * 5, // 5 minutes
});

const cacheKey = `${question}-${JSON.stringify(context)}`;
const cached = queryCache.get(cacheKey);
if (cached) return cached;
```

### Issue: Slow Response Times
**Solution**: Implement streaming responses
```typescript
export async function POST(request: NextRequest) {
  const encoder = new TextEncoder();
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();
  
  // Stream partial results as they become available
  const result = await askExecutive(question, context);
  await writer.write(encoder.encode(JSON.stringify(result)));
  await writer.close();
  
  return new Response(stream.readable, {
    headers: { "Content-Type": "text/event-stream" }
  });
}
```

### Issue: TypeScript Errors
**Solution**: Check module resolution and type definitions
```bash
npm install --save-dev @types/node
rm -rf node_modules/.cache
npm run type-check
```

## Next Steps

1. **Connect to Real Data**: Replace mock data in `riskRepo.ts`
2. **Add Authentication**: Implement user sessions and permissions
3. **Enhance UI**: Update dashboard to use the query API
4. **Add More Tools**: Extend MCP server with additional capabilities
5. **Deploy**: Configure production environment variables and deploy to Vercel/AWS

## Related Files

- `/types/risk.ts` - Type definitions
- `/lib/riskRepo.ts` - Data access layer
- `/lib/agent.ts` - LangChain agent
- `/mcp/server.ts` - MCP tool server
- `/app/api/query/route.ts` - API endpoint
- `/app/executive-dashboard.jsx` - Dashboard UI

## Support

For issues or questions, refer to:
- LangChain Docs: https://js.langchain.com/
- MCP Protocol: https://modelcontextprotocol.io/
- OpenAI API: https://platform.openai.com/docs
