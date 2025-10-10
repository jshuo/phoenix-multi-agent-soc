# Natural Language Query Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Executive Dashboard (executive-dashboard.jsx)               │  │
│  │  - AI Assistant Chat                                         │  │
│  │  - Quick Questions                                           │  │
│  │  - Battery Health Cards                                      │  │
│  │  - Risk Visualization                                        │  │
│  └────────────┬─────────────────────────────────────────────────┘  │
└───────────────┼─────────────────────────────────────────────────────┘
                │
                │ useNaturalLanguageQuery()
                │
┌───────────────▼─────────────────────────────────────────────────────┐
│                      NEXT.JS API LAYER                              │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  /api/query/route.ts                                         │  │
│  │  - POST /api/query (JSON body)                               │  │
│  │  - GET /api/query?q=...                                      │  │
│  │  - Request validation                                        │  │
│  │  - Error handling                                            │  │
│  └────────────┬─────────────────────────────────────────────────┘  │
└───────────────┼─────────────────────────────────────────────────────┘
                │
                │ askExecutive(question, context)
                │
┌───────────────▼─────────────────────────────────────────────────────┐
│                     LANGCHAIN AGENT LAYER                           │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  lib/agent.ts                                                │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  1. Intent Analysis                                    │ │  │
│  │  │     - topRisks / trends / summary / overview           │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  2. Parameter Extraction                               │ │  │
│  │  │     - region, days, severity, minScore                 │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  3. Tool Selection & Orchestration                     │ │  │
│  │  │     - getTopRisks()                                    │ │  │
│  │  │     - getRiskTrends()                                  │ │  │
│  │  │     - getRiskSummary()                                 │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  4. LLM Response Generation (GPT-4o)                   │ │  │
│  │  │     - Structured output with Zod schema                │ │  │
│  │  │     - Executive summary                                │ │  │
│  │  │     - Recommendations                                  │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └────────────┬─────────────────────────────────────────────────┘  │
└───────────────┼─────────────────────────────────────────────────────┘
                │
                │ Function Calls
                │
┌───────────────▼─────────────────────────────────────────────────────┐
│                     RISK REPOSITORY LAYER                           │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  lib/riskRepo.ts                                             │  │
│  │                                                              │  │
│  │  getTopRisks(params)      ──────────┐                       │  │
│  │  getRiskById(id)          ──────────┤                       │  │
│  │  getRiskTrends(params)    ──────────┼─> Data Fetching       │  │
│  │  getRiskSummary(params)   ──────────┘                       │  │
│  └────────────┬─────────────────────────────────────────────────┘  │
└───────────────┼─────────────────────────────────────────────────────┘
                │
                ├─── Currently: Mock Data
                │    (4 sample IoT devices with battery metrics)
                │
                └─── Future: FastAPI Backend
                     (Real-time IoT sensor data)
                     
┌────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐               │
│  │   Mock Risk Data     │  │  Future: FastAPI     │               │
│  │   ─────────────      │  │  Backend             │               │
│  │   • GPS-TRACKER-B2   │  │  ──────────────      │               │
│  │   • PRESSURE-MON-D4  │  │  • Kalman Filtering  │               │
│  │   • TEMP-SENSOR-A1   │  │  • EWMA Smoothing    │               │
│  │   • HUMIDITY-SENS-C3 │  │  • Z-score Analysis  │               │
│  └──────────────────────┘  │  • Rule Engine       │               │
│                             │  • Real-time Sensors │               │
│                             └──────────────────────┘               │
└────────────────────────────────────────────────────────────────────┘
```

## Data Flow Example

### Example Query: "What are the top 3 risks in Asia-Pacific?"

```
1. USER INPUT
   ├─ Question: "What are the top 3 risks in Asia-Pacific?"
   └─ Context: { region: "Asia-Pacific", days: 7 }

2. API ROUTE (/api/query)
   ├─ Validate input
   ├─ Extract question & context
   └─ Call askExecutive()

3. LANGCHAIN AGENT
   ├─ Intent Analysis
   │  └─ Detected: "topRisks"
   │
   ├─ Parameter Extraction
   │  ├─ region: "Asia-Pacific"
   │  ├─ days: 7
   │  └─ limit: 3
   │
   ├─ Tool Selection
   │  └─ Call: getTopRisks({ region: "Asia-Pacific", days: 7, limit: 3 })
   │
   └─ LLM Processing (GPT-4o)
      ├─ System Prompt: "You are an Executive Risk Assistant..."
      ├─ User Prompt: "Question + Available Data"
      └─ Structured Output: {summary, data, recommendations}

4. RISK REPOSITORY
   ├─ Filter risks by region="Asia-Pacific"
   ├─ Sort by score (descending)
   ├─ Take top 3
   └─ Return: RiskQueryResponse

5. RESPONSE TO USER
   {
     "success": true,
     "result": {
       "summary": "2 high-priority risks detected in Asia-Pacific...",
       "data": [
         {
           "id": "risk-001",
           "assetId": "GPS-TRACKER-B2",
           "score": 87,
           "severity": "high",
           "region": "Asia-Pacific"
         },
         {
           "id": "risk-004",
           "assetId": "HUMIDITY-SENSOR-C3",
           "score": 22,
           "severity": "low",
           "region": "Asia-Pacific"
         }
       ],
       "recommendations": [
         "URGENT: Immediate replacement for GPS-TRACKER-B2",
         "Battery degradation detected - schedule maintenance"
       ]
     }
   }
```

## MCP Server Architecture (Optional)

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP SERVER (Optional)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  mcp/server.ts                                      │   │
│  │                                                     │   │
│  │  Tool Registry:                                     │   │
│  │  ├─ getTopRisks                                     │   │
│  │  ├─ getRiskById                                     │   │
│  │  ├─ getRiskTrends                                   │   │
│  │  └─ getRiskSummary                                  │   │
│  │                                                     │   │
│  │  Transport: StdioServerTransport                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
        │                                    ▲
        │ Tool Calls                         │ Results
        ▼                                    │
┌─────────────────────────────────────────────────────────────┐
│              LangChain Agent (if using MCP)                  │
│                                                               │
│  convertMCPToolsToLangChainTools()                          │
│  ├─ Connect to MCP server                                   │
│  ├─ Convert tools to LangChain format                       │
│  └─ Execute tools via MCP protocol                          │
└─────────────────────────────────────────────────────────────┘

Note: Current implementation uses direct function calls.
      MCP server is included for future scalability.
```

## Type Safety Flow

```
┌──────────────────────────────────────────────────────────────┐
│                       TYPE DEFINITIONS                        │
│                                                                │
│  types/risk.ts                                               │
│  ├─ RiskItem                                                 │
│  ├─ RiskQueryParams                                          │
│  ├─ RiskQueryResponse                                        │
│  ├─ RiskTrend                                                │
│  └─ ExecutiveSummary                                         │
└──────────────────────────────────────────────────────────────┘
                           │
                           │ Imported by all layers
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────────┐  ┌──────────────┐
│   Zod       │  │   TypeScript    │  │   OpenAI     │
│   Schemas   │  │   Interfaces    │  │   Structured │
│             │  │                 │  │   Output     │
│ Validation  │  │ Compile-time    │  │  Runtime     │
│ at runtime  │  │ type checking   │  │  validation  │
└─────────────┘  └─────────────────┘  └──────────────┘
```

## Component Integration

```
┌─────────────────────────────────────────────────────────────┐
│         Executive Dashboard (executive-dashboard.jsx)        │
│                                                               │
│  Current:                          Future:                   │
│  ├─ Mock responses                 ├─ useNaturalLanguageQuery()│
│  ├─ Hardcoded data                 ├─ Real-time queries      │
│  └─ Static battery metrics         └─ Dynamic risk updates  │
│                                                               │
│  Integration Steps:                                          │
│  1. Import useNaturalLanguageQuery hook                     │
│  2. Replace mockResponses with API calls                    │
│  3. Add loading states                                       │
│  4. Handle error responses                                   │
│  5. Display recommendations                                  │
└─────────────────────────────────────────────────────────────┘
```

## Security & Performance Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      FUTURE ENHANCEMENTS                      │
│                                                               │
│  Authentication        Rate Limiting        Caching          │
│  ──────────────        ────────────         ───────          │
│  • NextAuth.js         • Upstash Redis      • Redis          │
│  • Session tokens      • 10 req/min         • 5min TTL       │
│  • Role-based          • Per-user limits    • Query cache    │
│                                                               │
│  Monitoring            Error Tracking       Logging          │
│  ──────────            ──────────────       ───────          │
│  • OpenTelemetry       • Sentry             • Winston        │
│  • Performance         • Error alerts       • Structured     │
│  • Usage metrics       • Stack traces       • logs           │
└─────────────────────────────────────────────────────────────┘
```

## Files Reference

```
dashboard/
├── types/
│   └── risk.ts                    # Type definitions
├── lib/
│   ├── riskRepo.ts                # Data access layer
│   ├── agent.ts                   # LangChain agent
│   └── useNaturalLanguageQuery.tsx # React hook
├── mcp/
│   └── server.ts                  # MCP server
├── app/
│   ├── api/
│   │   └── query/
│   │       └── route.ts           # API endpoint
│   └── executive-dashboard.jsx    # UI component
├── install-nlq.sh                 # Installation script
├── test-nlq-api.sh                # Test script
├── NATURAL_LANGUAGE_QUERY_SETUP.md # Setup guide
├── IMPLEMENTATION_SUMMARY.md      # This file
└── .env.local                     # Environment variables
```

---

**Legend:**
- `→` Direct function call
- `┼` Data flow
- `├─` Branch
- `└─` End branch
- `▼` Input
- `▲` Output
