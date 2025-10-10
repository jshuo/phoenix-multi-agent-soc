# Quick Reference Card - Natural Language Query System

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install dependencies
cd dashboard && ./install-nlq.sh

# 2. Add your OpenAI key to .env.local
echo "OPENAI_API_KEY=sk-your-key-here" >> .env.local

# 3. Start the server and test
npm run dev
./test-nlq-api.sh
```

## 📁 Files Created

| File | Purpose |
|------|---------|
| `types/risk.ts` | TypeScript type definitions |
| `lib/riskRepo.ts` | Data fetching layer (mock data) |
| `lib/agent.ts` | LangChain agent with GPT-4o |
| `lib/useNaturalLanguageQuery.tsx` | React hook for queries |
| `mcp/server.ts` | MCP server (optional) |
| `app/api/query/route.ts` | Next.js API endpoints |
| `NATURAL_LANGUAGE_QUERY_SETUP.md` | Full documentation |
| `IMPLEMENTATION_SUMMARY.md` | Implementation overview |
| `ARCHITECTURE.md` | System architecture diagrams |

## 🔧 API Endpoints

### POST /api/query
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top 3 risks?",
    "context": { "region": "Asia-Pacific", "days": 7 }
  }'
```

### GET /api/query
```bash
curl "http://localhost:3000/api/query?q=Show%20battery%20issues"
```

## 💬 Example Queries

| Query Type | Example |
|------------|---------|
| **Top Risks** | "What are the top 3 risks in Asia-Pacific?" |
| **Battery** | "Show me IoT battery performance issues" |
| **Trends** | "What are the risk trends over the past month?" |
| **Summary** | "Give me an overview of current risks" |
| **Specific Asset** | "What's the status of GPS-TRACKER-B2?" |

## 🎯 Response Format

```json
{
  "success": true,
  "result": {
    "summary": "Executive summary text...",
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
      "Immediate replacement recommended..."
    ],
    "trends": [...],
    "sources": ["Risk Repository"]
  }
}
```

## 🔌 React Integration

```typescript
import { useNaturalLanguageQuery } from '@/lib/useNaturalLanguageQuery';

function MyComponent() {
  const { query, loading, error } = useNaturalLanguageQuery();
  
  const handleQuery = async () => {
    const result = await query('Top risks?', { region: 'Europe' });
    console.log(result);
  };
  
  return <button onClick={handleQuery}>Query</button>;
}
```

## 🧩 Main Functions

### askExecutive()
```typescript
import { askExecutive } from '@/lib/agent';

const result = await askExecutive(
  'What are the top risks?',
  { region: 'Asia-Pacific', days: 7 }
);
```

### getTopRisks()
```typescript
import { getTopRisks } from '@/lib/riskRepo';

const response = await getTopRisks({
  region: 'Europe',
  severity: 'high',
  limit: 5
});
```

## 🔍 Intent Detection

The system automatically detects:

| Intent | Trigger Words | Tools Called |
|--------|---------------|--------------|
| `topRisks` | "top", "highest", "priority" | getTopRisks() |
| `trends` | "trend", "over time", "pattern" | getRiskTrends() |
| `summary` | "summary", "overview", "how many" | getRiskSummary() |
| `general` | (default) | All tools |

## 📊 Mock Data Available

| Asset ID | Region | Severity | Score |
|----------|--------|----------|-------|
| GPS-TRACKER-B2 | Asia-Pacific | High | 87 |
| PRESSURE-MONITOR-D4 | Europe | High | 92 |
| TEMP-SENSOR-A1 | North America | Medium | 45 |
| HUMIDITY-SENSOR-C3 | Asia-Pacific | Low | 22 |

## ⚙️ Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
API_BASE_URL=http://localhost:8000  # FastAPI backend
NODE_ENV=development
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot find module" | Run `./install-nlq.sh` |
| "Unauthorized" OpenAI | Check `OPENAI_API_KEY` in `.env.local` |
| Server not responding | Run `npm run dev` |
| TypeScript errors | Run `npm run type-check` |

## 📚 Documentation

- **Full Setup**: `NATURAL_LANGUAGE_QUERY_SETUP.md`
- **Architecture**: `ARCHITECTURE.md`
- **Summary**: `IMPLEMENTATION_SUMMARY.md`

## 🔗 External Resources

- [LangChain JS Docs](https://js.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Next.js API Routes](https://nextjs.org/docs/app/building-your-application/routing/route-handlers)

## ✅ Implementation Checklist

- [x] Type definitions created
- [x] Risk repository implemented
- [x] LangChain agent configured
- [x] API routes created
- [x] React hook provided
- [x] MCP server implemented
- [x] Documentation written
- [ ] Frontend integration (next step)
- [ ] Backend connection (when ready)
- [ ] Testing & validation

## 🎓 Phase I Alignment (MOEA Proposal)

| Requirement | Status |
|-------------|--------|
| IoT Battery Performance Analysis | ✅ Implemented |
| AI-Generated Priority Alerts | ✅ Implemented |
| LLM + Risk Scoring | ✅ GPT-4o + Explainable scores |
| Executive Dashboard | ✅ Ready for integration |
| Kalman Filter / EWMA | 🔄 Ready for real data |

## 🚦 Status

**Implementation**: ✅ Complete (Core functionality)
**Testing**: 🔄 Ready to test
**Integration**: 🔄 Next step
**Deployment**: ⏳ Pending

---

**Last Updated**: October 10, 2025
**Version**: 1.0.0
**Project**: ITracXing x Arviem MOEA AI Application
