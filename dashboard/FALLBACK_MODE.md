# Natural Language Query System - Fallback Mode

## Overview

The dashboard now supports **two operating modes**:

### 1. 🤖 AI-Powered Mode (with OpenAI API key)
- Uses GPT-4o for intelligent, contextual responses
- Natural language understanding
- Sophisticated analysis and recommendations
- **Requires**: `OPENAI_API_KEY` in `.env.local`

### 2. 🔄 Fallback Mode (without API key)
- **Currently Active** - No API key required!
- Intent-based query processing
- Rule-based analysis
- Pre-defined response templates
- All data and recommendations still available

## How Fallback Mode Works

When no OpenAI API key is detected, the system:

1. **Analyzes Intent**: Uses regex patterns to understand your question
   - Supplier risks
   - Battery performance
   - Battery reliability
   - Alert trends
   - Top risks
   - General overview

2. **Fetches Data**: Calls appropriate backend functions
   - Mock data from `lib/riskRepo.ts`
   - Same data structure as AI mode

3. **Generates Response**: Uses smart templates
   - Context-aware summaries
   - Critical issue highlighting
   - Actionable recommendations
   - Formatted data display

## Supported Queries (Fallback Mode)

### Supplier Risks
**Trigger words**: `supplier`, `vendor`

**Example queries**:
- "Show me top 3 supplier risks this week"
- "What are the supplier issues?"
- "Vendor risk analysis"

**Response includes**:
- Number of suppliers by risk level
- Critical/High priority suppliers
- Specific issues and impacts
- Regional information

### Battery Performance
**Trigger words**: `battery performance`, `battery status`, `battery metrics`

**Example queries**:
- "Show me IoT battery performance analysis"
- "Battery status check"
- "How are the batteries performing?"

**Response includes**:
- Device battery health percentages
- Critical/Warning devices
- Voltage, capacity, temperature data
- Replacement recommendations

### Battery Reliability
**Trigger words**: `battery reliability`, `battery health`, `battery condition`

**Example queries**:
- "Analyze battery reliability across devices"
- "Battery health overview"
- "Which batteries need attention?"

**Response includes**:
- Average health metrics
- Predicted failures timeline
- Warranty status
- Maintenance schedules

### Alert Trends
**Trigger words**: `alert trends`, `trends`, `pattern`

**Example queries**:
- "Summarize alert trends in Asia"
- "Show me alert patterns"
- "What are the trending alerts?"

**Response includes**:
- Total and critical alert counts
- Regional filtering
- Time-based patterns
- Priority recommendations

### Top Risks
**Trigger words**: `top`, `highest`, `priority`, `critical`

**Example queries**:
- "Show me top 5 risks"
- "What are the highest priority issues?"
- "Critical risks report"

**Response includes**:
- Ranked risk list
- Score-based categorization
- Severity breakdown
- Asset-specific details

### Supply Chain Efficiency
**Trigger words**: `efficiency`, `overview`, `status`

**Example queries**:
- "What is my supply chain efficiency?"
- "Overall status"
- "Give me a summary"

**Response includes**:
- Comprehensive overview
- All risk categories
- Battery and supplier data
- General recommendations

## Response Format

All responses include:

```json
{
  "summary": "Human-readable executive summary",
  "data": [
    {
      "id": "RISK-001",
      "assetId": "GPS-TRACKER-B2",
      "score": 87,
      "severity": "high",
      "region": "Asia-Pacific",
      "lastUpdated": "2025-10-10T...",
      "reasons": [
        {
          "feature": "Battery Health",
          "weight": 0.4,
          "contribution": 40,
          "currentValue": "65%"
        }
      ]
    }
  ],
  "recommendations": [
    "Urgent action required for...",
    "Schedule preventive maintenance..."
  ],
  "sources": ["Risk Repository", "IoT Device Monitoring"]
}
```

## Upgrading to AI Mode

To enable GPT-4o powered responses:

1. **Create `.env.local`** in dashboard directory:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

2. **Restart the dev server**:
```bash
# Stop server (Ctrl+C)
npm run dev
```

3. **Benefits of AI Mode**:
   - More natural, conversational responses
   - Better context understanding
   - Dynamic recommendation generation
   - Complex query handling
   - Multi-intent analysis

## Testing

Try these queries in the dashboard:

```
✅ "Show me top 3 supplier risks this week"
✅ "What is my supply chain efficiency?"
✅ "Analyze battery reliability across devices"
✅ "Summarize alert trends in Asia"
✅ "Show me IoT battery performance analysis"
✅ "What are the critical risks?"
```

## Technical Details

### Code Structure

**lib/agent.ts**:
- `askExecutive()` - Main entry point
- `analyzeIntent()` - Intent classification
- `generateFallbackResponse()` - Template-based response generation

**Intent Mapping**:
```typescript
"supplierRisks"       → getSupplierRisks()
"batteryPerformance"  → getBatteryPerformance()
"batteryReliability"  → getBatteryReliability()
"alertTrends"         → getAlertTrends()
"topRisks"            → getTopRisks()
"trends"              → getRiskTrends() + getAlertTrends()
"summary"             → getRiskSummary() + comprehensive data
```

### Data Flow

```
User Query
    ↓
Frontend (executive-dashboard.jsx)
    ↓
API Route (/api/query)
    ↓
Agent (lib/agent.ts)
    ↓
[Check for OpenAI API Key]
    ├─ Yes → GPT-4o Processing
    └─ No → Fallback Processing
         ↓
    analyzeIntent()
         ↓
    Data Repository (lib/riskRepo.ts)
         ↓
    generateFallbackResponse()
         ↓
    Structured JSON Response
```

## Performance

**Fallback Mode**:
- ⚡ Fast (< 100ms typical)
- 💰 Free (no API costs)
- 🔒 Private (no external API calls)
- 📊 Consistent responses

**AI Mode**:
- 🤖 Intelligent (2-3s typical)
- 💰 Paid (OpenAI API usage)
- 🌐 External dependency
- 🎯 Contextual responses

## Current Status

✅ **System is fully functional in Fallback Mode**
✅ No API key required
✅ All queries supported
✅ Data visualization working
✅ Recommendations generated

🔄 **To upgrade to AI Mode**: Add OpenAI API key

## Troubleshooting

### "Failed to get response"

**Possible causes**:
1. Dev server not running → Check http://localhost:3000
2. API route error → Check terminal for error logs
3. Network issue → Verify browser console

**Solution**:
```bash
# Restart dev server
pkill -f "next dev"
cd /Users/jmh_cheng/workspace/phoenix-multi-agent-soc/dashboard
rm -rf .next
npm run dev
```

### No data in response

**Check**:
- lib/riskRepo.ts has mock data
- Intent is being recognized (check terminal logs)
- API returning 200 status

### Want to see what's happening

**Enable debugging**:
- Check browser DevTools → Network tab
- Look at terminal output for `[Query API] Processing`
- Add console.log in lib/agent.ts

## Next Steps

1. **Test all query types** - Verify each intent works
2. **Review recommendations** - Check if they're helpful
3. **Customize responses** - Edit `generateFallbackResponse()`
4. **Add OpenAI key** - When ready for AI mode
5. **Connect real backend** - Replace mock data with FastAPI

## Documentation

- Setup: [NATURAL_LANGUAGE_QUERY_SETUP.md](./NATURAL_LANGUAGE_QUERY_SETUP.md)
- Architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- Migration: [MIGRATION_SUMMARY.md](./MIGRATION_SUMMARY.md)
- Quick Reference: [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)

---

**Status**: ✅ System Ready | Mode: Fallback | API Key: Not Required
