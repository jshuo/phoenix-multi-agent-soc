#!/bin/bash

# Quick test script for Natural Language Query API
# Run this after starting the dev server (npm run dev)

set -e

API_URL="http://localhost:3000/api/query"

echo "🧪 Testing Natural Language Query API"
echo "======================================"
echo ""

# Check if server is running
echo "1. Checking if dev server is running..."
if curl -s -f -o /dev/null "$API_URL"; then
  echo "✅ Server is responding"
else
  echo "❌ Server not responding at $API_URL"
  echo "   Please run 'npm run dev' first"
  exit 1
fi

echo ""
echo "2. Testing basic query (GET)..."
RESPONSE=$(curl -s "${API_URL}?q=What%20are%20the%20top%20risks%3F")
if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo "✅ GET request successful"
  echo "   Summary: $(echo "$RESPONSE" | jq -r '.result.summary' | head -c 100)..."
else
  echo "❌ GET request failed"
  echo "   Response: $RESPONSE"
fi

echo ""
echo "3. Testing POST query with context..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top 3 risks in Asia-Pacific this week?",
    "context": {
      "region": "Asia-Pacific",
      "days": 7
    }
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo "✅ POST request successful"
  RISK_COUNT=$(echo "$RESPONSE" | jq -r '.result.data | length')
  echo "   Found $RISK_COUNT risks"
  echo "   Summary: $(echo "$RESPONSE" | jq -r '.result.summary' | head -c 100)..."
  
  if [ "$RISK_COUNT" -gt 0 ]; then
    echo ""
    echo "   Top Risk:"
    echo "$RESPONSE" | jq -r '.result.data[0] | "   - Asset: \(.assetId), Score: \(.score), Severity: \(.severity)"'
  fi
else
  echo "❌ POST request failed"
  echo "   Response: $RESPONSE"
fi

echo ""
echo "4. Testing battery performance query..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me battery performance issues"
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo "✅ Battery query successful"
  echo "   $(echo "$RESPONSE" | jq -r '.result.summary' | head -c 100)..."
else
  echo "❌ Battery query failed"
fi

echo ""
echo "5. Testing trend analysis query..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the risk trends over the past week?"
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo "✅ Trend query successful"
  HAS_TRENDS=$(echo "$RESPONSE" | jq -e '.result.trends' > /dev/null 2>&1 && echo "yes" || echo "no")
  echo "   Trends included: $HAS_TRENDS"
else
  echo "❌ Trend query failed"
fi

echo ""
echo "======================================"
echo "🎉 Testing complete!"
echo ""
echo "💡 Tips:"
echo "   - Check OPENAI_API_KEY in .env.local if requests fail"
echo "   - View full responses: curl http://localhost:3000/api/query?q=Top%20risks | jq"
echo "   - Check logs in terminal running 'npm run dev'"
echo ""
echo "📚 See NATURAL_LANGUAGE_QUERY_SETUP.md for more examples"
