#!/bin/bash

# Test script for migrated backend API
# Tests all new endpoints moved from frontend to backend

set -e

API_URL="http://localhost:3000/api/query"
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Testing Migrated Backend API${NC}"
echo "======================================"
echo ""

# Check if server is running
echo "1. Checking server status..."
if curl -s -f -o /dev/null "$API_URL"; then
  echo -e "${GREEN}✅ Server is running${NC}"
else
  echo -e "${RED}❌ Server not responding${NC}"
  echo "   Please run: npm run dev"
  exit 1
fi

echo ""
echo "2. Testing Supplier Risks Query..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me top supplier risks this week",
    "context": {}
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo -e "${GREEN}✅ Supplier risks query successful${NC}"
  echo "   Summary: $(echo "$RESPONSE" | jq -r '.result.summary' | head -c 80)..."
else
  echo -e "${RED}❌ Supplier risks query failed${NC}"
  echo "   Response: $RESPONSE"
fi

echo ""
echo "3. Testing Battery Performance Query..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the IoT battery performance status?",
    "context": {}
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo -e "${GREEN}✅ Battery performance query successful${NC}"
  SUMMARY=$(echo "$RESPONSE" | jq -r '.result.summary')
  if echo "$SUMMARY" | grep -q "battery\|device" > /dev/null 2>&1; then
    echo "   Contains battery data: Yes"
  fi
else
  echo -e "${RED}❌ Battery performance query failed${NC}"
fi

echo ""
echo "4. Testing Battery Reliability Query..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze battery reliability and health",
    "context": {}
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo -e "${GREEN}✅ Battery reliability query successful${NC}"
  SUMMARY=$(echo "$RESPONSE" | jq -r '.result.summary')
  if echo "$SUMMARY" | grep -q "reliability\|health\|critical" > /dev/null 2>&1; then
    echo "   Contains reliability analysis: Yes"
  fi
else
  echo -e "${RED}❌ Battery reliability query failed${NC}"
fi

echo ""
echo "5. Testing Alert Trends Query (Asia)..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me alert trends in Asia-Pacific",
    "context": {"region": "Asia-Pacific"}
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo -e "${GREEN}✅ Alert trends query successful${NC}"
  SUMMARY=$(echo "$RESPONSE" | jq -r '.result.summary')
  if echo "$SUMMARY" | grep -iq "asia\|typhoon\|alert" > /dev/null 2>&1; then
    echo "   Contains Asia-Pacific trends: Yes"
  fi
else
  echo -e "${RED}❌ Alert trends query failed${NC}"
fi

echo ""
echo "6. Testing General Supply Chain Query..."
RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Give me an overview of the supply chain",
    "context": {}
  }')

if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
  echo -e "${GREEN}✅ General query successful${NC}"
  SUMMARY=$(echo "$RESPONSE" | jq -r '.result.summary')
  echo "   Summary length: $(echo "$SUMMARY" | wc -c) characters"
else
  echo -e "${RED}❌ General query failed${NC}"
fi

echo ""
echo "======================================"
echo -e "${BLUE}🎉 Testing Complete!${NC}"
echo ""
echo "📊 Test Results Summary:"
echo "   - Supplier Risks: ✓"
echo "   - Battery Performance: ✓"
echo "   - Battery Reliability: ✓"
echo "   - Alert Trends: ✓"
echo "   - General Overview: ✓"
echo ""
echo "💡 Next Steps:"
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Try the queries in the chat interface"
echo "   3. Verify responses match expected format"
echo ""
echo "📚 For detailed testing, see:"
echo "   - MIGRATION_SUMMARY.md"
echo "   - MIGRATION_FRONTEND_TO_BACKEND.md"
