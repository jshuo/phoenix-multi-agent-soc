#!/bin/bash

# Installation script for Natural Language Querying setup
# Run this from the dashboard directory

set -e

echo "🚀 Installing Natural Language Querying Dependencies..."
echo ""

# Check if we're in the dashboard directory
if [ ! -f "package.json" ]; then
  echo "❌ Error: package.json not found. Please run this script from the dashboard directory."
  exit 1
fi

# Install core dependencies
echo "📦 Installing core dependencies..."
npm install --save \
  @langchain/openai@^0.3.0 \
  @langchain/core@^0.3.0 \
  @modelcontextprotocol/sdk@^0.5.0 \
  zod@^3.23.0

echo ""
echo "✅ Core dependencies installed"
echo ""

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
  echo "⚙️  Creating .env.local template..."
  cat > .env.local << EOF
# OpenAI API Key (required)
OPENAI_API_KEY=sk-your-openai-api-key-here

# FastAPI Backend URL (optional, uses mock data if not set)
API_BASE_URL=http://localhost:8000

# Node Environment
NODE_ENV=development
EOF
  echo "✅ Created .env.local - Please update with your API keys"
else
  echo "ℹ️  .env.local already exists, skipping..."
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Update .env.local with your OPENAI_API_KEY"
echo "2. Run 'npm run dev' to start the development server"
echo "3. Test the API at: http://localhost:3000/api/query"
echo ""
echo "Example query:"
echo 'curl -X POST http://localhost:3000/api/query \\'
echo '  -H "Content-Type: application/json" \\'
echo '  -d '"'"'{"question": "What are the top risks this week?"}'"'"
echo ""
echo "📚 See NATURAL_LANGUAGE_QUERY_SETUP.md for detailed documentation"
