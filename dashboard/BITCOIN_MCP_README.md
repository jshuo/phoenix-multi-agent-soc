# Bitcoin MCP Interactive Query

This is a Next.js application that provides a natural language interface for Bitcoin blockchain operations using OpenAI's GPT-4o and the **official Bitcoin MCP Server** (`bitcoin-mcp@latest`).

## Features

### Bitcoin Blockchain
- � Generate new Bitcoin key pairs (private key, public key, address)
- ✅ Validate Bitcoin addresses
- � Get latest block information
- 💸 Look up transaction details by TXID
- 🔓 Decode raw Bitcoin transactions

### Lightning Network
- ⚡ Decode BOLT11 Lightning invoices
- 📊 Parse invoice details (amount, description, expiry)

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment variables:**
   
   Make sure your `.env` file contains:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open the application:**
   
   Navigate to `http://localhost:3000/mcp-fido-wallet`

## How It Works

The application uses:

- **Bitcoin MCP Server** (`npx -y bitcoin-mcp@latest`) - Official MCP server for Bitcoin operations
- **OpenAI GPT-4o** with function calling to understand natural language queries
- **JSON-RPC over STDIO** - Communication protocol between Next.js and MCP server
- **Next.js API Routes** to spawn and manage the MCP server process
- **React** for the interactive chat interface

## Example Queries

Try asking:

### Bitcoin Operations
- "Generate a new Bitcoin key pair"
- "Validate this address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
- "What's the latest Bitcoin block?"
- "Show me transaction details for a1075db55d416d3ca199f55b6084e2115b9345e16c5cf302fc80e9d5fbf5d48d"
- "Decode this raw transaction: 0100000001..."

### Lightning Network
- "Decode this Lightning invoice: lnbc..."
- "Parse this BOLT11 invoice"

## Architecture

```
User Query → Frontend (React)
    ↓
    API Route (/api/bitcoin-mcp)
    ↓
    OpenAI GPT-4o (function calling)
    ↓
    Spawn MCP Server Process
    ↓
    JSON-RPC via STDIO
    ↓
    Bitcoin MCP Server (npx bitcoin-mcp@latest)
    ↓
    Tool Execution (Bitcoin Core, Blockstream API, Lightning)
    ↓
    Response → User
```

## Technologies

- **Next.js 14** - React framework
- **OpenAI API** - GPT-4o for natural language understanding
- **Bitcoin MCP Server** - Official MCP server (`bitcoin-mcp@latest`)
- **Node.js Child Process** - Spawn and communicate with MCP server
- **JSON-RPC 2.0** - Protocol for MCP communication
- **Tailwind CSS** - Styling
- **Model Context Protocol** - Standard for tool orchestration

## Technical Details

### MCP Server Integration

The application spawns the Bitcoin MCP server as a child process for each request:

```javascript
spawn('npx', ['-y', 'bitcoin-mcp@latest'])
```

Communication happens via STDIO using JSON-RPC 2.0:
1. Initialize the MCP server
2. Send tool call request
3. Parse JSON-RPC response
4. Return result to OpenAI

### Supported MCP Tools

- `generate_key` - Generate Bitcoin key pairs
- `validate_address` - Validate Bitcoin addresses
- `decode_transaction` - Decode raw transactions
- `get_latest_block` - Get latest block info
- `get_transaction` - Get transaction details
- `decode_invoice` - Decode Lightning invoices

## Security Notes

⚠️ **Important:**
- Private keys generated are displayed in the UI - never use in production
- This is a demo application - do not use generated keys for real funds
- Always verify transactions before broadcasting
- Keep API keys secure and never commit them to version control

## License

MIT
