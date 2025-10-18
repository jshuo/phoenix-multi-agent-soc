import { OpenAI } from 'openai';
import { NextResponse } from 'next/server';
import { spawn } from 'child_process';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// MCP Client to communicate with Bitcoin MCP server
class MCPClient {
  constructor() {
    this.messageId = 0;
  }

  async callTool(toolName, args = {}) {
    return new Promise((resolve, reject) => {
      // Spawn the Bitcoin MCP server process
      const mcpProcess = spawn('npx', ['-y', 'bitcoin-mcp@latest'], {
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      let hasResponded = false;

      const timeout = setTimeout(() => {
        if (!hasResponded) {
          mcpProcess.kill();
          reject(new Error('MCP server timeout'));
        }
      }, 30000); // 30 second timeout

      mcpProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        
        // Try to parse JSON-RPC responses
        const lines = stdout.split('\n');
        for (const line of lines) {
          if (line.trim() && !hasResponded) {
            try {
              const response = JSON.parse(line);
              if (response.id === this.messageId) {
                hasResponded = true;
                clearTimeout(timeout);
                mcpProcess.kill();
                
                if (response.error) {
                  reject(new Error(response.error.message || 'MCP server error'));
                } else {
                  resolve(response.result);
                }
              }
            } catch (e) {
              // Not JSON or not complete yet, continue accumulating
            }
          }
        }
      });

      mcpProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      mcpProcess.on('close', (code) => {
        clearTimeout(timeout);
        if (!hasResponded) {
          if (stderr) {
            reject(new Error(`MCP server error: ${stderr}`));
          } else {
            reject(new Error(`MCP server exited with code ${code}`));
          }
        }
      });

      mcpProcess.on('error', (error) => {
        clearTimeout(timeout);
        if (!hasResponded) {
          reject(new Error(`Failed to start MCP server: ${error.message}`));
        }
      });

      // Send the initialization message
      const initMessage = {
        jsonrpc: '2.0',
        id: this.messageId++,
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: {
            name: 'bitcoin-mcp-client',
            version: '1.0.0',
          },
        },
      };

      mcpProcess.stdin.write(JSON.stringify(initMessage) + '\n');

      // Wait a bit for initialization, then send the tool call
      setTimeout(() => {
        const toolMessage = {
          jsonrpc: '2.0',
          id: this.messageId,
          method: 'tools/call',
          params: {
            name: toolName,
            arguments: args,
          },
        };

        mcpProcess.stdin.write(JSON.stringify(toolMessage) + '\n');
        mcpProcess.stdin.end();
      }, 1000);
    });
  }
}

// Bitcoin MCP tools configuration
const tools = [
  {
    type: 'function',
    function: {
      name: 'bitcoin_generate_key',
      description: 'Generate a new Bitcoin key pair including private key (WIF), public key, and address',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_validate_address',
      description: 'Validate if a Bitcoin address is correctly formatted',
      parameters: {
        type: 'object',
        properties: {
          address: {
            type: 'string',
            description: 'The Bitcoin address to validate',
          },
        },
        required: ['address'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_decode_transaction',
      description: 'Decode a raw Bitcoin transaction hex string',
      parameters: {
        type: 'object',
        properties: {
          txHex: {
            type: 'string',
            description: 'The raw transaction hex string to decode',
          },
        },
        required: ['txHex'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_get_latest_block',
      description: 'Get information about the latest Bitcoin block',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_get_transaction',
      description: 'Get detailed information about a Bitcoin transaction by TXID',
      parameters: {
        type: 'object',
        properties: {
          txid: {
            type: 'string',
            description: 'The transaction ID (TXID)',
          },
        },
        required: ['txid'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'lightning_decode_invoice',
      description: 'Decode a BOLT11 Lightning Network invoice',
      parameters: {
        type: 'object',
        properties: {
          invoice: {
            type: 'string',
            description: 'The BOLT11 Lightning invoice string',
          },
        },
        required: ['invoice'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_get_address_transactions',
      description: 'Get transaction history for a Bitcoin address. Can retrieve up to the last 25 transactions.',
      parameters: {
        type: 'object',
        properties: {
          address: {
            type: 'string',
            description: 'The Bitcoin address to look up',
          },
          limit: {
            type: 'number',
            description: 'Maximum number of transactions to return (default: 5, max: 25)',
          },
        },
        required: ['address'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_get_address_info',
      description: 'Get detailed information about a Bitcoin address including balance, transaction count, and statistics',
      parameters: {
        type: 'object',
        properties: {
          address: {
            type: 'string',
            description: 'The Bitcoin address to look up',
          },
        },
        required: ['address'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_get_fee_estimates',
      description: 'Get current Bitcoin network fee estimates for different confirmation targets',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_get_price',
      description: 'Get current Bitcoin price in multiple currencies including USD, EUR, GBP, and market data (24h change, market cap, volume)',
      parameters: {
        type: 'object',
        properties: {
          currency: {
            type: 'string',
            description: 'Currency to get price in (default: usd). Supports: usd, eur, gbp, jpy, cny, etc.',
          },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bitcoin_get_price_history',
      description: 'Get Bitcoin price history for a specific time range (24h, 7d, 30d, 90d, 1y)',
      parameters: {
        type: 'object',
        properties: {
          days: {
            type: 'string',
            description: 'Time range: 1, 7, 30, 90, 365, or max',
          },
          currency: {
            type: 'string',
            description: 'Currency for price data (default: usd)',
          },
        },
        required: ['days'],
      },
    },
  },
];

async function executeFunction(functionName, args) {
  const client = new MCPClient();
  
  try {
    // Map OpenAI function names to MCP tool names
    const toolNameMap = {
      'bitcoin_generate_key': 'generate_key',
      'bitcoin_validate_address': 'validate_address',
      'bitcoin_decode_transaction': 'decode_transaction',
      'bitcoin_get_latest_block': 'get_latest_block',
      'bitcoin_get_transaction': 'get_transaction',
      'lightning_decode_invoice': 'decode_invoice',
    };

    // Handle Blockstream API calls for additional features
    if (functionName === 'bitcoin_get_address_transactions') {
      const { address, limit = 5 } = args;
      const actualLimit = Math.min(limit, 25);
      
      const response = await fetch(`https://blockstream.info/api/address/${address}/txs`);
      if (!response.ok) {
        throw new Error(`Failed to fetch transactions: ${response.statusText}`);
      }
      
      const allTxs = await response.json();
      const limitedTxs = allTxs.slice(0, actualLimit);
      
      return {
        address,
        transaction_count: allTxs.length,
        transactions: limitedTxs.map(tx => ({
          txid: tx.txid,
          confirmed: tx.status.confirmed,
          block_height: tx.status.block_height,
          block_time: tx.status.block_time ? new Date(tx.status.block_time * 1000).toISOString() : null,
          fee: tx.fee,
          size: tx.size,
          weight: tx.weight,
          vin_count: tx.vin.length,
          vout_count: tx.vout.length,
        })),
      };
    }

    if (functionName === 'bitcoin_get_address_info') {
      const { address } = args;
      
      const response = await fetch(`https://blockstream.info/api/address/${address}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch address info: ${response.statusText}`);
      }
      
      const data = await response.json();
      const balance = data.chain_stats.funded_txo_sum - data.chain_stats.spent_txo_sum;
      
      return {
        address,
        balance_sat: balance,
        balance_btc: (balance / 100000000).toFixed(8),
        total_received_sat: data.chain_stats.funded_txo_sum,
        total_sent_sat: data.chain_stats.spent_txo_sum,
        transaction_count: data.chain_stats.tx_count,
        unspent_outputs: data.chain_stats.funded_txo_count - data.chain_stats.spent_txo_count,
        mempool_stats: {
          transaction_count: data.mempool_stats.tx_count,
          balance_change_sat: data.mempool_stats.funded_txo_sum - data.mempool_stats.spent_txo_sum,
        },
      };
    }

    if (functionName === 'bitcoin_get_fee_estimates') {
      const response = await fetch('https://blockstream.info/api/fee-estimates');
      if (!response.ok) {
        throw new Error(`Failed to fetch fee estimates: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      return {
        fee_estimates: {
          fastest: { blocks: 1, sat_per_vbyte: data['1'] },
          half_hour: { blocks: 3, sat_per_vbyte: data['3'] },
          hour: { blocks: 6, sat_per_vbyte: data['6'] },
          economy: { blocks: 144, sat_per_vbyte: data['144'] },
          minimum: { blocks: 504, sat_per_vbyte: data['504'] },
        },
        all_estimates: data,
      };
    }

    if (functionName === 'bitcoin_get_price') {
      const { currency = 'usd' } = args;
      const currencies = ['usd', 'eur', 'gbp', 'jpy', 'cny'].join(',');
      
      const response = await fetch(
        `https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=${currencies}&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true&include_last_updated_at=true`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to fetch Bitcoin price: ${response.statusText}`);
      }
      
      const data = await response.json();
      const btcData = data.bitcoin;
      
      return {
        symbol: 'BTC',
        name: 'Bitcoin',
        current_price: {
          usd: btcData.usd,
          eur: btcData.eur,
          gbp: btcData.gbp,
          jpy: btcData.jpy,
          cny: btcData.cny,
        },
        primary_currency: currency.toLowerCase(),
        primary_price: btcData[currency.toLowerCase()] || btcData.usd,
        market_data: {
          market_cap_usd: btcData.usd_market_cap,
          volume_24h_usd: btcData.usd_24h_vol,
          price_change_24h_percent: btcData.usd_24h_change?.toFixed(2),
        },
        last_updated: new Date(btcData.last_updated_at * 1000).toISOString(),
      };
    }

    if (functionName === 'bitcoin_get_price_history') {
      const { days, currency = 'usd' } = args;
      
      const response = await fetch(
        `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=${currency}&days=${days}`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to fetch Bitcoin price history: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Sample data points to avoid overwhelming response
      const sampleRate = days === '1' ? 6 : days === '7' ? 24 : days === '30' ? 48 : 96;
      const sampledPrices = data.prices.filter((_, idx) => idx % sampleRate === 0);
      
      return {
        symbol: 'BTC',
        currency: currency.toUpperCase(),
        period: `${days} days`,
        price_history: sampledPrices.map(([timestamp, price]) => ({
          timestamp: new Date(timestamp).toISOString(),
          price: price,
        })),
        summary: {
          start_price: data.prices[0][1],
          end_price: data.prices[data.prices.length - 1][1],
          change: ((data.prices[data.prices.length - 1][1] - data.prices[0][1]) / data.prices[0][1] * 100).toFixed(2) + '%',
          highest: Math.max(...data.prices.map(p => p[1])),
          lowest: Math.min(...data.prices.map(p => p[1])),
        },
      };
    }

    const mcpToolName = toolNameMap[functionName] || functionName;
    const result = await client.callTool(mcpToolName, args);
    
    return result;
  } catch (error) {
    console.error(`Error executing ${functionName}:`, error);
    return { error: error.message };
  }
}

export async function POST(request) {
  try {
    const { message, conversationHistory = [] } = await request.json();

    const messages = [
      {
        role: 'system',
        content: `You are a Bitcoin blockchain assistant powered by the official Bitcoin MCP Server with enhanced Blockstream API integration. You help users with:

📦 **Bitcoin Blockchain:**
- Generate new Bitcoin key pairs (address, public key, private key)
- Validate Bitcoin addresses
- Get latest block information
- Look up transaction details by TXID
- Decode raw Bitcoin transactions
- Get address transaction history (up to last 25 transactions)
- Get address balance and statistics
- Get current network fee estimates

💰 **Bitcoin Market Data:**
- Get current Bitcoin price in multiple currencies (USD, EUR, GBP, JPY, CNY)
- Get 24-hour price change, market cap, and trading volume
- Get historical price data (24h, 7d, 30d, 90d, 1y)
- View price trends and statistics

⚡ **Lightning Network:**
- Decode BOLT11 Lightning invoices
- Parse invoice details (amount, description, expiry, etc.)

Always provide clear, formatted responses with relevant details. Use emojis to make responses engaging. When showing sensitive data like private keys, always include security warnings. Format transaction data in a readable way with timestamps and confirmation status. For price data, show multiple currencies and highlight significant changes.`,
      },
      ...conversationHistory,
      {
        role: 'user',
        content: message,
      },
    ];

    let response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      tools,
      tool_choice: 'auto',
    });

    let responseMessage = response.choices[0].message;

    // Handle function calls
    while (responseMessage.tool_calls && responseMessage.tool_calls.length > 0) {
      messages.push(responseMessage);

      for (const toolCall of responseMessage.tool_calls) {
        const functionName = toolCall.function.name;
        const functionArgs = JSON.parse(toolCall.function.arguments);
        const functionResult = await executeFunction(functionName, functionArgs);

        messages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify(functionResult),
        });
      }

      response = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages,
        tools,
        tool_choice: 'auto',
      });

      responseMessage = response.choices[0].message;
    }

    return NextResponse.json({
      message: responseMessage.content,
      conversationHistory: messages.slice(1), // Exclude system message
    });
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: error.message || 'An error occurred' },
      { status: 500 }
    );
  }
}
