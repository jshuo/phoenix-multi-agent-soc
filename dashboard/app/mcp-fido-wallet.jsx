'use client';

import { useState, useRef, useEffect } from 'react';

const BitcoinMCPDemo = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Welcome to Bitcoin MCP Interactive Query Service! 🚀\n\nPowered by the official **Bitcoin MCP Server** with enhanced blockchain & market data integration\n\nI can help you with:\n\n📦 **Bitcoin Blockchain:**\n• Generate new Bitcoin key pairs\n• Validate Bitcoin addresses\n• Get latest block information\n• Look up transaction details by TXID\n• Decode raw Bitcoin transactions\n• Get address transaction history\n• Check address balance and statistics\n• Get network fee estimates\n\n💰 **Bitcoin Market Data:**\n• Get current Bitcoin price (USD, EUR, GBP, JPY, CNY)\n• View 24h price change, market cap, volume\n• Get historical price charts (24h, 7d, 30d, 1y)\n\n⚡ **Lightning Network:**\n• Decode BOLT11 Lightning invoices\n\nJust ask me naturally! For example:\n• "What is the current Bitcoin price?"\n• "Show me Bitcoin price in EUR and GBP"\n• "Get last 7 days price history"\n• "Get last 5 transactions of 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"\n• "What\'s the latest Bitcoin block?"\n\nWhat would you like to do?'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const processQuery = async (userQuery) => {
    // Add user message
    const userMessage = { role: 'user', content: userQuery };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('/api/bitcoin-mcp', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userQuery,
          conversationHistory,
        }),
      });

      const data = await response.json();

      if (data.error) {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `❌ **Error**: ${data.error}\n\nPlease check your query and try again.` 
        }]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.message 
        }]);
        setConversationHistory(data.conversationHistory);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `❌ **Error**: ${error.message}\n\nPlease check your connection and try again.` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const query = input.trim();
    setInput('');
    await processQuery(query);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-purple-800 p-5">
      <div className="max-w-7xl mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white p-8 text-center">
          <h1 className="text-4xl font-bold mb-2 flex items-center justify-center gap-3">
            <span className="text-5xl">₿</span> Bitcoin MCP Interactive Query
          </h1>
          <p className="text-white/90 text-lg">
            Natural language interface for Bitcoin blockchain powered by Model Context Protocol
          </p>
        </div>

        <div className="p-8">
          {/* Chat Interface */}
          <div className="flex flex-col h-[700px] max-w-4xl mx-auto">
            <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white p-4 rounded-t-xl">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                💬 Interactive Query Service
              </h2>
              <p className="text-sm text-white/80 mt-1">Ask questions in natural language</p>
            </div>
            
            {/* Messages */}
            <div className="flex-1 bg-gray-50 p-4 overflow-y-auto space-y-4">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] p-4 rounded-lg ${
                    msg.role === 'user' 
                      ? 'bg-orange-500 text-white' 
                      : 'bg-white border-2 border-gray-200 text-gray-800'
                  }`}>
                    <div className="whitespace-pre-wrap text-sm leading-relaxed">
                      {msg.content}
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border-2 border-gray-200 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-600">
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      <span className="ml-2 text-sm">Processing...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="border-t-2 border-gray-200 p-4 bg-white rounded-b-xl">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about Bitcoin: price, transactions, balance, latest block, fees, history..."
                  className="flex-1 p-3 border-2 border-gray-200 rounded-lg text-sm focus:outline-none focus:border-orange-500"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="px-6 py-3 bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg font-semibold transition-all hover:shadow-lg hover:-translate-y-0.5 disabled:bg-gray-400 disabled:cursor-not-allowed disabled:transform-none"
                >
                  Send
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BitcoinMCPDemo;
