'use client';

import React, { useState } from 'react';
import { Send, Mic, TrendingUp, AlertTriangle, Globe, Package, Battery, Zap, Thermometer, Activity } from 'lucide-react';

const ExecutiveDashboard = () => {
  const [query, setQuery] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: 'assistant',
      content: 'Hello! I can help you analyze supply chain data. Try asking me about supplier risks, alert trends, battery performance, or regional insights.'
    }
  ]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userQuery = query;
    setQuery('');
    setMessages(prev => [...prev, { type: 'user', content: userQuery }]);
    setLoading(true);

    try {
      // Call the API
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userQuery,
          context: {}
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      if (data.success && data.result) {
        // Transform API response to message format
        const assistantMessage = {
          type: 'assistant',
          content: data.result.summary || 'Here are the results:',
          data: data.result.data,
          batteryData: data.result.batteryData,
          summary: data.result.detailedSummary,
          recommendations: data.result.recommendations
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error(data.error || 'Unknown error');
      }
    } catch (error) {
      console.error('Query error:', error);
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        summary: error.message
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleVoiceInput = () => {
    setIsListening(!isListening);
    if (!isListening) {
      setTimeout(() => {
        setQuery('Show me IoT battery performance analysis');
        setIsListening(false);
      }, 2000);
    }
  };

  const quickQuestions = [
    'Show me IoT battery performance analysis',
    'Analyze battery reliability across devices',
    'Show me top 3 supplier risks this week',
    'Summarize alert trends in Asia',
    'What is my supply chain efficiency?'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Package className="w-8 h-8 text-blue-400" />
              <div>
                <h1 className="text-2xl font-bold text-white">Supply Chain Intelligence</h1>
                <p className="text-sm text-slate-400">Executive Dashboard</p>
              </div>
            </div>
            <div className="flex gap-4 text-sm">
              <div className="bg-slate-800 px-4 py-2 rounded-lg">
                <div className="text-slate-400">Active Alerts</div>
                <div className="text-2xl font-bold text-white">12</div>
              </div>
              <div className="bg-slate-800 px-4 py-2 rounded-lg">
                <div className="text-slate-400">High Priority</div>
                <div className="text-2xl font-bold text-red-400">3</div>
              </div>
              <div className="bg-slate-800 px-4 py-2 rounded-lg">
                <div className="text-slate-400">Battery Health</div>
                <div className="text-2xl font-bold text-green-400">67%</div>
              </div>
              <div className="bg-slate-800 px-4 py-2 rounded-lg">
                <div className="text-slate-400">Efficiency</div>
                <div className="text-2xl font-bold text-green-400">94%</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
          {/* Stats Cards */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-2">
              <TrendingUp className="w-5 h-5 text-green-400" />
              <h3 className="text-slate-300 font-medium">Performance</h3>
            </div>
            <div className="text-3xl font-bold text-white mb-1">94%</div>
            <div className="text-sm text-slate-400">+2.3% from last week</div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-2">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
              <h3 className="text-slate-300 font-medium">Risk Score</h3>
            </div>
            <div className="text-3xl font-bold text-white mb-1">Medium</div>
            <div className="text-sm text-slate-400">3 suppliers need attention</div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-2">
              <Battery className="w-5 h-5 text-blue-400" />
              <h3 className="text-slate-300 font-medium">IoT Battery Health</h3>
            </div>
            <div className="text-3xl font-bold text-white mb-1">67%</div>
            <div className="text-sm text-slate-400">4 devices need replacement</div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-2">
              <Globe className="w-5 h-5 text-cyan-400" />
              <h3 className="text-slate-300 font-medium">Global Status</h3>
            </div>
            <div className="text-3xl font-bold text-white mb-1">247</div>
            <div className="text-sm text-slate-400">Active suppliers monitored</div>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl overflow-hidden">
          <div className="border-b border-slate-700 px-6 py-4">
            <h2 className="text-lg font-semibold text-white">AI Assistant</h2>
            <p className="text-sm text-slate-400">Ask questions in natural language</p>
          </div>

          {/* Quick Questions */}
          <div className="px-6 py-4 border-b border-slate-700">
            <div className="text-xs text-slate-400 mb-2">Quick Questions:</div>
            <div className="flex flex-wrap gap-2">
              {quickQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(q)}
                  className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-full transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Messages */}
          <div className="h-96 overflow-y-auto px-6 py-4 space-y-4">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-2xl ${msg.type === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-100'} rounded-lg px-4 py-3`}>
                  <div className="text-sm">{msg.content}</div>
                  {msg.data && (
                    <div className="mt-3 space-y-2">
                      {msg.data.map((item, i) => (
                        <div key={i} className="bg-slate-800 rounded-lg p-3 border border-slate-600">
                          <div className="flex justify-between items-start mb-2">
                            <div className="font-semibold text-white">{item.name}</div>
                            <span className={`text-xs px-2 py-1 rounded ${item.risk === 'High' ? 'bg-red-500/20 text-red-300' : 'bg-amber-500/20 text-amber-300'}`}>
                              {item.risk} Risk
                            </span>
                          </div>
                          <div className="text-sm text-slate-300">{item.issue}</div>
                          <div className="text-xs text-slate-400 mt-1">Est. Impact: {item.impact}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  {msg.batteryData && (
                    <div className="mt-3 space-y-2">
                      {msg.batteryData.map((battery, i) => (
                        <div key={i} className="bg-slate-800 rounded-lg p-3 border border-slate-600">
                          <div className="flex justify-between items-start mb-2">
                            <div className="font-semibold text-white flex items-center gap-2">
                              <Battery className="w-4 h-4" />
                              {battery.device}
                            </div>
                            <span className={`text-xs px-2 py-1 rounded ${
                              battery.health === 'Excellent' ? 'bg-green-500/20 text-green-300' :
                              battery.health === 'Good' ? 'bg-blue-500/20 text-blue-300' :
                              battery.health === 'Warning' ? 'bg-amber-500/20 text-amber-300' :
                              'bg-red-500/20 text-red-300'
                            }`}>
                              {battery.health}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div className="flex items-center gap-2">
                              <Zap className="w-3 h-3 text-yellow-400" />
                              <span className="text-slate-300">Voltage: {battery.voltage}V</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Activity className="w-3 h-3 text-green-400" />
                              <span className="text-slate-300">Capacity: {battery.capacity}%</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Thermometer className="w-3 h-3 text-red-400" />
                              <span className="text-slate-300">Temp: {battery.temperature}°C</span>
                            </div>
                            <div className="text-slate-300">Cycles: {battery.cycles}</div>
                          </div>
                          <div className="text-xs text-slate-400 mt-2">Predicted Life: {battery.predictedLife}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  {msg.summary && (
                    <div className="mt-3 text-sm whitespace-pre-line bg-slate-800 rounded-lg p-3 border border-slate-600">
                      {msg.summary}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Input */}
          <div className="border-t border-slate-700 px-6 py-4">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask me anything about your supply chain..."
                className="flex-1 bg-slate-700 text-white placeholder-slate-400 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="button"
                onClick={handleVoiceInput}
                className={`p-3 rounded-lg transition-colors ${isListening ? 'bg-red-500 hover:bg-red-600' : 'bg-slate-700 hover:bg-slate-600'}`}
              >
                <Mic className={`w-5 h-5 ${isListening ? 'text-white animate-pulse' : 'text-slate-300'}`} />
              </button>
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
              >
                <Send className="w-5 h-5" />
                Send
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExecutiveDashboard;