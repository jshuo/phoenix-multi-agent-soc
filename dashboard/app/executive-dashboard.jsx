'use client';

import { useState } from 'react';
import { Send, Mic, TrendingUp, AlertTriangle, Package, Globe, Battery, Zap, Thermometer, Activity, Cloud, Droplets, Wind } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getCityNameForRegion } from '@/lib/weather';

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
          weatherData: data.result.weatherData,
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
                <h1 className="text-2xl font-bold text-white">Arviem-ItracXing: AI Monitoring as a Service </h1>
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
          <div className="h-[28rem] md:h-[36rem] lg:h-[40rem] overflow-y-auto px-6 py-4 space-y-6">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-3xl w-full ${msg.type === 'user' ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-900/50' : 'bg-slate-800/80 text-slate-100 border border-slate-700 shadow-xl'} rounded-2xl overflow-hidden`}>
                  {/* Message Header */}
                  {msg.type === 'assistant' && (
                    <div className="bg-gradient-to-r from-slate-700/50 to-slate-800/50 px-5 py-3 border-b border-slate-700/50">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                        <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">AI Analysis</span>
                      </div>
                    </div>
                  )}
                  
                  {/* Message Content */}
                  <div className="px-5 py-4">
                    {/* Enhanced Message Content Styling */}
                    <div className="relative">
                      {/* Decorative Quote Mark for Assistant */}
                      {msg.type === 'assistant' && (
                        <div className="absolute -left-2 -top-1 text-4xl text-blue-400/20 font-serif">"</div>
                      )}
                      
                      {/* Main Content with Enhanced Typography */}
                      {/* Message content with markdown support */}
                      <div className={`
                        text-base leading-relaxed
                        ${msg.type === 'user' 
                          ? 'font-medium text-white' 
                          : 'font-normal text-slate-100'
                        }
                        ${msg.content.length > 200 ? 'text-sm' : 'text-base'}
                      `}>
                        {/* Render markdown for assistant messages, plain text for user */}
                        {msg.type === 'assistant' ? (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              // Style for bold text
                              strong: ({node, ...props}) => <strong className="font-bold text-white" {...props} />,
                              // Style for italic text
                              em: ({node, ...props}) => <em className="italic text-blue-300" {...props} />,
                              // Style for lists
                              ul: ({node, ...props}) => <ul className="list-disc list-inside space-y-1 my-2" {...props} />,
                              ol: ({node, ...props}) => <ol className="list-decimal list-inside space-y-1 my-2" {...props} />,
                              li: ({node, ...props}) => <li className="text-slate-200 ml-2" {...props} />,
                              // Style for paragraphs
                              p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                              // Style for horizontal rules
                              hr: ({node, ...props}) => <hr className="my-4 border-slate-600" {...props} />,
                            }}
                          >
                            {msg.content}
                          </ReactMarkdown>
                        ) : (
                          msg.content
                        )}
                      </div>
                      
                      {/* Visual separator if there's more content below */}
                      {(msg.data?.length > 0 || msg.batteryData?.length > 0 || msg.weatherData || msg.recommendations?.length > 0 || msg.summary) && (
                        <div className="mt-4 pt-4 border-t border-slate-700/30"></div>
                      )}
                    </div>
                    
                    {/* Weather Data Cards - Show before battery data */}
                    {msg.weatherData && Object.keys(msg.weatherData).length > 0 && (
                      <div className="mt-5 space-y-3">
                        <div className="flex items-center gap-2 mb-3">
                          <Cloud className="w-4 h-4 text-sky-400" />
                          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">City Weather Conditions</span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                          {Object.entries(msg.weatherData).map(([region, weather]) => {
                            // Get city name from region or extract from location string
                            const cityName = getCityNameForRegion(region) || weather.location?.split(',')[0] || region;
                            return (
                            <div key={region} className="bg-gradient-to-br from-sky-900/30 to-blue-900/30 rounded-xl p-4 border border-sky-700/50 hover:border-sky-600 transition-all hover:shadow-lg hover:shadow-sky-900/50">
                              <div className="flex items-center gap-2 mb-3">
                                <Globe className="w-4 h-4 text-sky-400" />
                                <div className="font-bold text-white text-sm">{cityName}</div>
                              </div>
                              
                              <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-2">
                                  <Thermometer className={`w-5 h-5 ${
                                    weather.temperature > 30 ? 'text-red-400' :
                                    weather.temperature > 20 ? 'text-amber-400' :
                                    'text-blue-400'
                                  }`} />
                                  <span className={`text-2xl font-bold ${
                                    weather.temperature > 30 ? 'text-red-400' :
                                    weather.temperature > 20 ? 'text-amber-400' :
                                    'text-blue-400'
                                  }`}>
                                    {weather.temperature?.toFixed(1)}°C
                                  </span>
                                </div>
                                <div className="text-xs text-slate-300 bg-slate-800/50 px-2 py-1 rounded">
                                  {weather.conditions}
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                {weather.humidity !== null && weather.humidity !== undefined && (
                                  <div className="bg-slate-900/50 rounded-lg p-2 border border-slate-700/30">
                                    <div className="flex items-center gap-1 mb-1">
                                      <Droplets className="w-3 h-3 text-cyan-400" />
                                      <span className="text-slate-400">Humidity</span>
                                    </div>
                                    <div className="text-white font-semibold">{weather.humidity}%</div>
                                  </div>
                                )}
                                {weather.windSpeed !== null && weather.windSpeed !== undefined && (
                                  <div className="bg-slate-900/50 rounded-lg p-2 border border-slate-700/30">
                                    <div className="flex items-center gap-1 mb-1">
                                      <Wind className="w-3 h-3 text-slate-400" />
                                      <span className="text-slate-400">Wind</span>
                                    </div>
                                    <div className="text-white font-semibold">{weather.windSpeed?.toFixed(1)} km/h</div>
                                  </div>
                                )}
                                {weather.precipitation !== null && weather.precipitation !== undefined && (
                                  <div className="bg-slate-900/50 rounded-lg p-2 border border-slate-700/30">
                                    <div className="flex items-center gap-1 mb-1">
                                      <Droplets className="w-3 h-3 text-blue-400" />
                                      <span className="text-slate-400">Rain</span>
                                    </div>
                                    <div className="text-white font-semibold">{weather.precipitation} mm</div>
                                  </div>
                                )}
                                {weather.pressure !== null && weather.pressure !== undefined && (
                                  <div className="bg-slate-900/50 rounded-lg p-2 border border-slate-700/30">
                                    <div className="flex items-center gap-1 mb-1">
                                      <Activity className="w-3 h-3 text-purple-400" />
                                      <span className="text-slate-400">Pressure</span>
                                    </div>
                                    <div className="text-white font-semibold">{weather.pressure} hPa</div>
                                  </div>
                                )}
                              </div>
                              
                              {weather.location && (
                                <div className="mt-2 text-xs text-slate-500 truncate">
                                  📍 {weather.location}
                                </div>
                              )}
                            </div>
                            );
                          })}
                        </div>
                        
                        {/* Weather Impact Note */}
                        <div className="bg-sky-900/10 border border-sky-700/30 rounded-lg p-3">
                          <div className="flex items-start gap-2">
                            <Cloud className="w-4 h-4 text-sky-400 mt-0.5" />
                            <div className="text-xs text-slate-300">
                              <span className="font-semibold text-sky-300">Environmental Impact:</span> High temperatures 
                              ({">"} 30°C) can accelerate battery degradation and reduce capacity. Monitor devices in hot regions closely.
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Risk/Supplier Data Cards */}
                    {msg.data && msg.data.length > 0 && (
                      <div className="mt-5 space-y-3">
                        <div className="flex items-center gap-2 mb-3">
                          <AlertTriangle className="w-4 h-4 text-amber-400" />
                          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Risk Analysis ({msg.data.length} items)</span>
                        </div>
                        {msg.data.map((item, i) => (
                          <div key={i} className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-xl p-4 border border-slate-700/50 hover:border-slate-600 transition-all hover:shadow-lg hover:shadow-slate-900/50">
                            <div className="flex justify-between items-start mb-3">
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-gradient-to-br from-slate-700 to-slate-800 rounded-lg flex items-center justify-center font-bold text-white">
                                  {i + 1}
                                </div>
                                <div>
                                  <div className="font-bold text-white text-base">{item.assetId || item.name}</div>
                                  {(item.city || item.region) && (
                                    <div className="flex items-center gap-1 mt-1">
                                      <Globe className="w-3 h-3 text-slate-400" />
                                      <span className="text-xs text-slate-400">{item.city || item.region}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                              <div className="flex flex-col items-end gap-2">
                                {item.score && (
                                  <div className={`text-2xl font-bold ${
                                    item.score >= 80 ? 'text-red-400' :
                                    item.score >= 60 ? 'text-amber-400' :
                                    'text-green-400'
                                  }`}>
                                    {item.score}
                                  </div>
                                )}
                                {(item.severity || item.risk) && (
                                  <span className={`text-xs px-3 py-1.5 rounded-full font-semibold ${
                                    (item.severity === 'high' || item.risk === 'High') ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                                    (item.severity === 'medium' || item.risk === 'Medium') ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                                    'bg-green-500/20 text-green-300 border border-green-500/30'
                                  }`}>
                                    {item.severity?.toUpperCase() || item.risk} RISK
                                  </span>
                                )}
                              </div>
                            </div>
                            
                            {item.issue && (
                              <div className="bg-slate-900/50 rounded-lg p-3 mb-3 border border-slate-700/30">
                                <div className="text-sm text-slate-300 leading-relaxed">{item.issue}</div>
                              </div>
                            )}
                            
                            <div className="grid grid-cols-2 gap-3 text-xs">
                              {item.impact && (
                                <div className="bg-slate-900/30 rounded-lg px-3 py-2 border border-slate-700/20">
                                  <div className="text-slate-400 mb-1">Est. Impact</div>
                                  <div className="text-white font-semibold">{item.impact}</div>
                                </div>
                              )}
                              {item.lastUpdated && (
                                <div className="bg-slate-900/30 rounded-lg px-3 py-2 border border-slate-700/20">
                                  <div className="text-slate-400 mb-1">Last Updated</div>
                                  <div className="text-white font-semibold">{new Date(item.lastUpdated).toLocaleDateString()}</div>
                                </div>
                              )}
                            </div>
                            
                            {item.reasons && item.reasons.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-slate-700/30">
                                <div className="text-xs text-slate-400 mb-2 font-semibold">Contributing Factors:</div>
                                <div className="space-y-2">
                                  {item.reasons.map((reason, ri) => (
                                    <div key={ri} className="flex items-center gap-2">
                                      <div className="flex-1 bg-slate-900/50 rounded-full h-2 overflow-hidden">
                                        <div 
                                          className={`h-full rounded-full ${
                                            reason.contribution >= 30 ? 'bg-gradient-to-r from-red-500 to-red-600' :
                                            reason.contribution >= 20 ? 'bg-gradient-to-r from-amber-500 to-amber-600' :
                                            'bg-gradient-to-r from-blue-500 to-blue-600'
                                          }`}
                                          style={{ width: `${reason.contribution}%` }}
                                        ></div>
                                      </div>
                                      <span className="text-xs text-slate-300 w-32 truncate">{reason.feature}</span>
                                      <span className="text-xs text-slate-400 font-mono w-12 text-right">{reason.contribution}%</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* Battery Data Cards */}
                    {msg.batteryData && msg.batteryData.length > 0 && (
                      <div className="mt-5 space-y-3">
                        <div className="flex items-center gap-2 mb-3">
                          <Battery className="w-4 h-4 text-blue-400" />
                          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Battery Performance ({msg.batteryData.length} devices)</span>
                        </div>
                        {msg.batteryData.map((battery, i) => (
                          <div key={i} className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-xl p-4 border border-slate-700/50 hover:border-slate-600 transition-all hover:shadow-lg hover:shadow-slate-900/50">
                            <div className="flex justify-between items-start mb-4">
                              <div className="flex items-center gap-3">
                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                                  battery.health === 'Excellent' ? 'bg-gradient-to-br from-green-500 to-green-600' :
                                  battery.health === 'Good' ? 'bg-gradient-to-br from-blue-500 to-blue-600' :
                                  battery.health === 'Warning' ? 'bg-gradient-to-br from-amber-500 to-amber-600' :
                                  'bg-gradient-to-br from-red-500 to-red-600'
                                } shadow-lg`}>
                                  <Battery className="w-6 h-6 text-white" />
                                </div>
                                <div>
                                  <div className="font-bold text-white text-base">{battery.device || battery.deviceId}</div>
                                  {(battery.city || battery.region) && (
                                    <div className="flex items-center gap-1 mt-1">
                                      <Globe className="w-3 h-3 text-slate-400" />
                                      <span className="text-xs text-slate-400">{battery.city || battery.region}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                              <span className={`text-xs px-3 py-1.5 rounded-full font-semibold ${
                                battery.health === 'Excellent' ? 'bg-green-500/20 text-green-300 border border-green-500/30' :
                                battery.health === 'Good' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' :
                                battery.health === 'Warning' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                                'bg-red-500/20 text-red-300 border border-red-500/30'
                              }`}>
                                {battery.health}
                              </span>
                            </div>
                            
                            {/* Capacity Progress Bar */}
                            <div className="mb-4">
                              <div className="flex justify-between items-center mb-2">
                                <span className="text-xs text-slate-400 font-semibold">Battery Capacity</span>
                                <span className="text-sm text-white font-bold">{battery.capacity}%</span>
                              </div>
                              <div className="w-full bg-slate-900/50 rounded-full h-3 overflow-hidden border border-slate-700/30">
                                <div 
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    battery.capacity >= 80 ? 'bg-gradient-to-r from-green-500 to-green-600' :
                                    battery.capacity >= 50 ? 'bg-gradient-to-r from-blue-500 to-blue-600' :
                                    battery.capacity >= 30 ? 'bg-gradient-to-r from-amber-500 to-amber-600' :
                                    'bg-gradient-to-r from-red-500 to-red-600'
                                  }`}
                                  style={{ width: `${battery.capacity}%` }}
                                ></div>
                              </div>
                            </div>
                            
                            {/* Metrics Grid */}
                            <div className="grid grid-cols-2 gap-3 mb-3">
                              <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/30">
                                <div className="flex items-center gap-2 mb-1">
                                  <Zap className="w-3 h-3 text-yellow-400" />
                                  <span className="text-xs text-slate-400">Voltage</span>
                                  {battery.voltageZScore !== undefined && (
                                    <span className={`text-xs px-1.5 py-0.5 rounded font-semibold ${
                                      Math.abs(battery.voltageZScore) > 2 ? 'bg-red-500/20 text-red-300' :
                                      Math.abs(battery.voltageZScore) > 1 ? 'bg-amber-500/20 text-amber-300' :
                                      'bg-green-500/20 text-green-300'
                                    }`}>Z:{battery.voltageZScore.toFixed(2)}</span>
                                  )}
                                </div>
                                <div className="text-lg font-bold text-white">{battery.voltage}V</div>
                              </div>
                              <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/30">
                                <div className="flex items-center gap-2 mb-1">
                                  <Thermometer className="w-3 h-3 text-red-400" />
                                  <span className="text-xs text-slate-400">Temperature</span>
                                </div>
                                <div className={`text-lg font-bold ${
                                  battery.temperature > 35 ? 'text-red-400' :
                                  battery.temperature > 25 ? 'text-amber-400' :
                                  'text-green-400'
                                }`}>{battery.temperature}°C</div>
                              </div>
                              <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/30">
                                <div className="flex items-center gap-2 mb-1">
                                  <Activity className="w-3 h-3 text-cyan-400" />
                                  <span className="text-xs text-slate-400">Capacity</span>
                                  {battery.capacityZScore !== undefined && (
                                    <span className={`text-xs px-1.5 py-0.5 rounded font-semibold ${
                                      Math.abs(battery.capacityZScore) > 2 ? 'bg-red-500/20 text-red-300' :
                                      Math.abs(battery.capacityZScore) > 1 ? 'bg-amber-500/20 text-amber-300' :
                                      'bg-green-500/20 text-green-300'
                                    }`}>Z:{battery.capacityZScore.toFixed(2)}</span>
                                  )}
                                </div>
                                <div className="text-lg font-bold text-white">{battery.capacity}%</div>
                              </div>
                              <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/30">
                                <div className="flex items-center gap-2 mb-1">
                                  <Activity className="w-3 h-3 text-cyan-400" />
                                  <span className="text-xs text-slate-400">Charge Cycles</span>
                                </div>
                                <div className="text-lg font-bold text-white">{battery.cycles || battery.chargeCycles}</div>
                              </div>
                              <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/30">
                                <div className="flex items-center gap-2 mb-1">
                                  <TrendingUp className="w-3 h-3 text-purple-400" />
                                  <span className="text-xs text-slate-400">Predicted Life</span>
                                </div>
                                <div className="text-lg font-bold text-white">{battery.predictedLife}</div>
                              </div>
                            </div>
                            
                            {/* Z-Score Anomaly Alerts */}
                            {battery.alerts && battery.alerts.length > 0 && (
                              <div className="mt-3 space-y-2">
                                {battery.alerts.map((alert, alertIdx) => (
                                  <div key={alertIdx} className={`p-3 rounded-lg border ${
                                    alert.severity === 'critical' ? 'bg-red-500/10 border-red-500/30' :
                                    alert.severity === 'warning' ? 'bg-amber-500/10 border-amber-500/30' :
                                    'bg-blue-500/10 border-blue-500/30'
                                  }`}>
                                    <div className="flex items-start gap-2">
                                      <AlertTriangle className={`w-4 h-4 mt-0.5 ${
                                        alert.severity === 'critical' ? 'text-red-400' :
                                        alert.severity === 'warning' ? 'text-amber-400' :
                                        'text-blue-400'
                                      }`} />
                                      <div className="flex-1">
                                        <div className={`text-xs font-semibold mb-1 ${
                                          alert.severity === 'critical' ? 'text-red-300' :
                                          alert.severity === 'warning' ? 'text-amber-300' :
                                          'text-blue-300'
                                        }`}>
                                          {alert.message}
                                        </div>
                                        {alert.recommendation && (
                                          <div className="text-xs text-slate-400">{alert.recommendation}</div>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                            
                            {(battery.health === 'Critical' || battery.health === 'Warning') && (!battery.alerts || battery.alerts.length === 0) && (
                              <div className={`mt-3 p-3 rounded-lg border ${
                                battery.health === 'Critical' 
                                  ? 'bg-red-500/10 border-red-500/30' 
                                  : 'bg-amber-500/10 border-amber-500/30'
                              }`}>
                                <div className="flex items-start gap-2">
                                  <AlertTriangle className={`w-4 h-4 mt-0.5 ${
                                    battery.health === 'Critical' ? 'text-red-400' : 'text-amber-400'
                                  }`} />
                                  <span className={`text-xs ${
                                    battery.health === 'Critical' ? 'text-red-300' : 'text-amber-300'
                                  }`}>
                                    {battery.health === 'Critical' 
                                      ? '⚠️ Immediate replacement required' 
                                      : '⚠️ Schedule preventive maintenance'}
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* Recommendations Section */}
                    {msg.recommendations && msg.recommendations.length > 0 && (
                      <div className="mt-5 bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-xl p-4 border border-blue-500/30">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                            <TrendingUp className="w-4 h-4 text-white" />
                          </div>
                          <span className="text-sm font-bold text-blue-300">Recommendations</span>
                        </div>
                        <ul className="space-y-2">
                          {msg.recommendations.map((rec, i) => (
                            <li key={i} className="flex items-start gap-3 text-sm text-slate-300 leading-relaxed">
                              <div className="w-5 h-5 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                <span className="text-xs font-bold text-blue-400">{i + 1}</span>
                              </div>
                              <span>{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {/* Summary Section */}
                    {msg.summary && (
                      <div className="mt-5 bg-slate-900/50 rounded-xl p-4 border border-slate-700/50">
                        <div className="flex items-center gap-2 mb-3">
                          <Package className="w-4 h-4 text-slate-400" />
                          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Detailed Summary</span>
                        </div>
                        <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">
                          {msg.summary}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {/* Loading State */}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-slate-800/80 rounded-2xl p-5 border border-slate-700 shadow-xl">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm text-slate-400">Analyzing your query...</span>
                  </div>
                </div>
              </div>
            )}
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