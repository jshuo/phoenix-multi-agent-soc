import React, { useState, useEffect } from 'react';
import { Activity, Truck, Database, Brain, AlertTriangle, Users, Settings, TrendingUp, Thermometer, MapPin, Battery, Zap } from 'lucide-react';

const AIoTDigitalTwin = () => {
  const [activeLayer, setActiveLayer] = useState('overview');
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [telemetryData, setTelemetryData] = useState({
    temperature: 22,
    pressure: 101.3,
    battery: 87,
    routeDeviation: 0.5,
    speedSpike: 0,
    anomalyScore: 0.12
  });
  const [selectedCargo, setSelectedCargo] = useState('perishable');
  const [forwarderQuality, setForwarderQuality] = useState('HIGH');
  const [currentAction, setCurrentAction] = useState('monitor');
  const [alertCount, setAlertCount] = useState(0);

  const cargoTypes = {
    perishable: { vector: [1,0,0,0], color: 'bg-green-500', threshold: 'strict' },
    electronics: { vector: [0,1,0,0], color: 'bg-blue-500', threshold: 'tight' },
    hazardous: { vector: [0,0,1,0], color: 'bg-red-500', threshold: 'critical' },
    bulk: { vector: [0,0,0,1], color: 'bg-yellow-500', threshold: 'relaxed' }
  };

  const qualityLevels = {
    UNK: { vector: [1,0,0,0], color: 'bg-gray-500' },
    LOW: { vector: [0,1,0,0], color: 'bg-red-500' },
    MED: { vector: [0,0,1,0], color: 'bg-yellow-500' },
    HIGH: { vector: [0,0,0,1], color: 'bg-green-500' }
  };

  const actions = ['monitor', 'increase_sampling', 'calibrate', 'peer_check', 'escalate', 'flag'];

  useEffect(() => {
    if (simulationRunning) {
      const interval = setInterval(() => {
        setTelemetryData(prev => ({
          temperature: Math.max(15, Math.min(35, prev.temperature + (Math.random() - 0.5) * 2)),
          pressure: Math.max(95, Math.min(105, prev.pressure + (Math.random() - 0.5) * 0.5)),
          battery: Math.max(0, prev.battery - Math.random() * 0.1),
          routeDeviation: Math.max(0, prev.routeDeviation + (Math.random() - 0.6) * 0.3),
          speedSpike: Math.random() > 0.9 ? Math.random() * 10 : 0,
          anomalyScore: Math.random()
        }));

        // Decision logic simulation
        if (telemetryData.anomalyScore > 0.7) {
          setCurrentAction('escalate');
          setAlertCount(prev => prev + 1);
        } else if (telemetryData.anomalyScore > 0.5) {
          setCurrentAction('peer_check');
        } else if (telemetryData.temperature > 28 && selectedCargo === 'perishable') {
          setCurrentAction('increase_sampling');
        } else {
          setCurrentAction('monitor');
        }
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [simulationRunning, telemetryData.anomalyScore, telemetryData.temperature, selectedCargo]);

  const renderOverview = () => (
    <div className="grid grid-cols-3 gap-4">
      <div className="col-span-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
        <h2 className="text-2xl font-bold mb-2">Arviem-ITracXing MaaS AIoT Multi-Agent Logistics Monitoring Platform</h2>
        <p className="text-sm opacity-90">Supply Chain Monitoring as a Service (MaaS)</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
        <div className="flex items-center mb-3">
          <Database className="w-8 h-8 text-blue-500 mr-3" />
          <h3 className="text-lg font-semibold">Data Ingestion</h3>
        </div>
        <p className="text-sm text-gray-600 mb-2">IoT Sensors + Kalman Filters</p>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between"><span>Temperature:</span><span className="font-mono">{telemetryData.temperature.toFixed(1)}°C</span></div>
          <div className="flex justify-between"><span>Pressure:</span><span className="font-mono">{telemetryData.pressure.toFixed(1)} kPa</span></div>
          <div className="flex justify-between"><span>Battery:</span><span className="font-mono">{telemetryData.battery.toFixed(0)}%</span></div>
          <div className="flex justify-between"><span>Route Dev:</span><span className="font-mono">{telemetryData.routeDeviation.toFixed(2)} km</span></div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-purple-500">
        <div className="flex items-center mb-3">
          <Brain className="w-8 h-8 text-purple-500 mr-3" />
          <h3 className="text-lg font-semibold">ML Analysis</h3>
        </div>
        <p className="text-sm text-gray-600 mb-2">Feature Engineering + Anomaly Detection</p>
        <div className="mt-4">
          <div className="flex justify-between text-xs mb-1">
            <span>Anomaly Score</span>
            <span className="font-mono">{telemetryData.anomalyScore.toFixed(2)}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all ${telemetryData.anomalyScore > 0.7 ? 'bg-red-500' : telemetryData.anomalyScore > 0.4 ? 'bg-yellow-500' : 'bg-green-500'}`}
              style={{width: `${telemetryData.anomalyScore * 100}%`}}
            ></div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-500">
        <div className="flex items-center mb-3">
          <Zap className="w-8 h-8 text-green-500 mr-3" />
          <h3 className="text-lg font-semibold">RL Decision Engine</h3>
        </div>
        <p className="text-sm text-gray-600 mb-2">Action Selection + Safety Overrides</p>
        <div className="mt-4">
          <div className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
            currentAction === 'escalate' ? 'bg-red-100 text-red-700' :
            currentAction === 'peer_check' ? 'bg-yellow-100 text-yellow-700' :
            currentAction === 'increase_sampling' ? 'bg-blue-100 text-blue-700' :
            'bg-green-100 text-green-700'
          }`}>
            {currentAction.toUpperCase().replace('_', ' ')}
          </div>
        </div>
      </div>

      <div className="col-span-3 bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Context Integration</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm font-medium mb-2">Cargo Type (One-Hot Encoded)</p>
            <div className="flex gap-2">
              {Object.keys(cargoTypes).map(type => (
                <button
                  key={type}
                  onClick={() => setSelectedCargo(type)}
                  className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
                    selectedCargo === type 
                      ? `${cargoTypes[type].color} text-white` 
                      : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  {type.toUpperCase()}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2 font-mono">
              [{cargoTypes[selectedCargo].vector.join(', ')}]
            </p>
          </div>
          <div>
            <p className="text-sm font-medium mb-2">Forwarder Quality</p>
            <div className="flex gap-2">
              {Object.keys(qualityLevels).map(level => (
                <button
                  key={level}
                  onClick={() => setForwarderQuality(level)}
                  className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
                    forwarderQuality === level 
                      ? `${qualityLevels[level].color} text-white` 
                      : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2 font-mono">
              [{qualityLevels[forwarderQuality].vector.join(', ')}]
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDataFlow = () => (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold mb-4">Data Flow Pipeline</h3>
        <div className="relative">
          {/* Flow stages */}
          {['Ingestion', 'Analysis', 'Decision', 'Execution', 'Output', 'Learning'].map((stage, idx) => (
            <div key={stage} className="mb-6">
              <div className="flex items-center">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white ${
                  idx === 0 ? 'bg-blue-500' :
                  idx === 1 ? 'bg-purple-500' :
                  idx === 2 ? 'bg-green-500' :
                  idx === 3 ? 'bg-orange-500' :
                  idx === 4 ? 'bg-pink-500' :
                  'bg-indigo-500'
                }`}>
                  {idx + 1}
                </div>
                <div className="ml-4 flex-1">
                  <h4 className="font-semibold text-lg">{stage}</h4>
                  <p className="text-sm text-gray-600">
                    {idx === 0 && 'IoT sensors → Kalman filters process signals'}
                    {idx === 1 && 'Feature engineering → ML anomaly detection → Context integration'}
                    {idx === 2 && 'RL decision engine selects appropriate actions with safety overrides'}
                    {idx === 3 && 'Multi-agent system executes actions and coordinates responses'}
                    {idx === 4 && 'Results displayed on Operations Dashboard'}
                    {idx === 5 && 'System continuously learns from actions to improve decisions'}
                  </p>
                </div>
              </div>
              {idx < 5 && (
                <div className="ml-6 mt-2 mb-2 border-l-2 border-gray-300 h-8"></div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderPeerReview = () => (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold mb-4">Peer Review Decision Flows</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border-2 border-green-500 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center text-white font-bold">✓</div>
              <h4 className="ml-3 font-semibold">Review OK</h4>
            </div>
            <p className="text-sm text-gray-600 mb-2">False positive or resolved issue</p>
            <ul className="text-xs space-y-1 text-gray-700">
              <li>→ Monitor Action</li>
              <li>→ Trust score increases</li>
              <li>→ Labeled "verified_normal"</li>
              <li>→ Status update to dashboard</li>
            </ul>
          </div>

          <div className="border-2 border-red-500 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <div className="w-10 h-10 bg-red-500 rounded-full flex items-center justify-center text-white font-bold">!</div>
              <h4 className="ml-3 font-semibold">Review NOT OK</h4>
            </div>
            <p className="text-sm text-gray-600 mb-2">Genuine anomaly confirmed</p>
            <ul className="text-xs space-y-1 text-gray-700">
              <li>→ Escalate Action</li>
              <li>→ High-priority alert</li>
              <li>→ Notify operators</li>
              <li>→ Labeled "verified_anomaly"</li>
            </ul>
          </div>

          <div className="border-2 border-yellow-500 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <div className="w-10 h-10 bg-yellow-500 rounded-full flex items-center justify-center text-white font-bold">?</div>
              <h4 className="ml-3 font-semibold">Review Uncertain</h4>
            </div>
            <p className="text-sm text-gray-600 mb-2">Need more data for assessment</p>
            <ul className="text-xs space-y-1 text-gray-700">
              <li>→ Increase Sampling Action</li>
              <li>→ Boost data frequency</li>
              <li>→ Set timeout with defaults</li>
              <li>→ Route to escalation queue</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  const renderMetrics = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Telemetry Metrics</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center"><Thermometer className="w-4 h-4 mr-2"/>Temperature</span>
                <span className="font-mono">{telemetryData.temperature.toFixed(1)}°C</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full" style={{width: `${(telemetryData.temperature/40)*100}%`}}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center"><MapPin className="w-4 h-4 mr-2"/>Route Deviation</span>
                <span className="font-mono">{telemetryData.routeDeviation.toFixed(2)} km</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className={`h-2 rounded-full ${telemetryData.routeDeviation > 2 ? 'bg-red-500' : 'bg-green-500'}`} style={{width: `${Math.min((telemetryData.routeDeviation/5)*100, 100)}%`}}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="flex items-center"><Battery className="w-4 h-4 mr-2"/>Battery</span>
                <span className="font-mono">{telemetryData.battery.toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className={`h-2 rounded-full ${telemetryData.battery < 20 ? 'bg-red-500' : 'bg-green-500'}`} style={{width: `${telemetryData.battery}%`}}></div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">System Status</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm">Alert Count</span>
              <span className="text-2xl font-bold text-red-600">{alertCount}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Current Action</span>
              <span className="text-xs font-mono bg-gray-100 px-2 py-1 rounded">{currentAction}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Cargo Type</span>
              <span className={`text-xs font-semibold px-2 py-1 rounded text-white ${cargoTypes[selectedCargo].color}`}>
                {selectedCargo.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Forwarder Quality</span>
              <span className={`text-xs font-semibold px-2 py-1 rounded text-white ${qualityLevels[forwarderQuality].color}`}>
                {forwarderQuality}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Available Actions</h3>
        <div className="grid grid-cols-3 gap-3">
          {actions.map(action => (
            <div 
              key={action}
              className={`p-3 rounded-lg border-2 text-center text-sm font-semibold transition-all ${
                currentAction === action 
                  ? 'border-blue-500 bg-blue-50 text-blue-700' 
                  : 'border-gray-200 bg-gray-50 text-gray-600'
              }`}
            >
              {action.replace('_', ' ').toUpperCase()}
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Arviem-ITracXing MaaS AIoT Digital Twin</h1>
            <p className="text-gray-600">Arviem-ITracXing MaaS Multi-Agent Logistics Monitoring Platform</p>
          </div>
          <button
            onClick={() => setSimulationRunning(!simulationRunning)}
            className={`px-6 py-3 rounded-lg font-semibold flex items-center gap-2 transition-all ${
              simulationRunning 
                ? 'bg-red-500 hover:bg-red-600 text-white' 
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            <Activity className={simulationRunning ? 'animate-pulse' : ''} />
            {simulationRunning ? 'Stop Simulation' : 'Start Simulation'}
          </button>
        </div>

        {/* Navigation */}
        <div className="mb-6 bg-white rounded-lg shadow-lg p-2 flex gap-2">
          {[
            { id: 'overview', icon: Activity, label: 'System Overview' },
            { id: 'dataflow', icon: TrendingUp, label: 'Data Flow' },
            { id: 'peer', icon: Users, label: 'Peer Review' },
            { id: 'metrics', icon: Settings, label: 'Metrics & Actions' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveLayer(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-all ${
                activeLayer === tab.id
                  ? 'bg-blue-500 text-white shadow-md'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <tab.icon className="w-5 h-5" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="transition-all">
          {activeLayer === 'overview' && renderOverview()}
          {activeLayer === 'dataflow' && renderDataFlow()}
          {activeLayer === 'peer' && renderPeerReview()}
          {activeLayer === 'metrics' && renderMetrics()}
        </div>
      </div>
    </div>
  );
};

export default AIoTDigitalTwin;
