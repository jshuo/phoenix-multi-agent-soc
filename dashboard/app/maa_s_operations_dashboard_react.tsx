'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Activity, AlertTriangle, Bell, Brain, Filter, Truck, TrendingDown, Zap, Thermometer, MapPin, Battery, Database, Users, Settings, Settings2, ShieldCheck, PackageSearch, Globe, GaugeCircle, TrendingUp } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, AreaChart, Area } from 'recharts';
import GeolocationMap from '../components/GeolocationMap';

const CARGO = ["PERISH", "ELECT", "HAZ", "BULK", "FRAG"];
const FWDQ = ["UNK", "LOW", "MED", "HIGH"];
const IF_BUCKET = ["NORMAL", "WARN", "ANOMALOUS"];
const SEVERITY = ["LOW", "MED", "HIGH"];
const ACTIONS = ["monitor", "increase_sampling", "calibrate", "peer_check", "escalate"];
const lanes = ["TPE→LAX", "TPE→AMS", "TPE→NRT", "AMS→JFK", "NRT→FRA", "SIN→SYD"];
const forwarders = [
  { id: "DGF", name: "DHL Global Forwarding" },
  { id: "K+N", name: "Kuehne+Nagel" },
  { id: "DBS", name: "DB Schenker" },
  { id: "EXP", name: "Expeditors" }
];

const COLORS = { primary: "#8b5cf6", emerald: "#10b981", blue: "#3b82f6", amber: "#f59e0b", rose: "#f43f5e" };
const PIE_COLORS = ["#64748b", "#ef4444", "#f59e0b", "#10b981"];

function nowISO() {
  return new Date().toISOString().slice(0, 19).replace("T", " ");
}

function badgeForSeverity(sev) {
  const map = { LOW: "bg-emerald-100 text-emerald-700", MED: "bg-amber-100 text-amber-700", HIGH: "bg-rose-100 text-rose-700" };
  return map[sev] || "bg-slate-100 text-slate-700";
}

function badgeForIFBucket(b) {
  const map = { NORMAL: "bg-slate-100 text-slate-700", WARN: "bg-amber-100 text-amber-700", ANOMALOUS: "bg-rose-100 text-rose-700" };
  return map[b] || "bg-slate-100 text-slate-700";
}

function badgeForFwdq(b) {
  const map = { UNK: "bg-slate-100 text-slate-700", LOW: "bg-rose-100 text-rose-700", MED: "bg-amber-100 text-amber-700", HIGH: "bg-emerald-100 text-emerald-700" };
  return map[b] || "bg-slate-100 text-slate-700";
}

const timeseries = Array.from({ length: 24 }).map((_, i) => ({
  t: `${i}:00`,
  anomalies: Math.max(0, Math.round(8 + 6 * Math.sin(i / 3) + (Math.random() * 4 - 2))),
  alerts: Math.max(0, Math.round(4 + 3 * Math.cos(i / 4) + (Math.random() * 3 - 1.5))),
  mttr: Math.max(10, Math.round(70 - 15 * Math.sin(i / 5) + (Math.random() * 10 - 5)))
}));

const fwdqDistribution = [
  { name: "UNK", value: 8 },
  { name: "LOW", value: 14 },
  { name: "MED", value: 46 },
  { name: "HIGH", value: 22 }
];

const cargoAlerts = CARGO.map((c) => ({ cargo: c, count: Math.round(5 + Math.random() * 20) }));

function makeAlert(id: number) {
  // Use seeded random for consistent server/client rendering
  const seed = id * 1000;
  const seededRandom = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };
  
  const cargo = CARGO[Math.floor(seededRandom(seed + 1) * CARGO.length)];
  const lane = lanes[Math.floor(seededRandom(seed + 2) * lanes.length)];
  const fwd = forwarders[Math.floor(seededRandom(seed + 3) * forwarders.length)];
  const ifb = IF_BUCKET[Math.floor(seededRandom(seed + 4) * IF_BUCKET.length)];
  const sev = SEVERITY[Math.floor(seededRandom(seed + 5) * SEVERITY.length)];
  const fwdq = FWDQ[Math.floor(seededRandom(seed + 6) * FWDQ.length)];
  const rl = ACTIONS[Math.floor(seededRandom(seed + 7) * ACTIONS.length)];
  const reason = fwdq === "LOW" && (sev === "MED" || sev === "HIGH") ? "policy: low fwdq tilts escalate" : "rl: expected cost minimal";
  
  return {
    id: `EV-${1000 + id}`,
    time: nowISO(),
    tracker: `TRK${Math.floor(100000 + seededRandom(seed + 8) * 900000)}`,
    cargo,
    lane,
    forwarder: fwd,
    fwdq,
    if_bucket: ifb,
    severity: sev,
    suggested: rl,
    policy_reason: reason,
    temp_sla_violation_min: Math.floor(seededRandom(seed + 9) * 60)
  };
}

const AIoTDigitalTwin = () => {
  const [viewMode, setViewMode] = useState('dashboard');
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [telemetryData, setTelemetryData] = useState({
    temperature: 22,
    pressure: 101.3,
    battery: 87,
    routeDeviation: 0.5,
    speedSpike: 0,
    anomalyScore: 0.12
  });
  
  const [customer, setCustomer] = useState("Taiwan Pharma Export");
  const [range, setRange] = useState("24h");
  const [cargo, setCargo] = useState("ALL");
  const [lane, setLane] = useState("ALL");
  const [fwdq, setFwdq] = useState("ALL");
  const [severity, setSeverity] = useState("ALL");
  const [alerts, setAlerts] = useState<any[]>([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [activeTab, setActiveTab] = useState("all");
  const [currentTime, setCurrentTime] = useState("");
  const [archTab, setArchTab] = useState('overview');

  // Initialize client-side only data
  useEffect(() => {
    setIsClient(true);
    setCurrentTime(nowISO());
    setAlerts(Array.from({ length: 14 }).map((_, i) => makeAlert(i)));
  }, []);

  useEffect(() => {
    if (!isClient) return;
    
    const timer = setInterval(() => setCurrentTime(nowISO()), 30000);
    return () => clearInterval(timer);
  }, [isClient]);

  useEffect(() => {
    if (!isClient || !simulationRunning) return;
    
    const interval = setInterval(() => {
      setTelemetryData(prev => ({
        temperature: Math.max(15, Math.min(35, prev.temperature + (Math.random() - 0.5) * 2)),
        pressure: Math.max(95, Math.min(105, prev.pressure + (Math.random() - 0.5) * 0.5)),
        battery: Math.max(0, prev.battery - Math.random() * 0.1),
        routeDeviation: Math.max(0, prev.routeDeviation + (Math.random() - 0.6) * 0.3),
        speedSpike: Math.random() > 0.9 ? Math.random() * 10 : 0,
        anomalyScore: Math.random()
      }));

      if (Math.random() > 0.85) {
        setAlerts(prev => [makeAlert(Date.now()), ...prev.slice(0, 19)]);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [simulationRunning, isClient]);

  const filtered = useMemo(() => {
    let result = alerts.filter((a) =>
      (cargo === "ALL" || a.cargo === cargo) &&
      (lane === "ALL" || a.lane === lane) &&
      (fwdq === "ALL" || a.fwdq === fwdq) &&
      (severity === "ALL" || a.severity === severity) &&
      (query === "" || `${a.id} ${a.tracker} ${a.forwarder.name}`.toLowerCase().includes(query.toLowerCase()))
    );
    
    if (activeTab === "anomalous") {
      result = result.filter(a => a.if_bucket === "ANOMALOUS");
    } else if (activeTab === "needs-review") {
      result = result.filter(a => a.suggested === "peer_check" || a.severity === "HIGH");
    }
    
    return result;
  }, [alerts, cargo, lane, fwdq, severity, query, activeTab]);

  const kpis = useMemo(() => ({
    active: 1242,
    open: filtered.length,
    onTime: 0.92,
    mtta: 14,
    excursion: 38,
    falseAlarmReduction: 30,
    opexReduction: 20
  }), [filtered.length]);

  function resolve(id) {
    setAlerts(prev => prev.filter(x => x.id !== id));
    setSelected(null);
  }

  const renderArchitecture = () => (
    <div className="space-y-4">
      <div className="flex gap-2 mb-4">
        {['overview', 'dataflow', 'peer'].map(tab => (
          <button
            key={tab}
            onClick={() => setArchTab(tab)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              archTab === tab ? 'bg-violet-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            {tab === 'overview' ? 'System Overview' : tab === 'dataflow' ? 'Data Flow' : 'Peer Review'}
          </button>
        ))}
      </div>

      {archTab === 'overview' && (
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-3 bg-gradient-to-r from-violet-600 to-purple-600 rounded-lg p-6 text-white">
            <h2 className="text-2xl font-bold mb-2">Arviem-ITracXing AIoT Multi-Agent Logistics Monitoring Platform</h2>
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
            <div className="mt-4 space-y-2">
              <div className="text-xs text-gray-500">Live Actions from Dashboard:</div>
              <div className="flex flex-wrap gap-1">
                {ACTIONS.slice(0, 3).map(action => (
                  <span key={action} className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs">{action}</span>
                ))}
              </div>
            </div>
          </div>

          <div className="col-span-3 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Context Integration Layer</h3>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-sm font-medium mb-3">Cargo Types (One-Hot Encoded)</p>
                <div className="space-y-2">
                  {CARGO.map(type => (
                    <div key={type} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span className="text-sm font-medium">{type}</span>
                      <span className="text-xs font-mono text-gray-500">Vector encoding</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-sm font-medium mb-3">Forwarder Quality Levels</p>
                <div className="space-y-2">
                  {FWDQ.map(level => (
                    <div key={level} className="flex items-center justify-between p-2 rounded" style={{backgroundColor: badgeForFwdq(level).includes('emerald') ? '#d1fae5' : badgeForFwdq(level).includes('rose') ? '#fee2e2' : badgeForFwdq(level).includes('amber') ? '#fef3c7' : '#f1f5f9'}}>
                      <span className="text-sm font-medium">{level}</span>
                      <span className="text-xs font-mono">Quality: {['Unknown', 'Below avg', 'Standard', 'Excellent'][FWDQ.indexOf(level)]}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {archTab === 'dataflow' && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold mb-6">Data Flow Pipeline</h3>
          <div className="relative">
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
      )}

      {archTab === 'peer' && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold mb-6">Peer Review Decision Flows</h3>
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
      )}
    </div>
  );

  const renderDashboard = () => (
    <div className="space-y-4">
      {/* Top Navigation/Filters - Simplified */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <select value={range} onChange={(e) => setRange(e.target.value)} className="px-3 py-2 border border-gray-300 rounded-lg text-sm">
              <option value="2h">Last 2h</option>
              <option value="24h">Last 24h</option>
              <option value="7d">Last 7d</option>
            </select>
            <select value={cargo} onChange={(e) => setCargo(e.target.value)} className="px-3 py-2 border border-gray-300 rounded-lg text-sm">
              <option value="ALL">All Cargo</option>
              {CARGO.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="text-sm text-gray-600">
            {customer} • {isClient ? currentTime.slice(11, 16) : '--:--'}
          </div>
        </div>
      </div>

      {/* Main Dashboard Layout - 3 Column Grid */}
      <div className="grid grid-cols-12 gap-4">
        {/* Left Panel - Fleet Overview */}
        <div className="col-span-3 space-y-4">
          <div className="bg-gray-800 text-white rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Fleet Overview</h2>
            
            <div className="space-y-4">
              <div>
                <div className="text-sm text-gray-300">Total Active Shipments</div>
                <div className="text-3xl font-bold">1,245</div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-300">At Risk</div>
                  <div className="text-xl font-semibold">76 (6.3%)</div>
                </div>
                <div>
                  <div className="text-gray-300">On Time Delivery Rate</div>
                  <div className="text-xl font-semibold text-green-400">94.2%</div>
                </div>
              </div>
              
              <div>
                <div className="text-sm text-gray-300">Carbon Footprint</div>
                <div className="text-lg font-semibold">0,82 <span className="text-sm">tCO₂e</span></div>
                <div className="text-xs text-gray-400">0,82 t</div>
              </div>
            </div>
          </div>

          {/* Cold-Chain KPI */}
          <div className="bg-gray-800 text-white rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Cold-Chain KPI</h3>
            
            <div className="space-y-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-300">% Shipments within SLA (past 7d)</span>
                  <span className="text-sm font-semibold">97%</span>
                </div>
                <div className="text-green-400 text-sm font-medium">07A</div>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
                  <div className="bg-green-400 h-2 rounded-full" style={{width: '97%'}}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">Mean Time to Detect (MTTD) Temp Breach</span>
                  <span className="text-sm font-semibold">3.7%</span>
                </div>
                <div className="text-sm">8 min</div>
              </div>
              
              <div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">False Alarm Rate (Kalman HIF+ RL)</span>
                  <span className="text-sm font-semibold">3.1%</span>
                </div>
                <div className="text-sm">3.1%</div>
              </div>
            </div>
          </div>
        </div>

        {/* Center Panel - World Map */}
        <div className="col-span-6">
          <div className="bg-white rounded-lg shadow-md p-4 h-full">
            <GeolocationMap 
              height="600px"
              selectedShipment={selected?.tracker}
              onShipmentSelect={(shipmentId) => {
                const matchingAlert = alerts.find(alert => 
                  alert.tracker === shipmentId || alert.id === shipmentId
                );
                if (matchingAlert) {
                  setSelected(matchingAlert);
                }
              }}
            />
          </div>
        </div>

        {/* Right Panel - Real-Time Risk & Insights */}
        <div className="col-span-3 space-y-4">
          <div className="bg-gray-800 text-white rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Real-Time Risk</h3>
            
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2 text-xs font-semibold text-gray-300 border-b border-gray-600 pb-2">
                <div>Shipment ID</div>
                <div>Cargo Type</div>
                <div>Issue</div>
              </div>
              
              {[
                { id: 'ARV-99213', cargo: 'Pharma', issue: '0.37' },
                { id: 'ARV-99462', cargo: 'Electronics', issue: '0.46' },
                { id: 'ARV-99311', cargo: 'Food', issue: '0.72' },
                { id: 'ARV-77554', cargo: 'Defense', issue: '0.31' }
              ].map((item, i) => (
                <div key={i} className="grid grid-cols-3 gap-2 text-sm py-1">
                  <div className="text-blue-400">{item.id}</div>
                  <div>{item.cargo}</div>
                  <div className="text-red-400">{item.issue}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Freight Forwarder Metrics */}
          <div className="bg-gray-800 text-white rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Freight Forwarder Metrics</h3>
            
            <div className="space-y-4 text-sm">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-gray-300">DHL Global</div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-400"></div>
                    <span className="font-medium">HIGH (98.2%)</span>
                  </div>
                </div>
                <div>
                  <div className="text-gray-300">Kuehne+Nagel</div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-yellow-400"></div>
                    <span className="font-medium">MED (94.1%)</span>
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-gray-300">DB Schenker</div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-yellow-400"></div>
                    <span className="font-medium">MED (91.7%)</span>
                  </div>
                </div>
                <div>
                  <div className="text-gray-300">Expeditors</div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-red-400"></div>
                    <span className="font-medium">LOW (87.3%)</span>
                  </div>
                </div>
              </div>
              
              <div className="pt-2 border-t border-gray-600">
                <div className="text-xs text-gray-400">
                  Quality Score: On-time delivery, temperature compliance, damage rate
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section - Incident Timeline and Additional Insights */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800 text-white rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Incident Timeline</h3>
          
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-yellow-400"></div>
              <div className="text-sm">
                <span className="text-yellow-400 font-medium">10:22</span>
                <span className="ml-2">GPS drift-detected, ARV-99202</span>
                <div className="text-gray-400">→ Peer, Check triggered</div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-red-400"></div>
              <div className="text-sm">
                <span className="text-red-400 font-medium">09:47</span>
                <span className="ml-2">Temp breach, ARV-99213</span>
                <div className="text-gray-400">→ Escalation ticket sent</div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-green-400"></div>
              <div className="text-sm">
                <span className="text-green-400 font-medium">08:30</span>
                <span className="ml-2">Battery recalibrated, ARV-99311</span>
                <div className="text-gray-400">→ RRV- status back to Normal</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 text-white rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Predictive Insights</h3>
          
          <div className="space-y-4 text-sm">
            <div>
              <div className="font-medium">Forecasted Cold Chain Breaches (next x8 P)</div>
              <div className="text-gray-300">3 (Food lane→ Hamburg)</div>
              <div className="text-gray-300">Pharma lane→ Chicago</div>
            </div>
            
            <div>
              <div className="font-medium">Predicted Port Congestion Risk</div>
              <div className="text-red-400">High at Rotterdam</div>
              <div className="text-gray-300">(ETA delays +12h)</div>
            </div>
            
            <div>
              <div className="font-medium">"What-if" Digital Twin Simulation 24h</div>
              <div className="text-gray-300">28% reduction in spoilage if</div>
              <div className="text-gray-300">sampling rate adapted dynamically</div>
            </div>
            
            <div>
              <div className="font-medium">AI Optimization Recommendations</div>
              <div className="text-green-400">Route TPE→AMS: Switch to DHL (+2.3% efficiency)</div>
              <div className="text-yellow-400">Cold chain: Increase monitoring frequency 15%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="w-full min-h-screen bg-gradient-to-b from-slate-50 to-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-violet-100 rounded-lg">
              <PackageSearch className="h-8 w-8 text-violet-700" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">
                Arviem-ITracXing AIoT Digital Twin - MaaS Platform
              </h1>
              <p className="text-sm text-slate-600">
                Multi-Agent Logistics Monitoring · Taiwan AI Excellence · {isClient ? currentTime : 'Loading...'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 mt-4 md:mt-0">
            <button
              onClick={() => setSimulationRunning(!simulationRunning)}
              className={`px-4 py-2 rounded-lg font-semibold flex items-center gap-2 transition-all ${
                simulationRunning 
                  ? 'bg-red-500 hover:bg-red-600 text-white' 
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              <Activity className={simulationRunning ? 'animate-pulse' : ''} />
              {simulationRunning ? 'Stop Simulation' : 'Start Simulation'}
            </button>
            <button
              onClick={() => setViewMode(viewMode === 'dashboard' ? 'architecture' : 'dashboard')}
              className="px-4 py-2 rounded-lg font-semibold bg-violet-600 hover:bg-violet-700 text-white flex items-center gap-2"
            >
              {viewMode === 'dashboard' ? <Settings className="h-4 w-4"/> : <Activity className="h-4 w-4"/>}
              {viewMode === 'dashboard' ? 'Architecture' : 'Dashboard'}
            </button>
          </div>
        </div>

        {viewMode === 'dashboard' ? renderDashboard() : renderArchitecture()}
      </div>
    </div>
  );
};

export default AIoTDigitalTwin;