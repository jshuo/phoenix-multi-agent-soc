'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Activity, AlertTriangle, Bell, Brain, Filter, Truck, TrendingDown, Zap, Thermometer, MapPin, Battery, Database, Users, Settings, Settings2, ShieldCheck, PackageSearch, Globe, GaugeCircle, TrendingUp } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, AreaChart, Area } from 'recharts';

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

function makeAlert(id) {
  const cargo = CARGO[Math.floor(Math.random() * CARGO.length)];
  const lane = lanes[Math.floor(Math.random() * lanes.length)];
  const fwd = forwarders[Math.floor(Math.random() * forwarders.length)];
  const ifb = IF_BUCKET[Math.floor(Math.random() * IF_BUCKET.length)];
  const sev = SEVERITY[Math.floor(Math.random() * SEVERITY.length)];
  const fwdq = FWDQ[Math.floor(Math.random() * FWDQ.length)];
  const rl = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
  const reason = fwdq === "LOW" && (sev === "MED" || sev === "HIGH") ? "policy: low fwdq tilts escalate" : "rl: expected cost minimal";
  
  return {
    id: `EV-${1000 + id}`,
    time: nowISO(),
    tracker: `TRK${Math.floor(100000 + Math.random() * 900000)}`,
    cargo,
    lane,
    forwarder: fwd,
    fwdq,
    if_bucket: ifb,
    severity: sev,
    suggested: rl,
    policy_reason: reason,
    temp_sla_violation_min: Math.round(Math.random() * 60)
  };
}

const AIoTDigitalTwin = () => {
  const [viewMode, setViewMode] = useState('dashboard');
  const [simulationRunning, setSimulationRunning] = useState(false);
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
  const [alerts, setAlerts] = useState(Array.from({ length: 14 }).map((_, i) => makeAlert(i)));
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [activeTab, setActiveTab] = useState("all");
  const [currentTime, setCurrentTime] = useState(nowISO());
  const [archTab, setArchTab] = useState('overview');

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(nowISO()), 30000);
    return () => clearInterval(timer);
  }, []);

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

        if (Math.random() > 0.85) {
          setAlerts(prev => [makeAlert(Date.now()), ...prev.slice(0, 19)]);
        }
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [simulationRunning]);

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
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="grid grid-cols-6 gap-3">
          <div>
            <label className="text-xs font-medium text-gray-700 block mb-1">Customer</label>
            <input value={customer} onChange={(e) => setCustomer(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm" />
          </div>
          <div>
            <label className="text-xs font-medium text-gray-700 block mb-1">Range</label>
            <select value={range} onChange={(e) => setRange(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm">
              <option value="2h">Last 2h</option>
              <option value="24h">Last 24h</option>
              <option value="7d">Last 7d</option>
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-700 block mb-1">Cargo</label>
            <select value={cargo} onChange={(e) => setCargo(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm">
              <option value="ALL">ALL</option>
              {CARGO.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-700 block mb-1">Lane</label>
            <select value={lane} onChange={(e) => setLane(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm">
              <option value="ALL">ALL</option>
              {lanes.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-700 block mb-1">Fwd Quality</label>
            <select value={fwdq} onChange={(e) => setFwdq(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm">
              <option value="ALL">ALL</option>
              {FWDQ.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-700 block mb-1">Severity</label>
            <select value={severity} onChange={(e) => setSeverity(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm">
              <option value="ALL">ALL</option>
              {SEVERITY.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <Filter className="h-4 w-4 text-gray-500"/>
          <input placeholder="Search alert id / tracker / forwarder" value={query} onChange={e=>setQuery(e.target.value)} className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"/>
        </div>
      </div>

      <div className="grid grid-cols-7 gap-3">
        {[
          { title: "Active Shipments", icon: Truck, value: kpis.active.toLocaleString(), sub: "across all tenants" },
          { title: "Open Alerts", icon: AlertTriangle, value: kpis.open, sub: "filtered view" },
          { title: "On-time", icon: ShieldCheck, value: `${Math.round(kpis.onTime*100)}%`, sub: "milestones" },
          { title: "MTTA", icon: Activity, value: `${kpis.mtta}m`, sub: "acknowledge" },
          { title: "Excursion", icon: Thermometer, value: kpis.excursion, sub: "per 1k" },
          { title: "False Alarm ↓", icon: TrendingDown, value: `-${kpis.falseAlarmReduction}%`, sub: "Taiwan RL", bg: "bg-gradient-to-br from-emerald-50 to-emerald-100" },
          { title: "OPEX ↓", icon: Zap, value: `-${kpis.opexReduction}%`, sub: "Multi-Agent", bg: "bg-gradient-to-br from-blue-50 to-blue-100" }
        ].map((kpi, i) => (
          <div key={i} className={`${kpi.bg || 'bg-white'} rounded-lg shadow-md p-4`}>
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs font-medium text-gray-700">{kpi.title}</div>
              <kpi.icon className="h-4 w-4 text-gray-500"/>
            </div>
            <div className="text-2xl font-bold text-gray-900">{kpi.value}</div>
            <div className="text-xs text-gray-500 mt-1">{kpi.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold">Anomalies & Alerts (24h)</h3>
            <Brain className="h-4 w-4 text-violet-600"/>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeseries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="t" hide />
                <YAxis stroke="#64748b" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="anomalies" stroke={COLORS.primary} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="alerts" stroke={COLORS.rose} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-4">
          <h3 className="text-sm font-semibold mb-3">Forwarder Quality</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={fwdqDistribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} label>
                  {fwdqDistribution.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 bg-white rounded-lg shadow-md">
          <div className="p-4 border-b flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">Live Alerts</h3>
              <span className="px-2 py-1 bg-gray-100 rounded-full text-xs font-medium">{filtered.length}</span>
            </div>
            <div className="flex gap-2">
              {['all', 'anomalous', 'needs-review'].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1 rounded text-xs font-medium ${activeTab === tab ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700'}`}
                >
                  {tab.replace('-', ' ').toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <div className="overflow-auto max-h-96">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-white border-b">
                <tr className="text-left text-gray-600">
                  <th className="py-2 px-2">Time</th>
                  <th className="py-2 px-2">ID</th>
                  <th className="py-2 px-2">Tracker</th>
                  <th className="py-2 px-2">Cargo</th>
                  <th className="py-2 px-2">Lane</th>
                  <th className="py-2 px-2">Fwd</th>
                  <th className="py-2 px-2">FwdQ</th>
                  <th className="py-2 px-2">IF</th>
                  <th className="py-2 px-2">Sev</th>
                  <th className="py-2 px-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((a) => (
                  <tr key={a.id} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="py-2 px-2 text-slate-500">{a.time.slice(11, 16)}</td>
                    <td className="py-2 px-2">
                      <button className="text-violet-700 hover:underline font-medium" onClick={()=>setSelected(a)}>{a.id}</button>
                    </td>
                    <td className="py-2 px-2 text-slate-600">{a.tracker.slice(-6)}</td>
                    <td className="py-2 px-2">
                      <span className="px-2 py-0.5 bg-gray-100 rounded text-xs">{a.cargo}</span>
                    </td>
                    <td className="py-2 px-2 text-slate-600">{a.lane}</td>
                    <td className="py-2 px-2 text-slate-600">{a.forwarder.id}</td>
                    <td className="py-2 px-2">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${badgeForFwdq(a.fwdq)}`}>{a.fwdq}</span>
                    </td>
                    <td className="py-2 px-2">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${badgeForIFBucket(a.if_bucket)}`}>{a.if_bucket.slice(0,4)}</span>
                    </td>
                    <td className="py-2 px-2">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${badgeForSeverity(a.severity)}`}>{a.severity}</span>
                    </td>
                    <td className="py-2 px-2 text-right">
                      <div className="flex gap-1 justify-end">
                        <button className="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50" onClick={()=>resolve(a.id)}>
                          Mon
                        </button>
                        <button className="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50" onClick={()=>resolve(a.id)}>
                          Peer
                        </button>
                        <button className="px-2 py-1 text-xs bg-rose-600 text-white rounded hover:bg-rose-700" onClick={()=>resolve(a.id)}>
                          Esc
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={10} className="py-8 text-center text-slate-500">No alerts match current filters.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold">Alert Details</h3>
            <ShieldCheck className="h-4 w-4 text-green-600"/>
          </div>
          {selected ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="font-medium">{selected.id}</div>
                <span className="px-2 py-1 bg-gray-100 rounded text-xs">{selected.tracker}</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><span className="text-slate-500">Cargo:</span> {selected.cargo}</div>
                <div><span className="text-slate-500">Lane:</span> {selected.lane}</div>
                <div className="col-span-2"><span className="text-slate-500">Forwarder:</span> {selected.forwarder.name}</div>
                <div>
                  <span className="text-slate-500">FwdQ:</span>{' '}
                  <span className={`px-2 py-0.5 rounded-full text-xs ${badgeForFwdq(selected.fwdq)}`}>{selected.fwdq}</span>
                </div>
                <div>
                  <span className="text-slate-500">IF:</span>{' '}
                  <span className={`px-2 py-0.5 rounded-full text-xs ${badgeForIFBucket(selected.if_bucket)}`}>{selected.if_bucket}</span>
                </div>
                <div>
                  <span className="text-slate-500">Severity:</span>{' '}
                  <span className={`px-2 py-0.5 rounded-full text-xs ${badgeForSeverity(selected.severity)}`}>{selected.severity}</span>
                </div>
                <div><span className="text-slate-500">Temp SLA:</span> {selected.temp_sla_violation_min}m</div>
              </div>
              <div className="text-xs bg-gray-50 p-2 rounded">
                <div className="font-medium mb-1">RL Suggestion: {selected.suggested}</div>
                <div className="text-gray-600">{selected.policy_reason}</div>
              </div>
              <div className="space-y-2 pt-2 border-t">
                <div className="text-xs text-gray-500 mb-2">Peer Review Actions:</div>
                <button className="w-full px-3 py-2 text-sm border border-green-500 text-green-700 rounded hover:bg-green-50" onClick={()=>resolve(selected.id)}>
                  ✓ Review OK → Monitor
                </button>
                <button className="w-full px-3 py-2 text-sm border border-red-500 text-red-700 rounded hover:bg-red-50" onClick={()=>resolve(selected.id)}>
                  ! Review NOT OK → Escalate
                </button>
                <button className="w-full px-3 py-2 text-sm border border-yellow-500 text-yellow-700 rounded hover:bg-yellow-50" onClick={()=>resolve(selected.id)}>
                  ? Uncertain → Increase Sampling
                </button>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <AlertTriangle className="h-12 w-12 mx-auto mb-2 opacity-50"/>
              <p className="text-sm">Select an alert to view details</p>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h3 className="text-sm font-semibold mb-3">Alerts by Cargo</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cargoAlerts}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="cargo" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip />
                <Bar dataKey="count" fill={COLORS.blue} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold">MTTR Trend</h3>
            <GaugeCircle className="h-4 w-4 text-gray-500"/>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeseries}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={COLORS.emerald} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={COLORS.emerald} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="t" hide />
                <YAxis stroke="#64748b" />
                <Tooltip />
                <Area type="monotone" dataKey="mttr" stroke={COLORS.emerald} strokeWidth={2} fillOpacity={1} fill="url(#g1)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold">Global Coverage</h3>
            <Globe className="h-4 w-4 text-gray-500"/>
          </div>
          <div className="h-48 rounded-xl bg-gradient-to-br from-violet-50 to-blue-50 border border-slate-200 flex items-center justify-center">
            <div className="text-center p-6">
              <Globe className="h-12 w-12 text-violet-400 mx-auto mb-2" />
              <p className="text-sm text-slate-600 font-medium">Real-time Global Tracking</p>
              <p className="text-xs text-slate-500 mt-1">Taiwan → LA/AMS/NRT + 15 routes</p>
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
                Multi-Agent Logistics Monitoring · Taiwan AI Excellence · {currentTime}
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