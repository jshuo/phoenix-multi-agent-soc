import React, { useMemo, useState } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Activity,
  AlertTriangle,
  Bell,
  Brain,
  Filter,
  GaugeCircle,
  Map as MapIcon,
  PackageSearch,
  Settings2,
  ShieldCheck,
  Thermometer,
  Truck,
} from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  AreaChart,
  Area,
} from "recharts";

/**
 * AIoT Multi‑Agent Logistics Monitoring (MaaS) Dashboard
 * ------------------------------------------------------
 * - Real‑time anomaly feed (Isolation Forest) with RL action suggestions
 * - Contextual Fusion: cargo_type ONLY (per design) + forwarder_quality_bucket injected into RL state
 * - Forwarder quality computed via Quality Snapshot (90d baseline) + optional 48h nowcast
 * - Human‑in‑the‑loop peer review flows: OK→Monitor, NOT OK→Escalate, Uncertain→Increase Sampling
 * - Built with shadcn/ui + Recharts + lucide-react
 */

// --- Demo data (mocked). Replace with live queries.
const CARGO = ["PERISH", "ELECT", "HAZ", "BULK", "FRAG"] as const;
const FWDQ = ["UNK", "LOW", "MED", "HIGH"] as const; // forwarder_quality_bucket
const IF_BUCKET = ["NORMAL", "WARN", "ANOMALOUS"] as const;
const SEVERITY = ["LOW", "MED", "HIGH"] as const;
const ACTIONS = ["monitor", "increase_sampling", "calibrate", "peer_check", "escalate"] as const;

const lanes = ["TPE→LAX", "AMS→JFK", "NRT→FRA", "SIN→SYD"]; 
const forwarders = [
  { id: "DGF", name: "DHL Global Forwarding" },
  { id: "K+N", name: "Kuehne+Nagel" },
  { id: "DBS", name: "DB Schenker" },
  { id: "EXP", name: "Expeditors" },
];

function nowISO() {
  return new Date().toISOString().slice(0, 19).replace("T", " ");
}

function badgeForSeverity(sev: string) {
  const map: Record<string, string> = {
    LOW: "bg-emerald-100 text-emerald-700",
    MED: "bg-amber-100 text-amber-700",
    HIGH: "bg-rose-100 text-rose-700",
  };
  return map[sev] || "bg-slate-100 text-slate-700";
}

function badgeForIFBucket(b: string) {
  const map: Record<string, string> = {
    NORMAL: "bg-slate-100 text-slate-700",
    WARN: "bg-amber-100 text-amber-700",
    ANOMALOUS: "bg-rose-100 text-rose-700",
  };
  return map[b] || "bg-slate-100 text-slate-700";
}

function badgeForFwdq(b: string) {
  const map: Record<string, string> = {
    UNK: "bg-slate-100 text-slate-700",
    LOW: "bg-rose-100 text-rose-700",
    MED: "bg-amber-100 text-amber-700",
    HIGH: "bg-emerald-100 text-emerald-700",
  };
  return map[b] || "bg-slate-100 text-slate-700";
}

// Fake timeseries for charts
const timeseries = Array.from({ length: 24 }).map((_, i) => ({
  t: `${i}:00`,
  anomalies: Math.max(0, Math.round(8 + 6 * Math.sin(i / 3) + (Math.random() * 4 - 2))),
  alerts: Math.max(0, Math.round(4 + 3 * Math.cos(i / 4) + (Math.random() * 3 - 1.5))),
  mttr: Math.max(10, Math.round(70 - 15 * Math.sin(i / 5) + (Math.random() * 10 - 5))),
}));

const fwdqDistribution = [
  { name: "UNK", value: 8 },
  { name: "LOW", value: 14 },
  { name: "MED", value: 46 },
  { name: "HIGH", value: 22 },
];

const cargoAlerts = CARGO.map((c) => ({ cargo: c, count: Math.round(5 + Math.random() * 20) }));

// Mocked live alerts feed
function makeAlert(id: number) {
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
    temp_sla_violation_min: Math.round(Math.random() * 60),
  };
}
const initialAlerts = Array.from({ length: 14 }).map((_, i) => makeAlert(i));

export default function MaasDashboard() {
  const [customer, setCustomer] = useState("Acme Biologics");
  const [range, setRange] = useState("24h");
  const [cargo, setCargo] = useState<string | "ALL">("ALL");
  const [lane, setLane] = useState<string | "ALL">("ALL");
  const [fwdq, setFwdq] = useState<string | "ALL">("ALL");
  const [severity, setSeverity] = useState<string | "ALL">("ALL");
  const [nowcast, setNowcast] = useState(true);
  const [alerts, setAlerts] = useState(initialAlerts);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<typeof alerts[number] | null>(null);

  const filtered = useMemo(() => {
    return alerts.filter((a) =>
      (cargo === "ALL" || a.cargo === cargo) &&
      (lane === "ALL" || a.lane === lane) &&
      (fwdq === "ALL" || a.fwdq === fwdq) &&
      (severity === "ALL" || a.severity === severity) &&
      (query === "" || `${a.id} ${a.tracker} ${a.forwarder.name}`.toLowerCase().includes(query.toLowerCase()))
    );
  }, [alerts, cargo, lane, fwdq, severity, query]);

  const kpis = useMemo(() => {
    const open = filtered.length;
    const active = 1242; // mock
    const onTime = 0.92; // mock
    const mtta = 14; // minutes
    const excursion = 38; // min/1k
    return { active, open, onTime, mtta, excursion };
  }, [filtered.length]);

  function resolve(id: string, outcome: "monitor" | "increase_sampling" | "calibrate" | "peer_check" | "escalate") {
    const a = alerts.find((x) => x.id === id);
    if (!a) return;
    // Remove from open list and set detail panel
    setAlerts((prev) => prev.filter((x) => x.id !== id));
    setSelected({ ...a, suggested: outcome });
  }

  return (
    <div className="w-full min-h-screen bg-gradient-to-b from-slate-50 to-white p-4 md:p-8">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <PackageSearch className="h-8 w-8 text-violet-700" />
          <div>
            <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">MaaS Operations Dashboard</h1>
            <p className="text-sm text-slate-600">AIoT Multi‑Agent Logistics Monitoring · last update {nowISO()}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" className="gap-2"><Settings2 className="h-4 w-4"/>Settings</Button>
          <Button className="gap-2"><Bell className="h-4 w-4"/>Subscribe</Button>
        </div>
      </div>

      {/* Filters */}
      <Card className="mt-6">
        <CardContent className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-6 gap-3 items-end">
            <div>
              <Label>Customer</Label>
              <Input value={customer} onChange={(e) => setCustomer(e.target.value)} />
            </div>
            <div>
              <Label>Time Range</Label>
              <Select value={range} onValueChange={setRange}>
                <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
                <SelectContent>
                  <SelectItem value="2h">Last 2h</SelectItem>
                  <SelectItem value="24h">Last 24h</SelectItem>
                  <SelectItem value="7d">Last 7d</SelectItem>
                  <SelectItem value="30d">Last 30d</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Cargo Type</Label>
              <Select value={cargo} onValueChange={(v) => setCargo(v as any)}>
                <SelectTrigger className="w-full"><SelectValue placeholder="ALL"/></SelectTrigger>
                <SelectContent>
                  <SelectItem value="ALL">ALL</SelectItem>
                  {CARGO.map((c) => (<SelectItem key={c} value={c}>{c}</SelectItem>))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Lane</Label>
              <Select value={lane} onValueChange={(v) => setLane(v as any)}>
                <SelectTrigger className="w-full"><SelectValue placeholder="ALL"/></SelectTrigger>
                <SelectContent>
                  <SelectItem value="ALL">ALL</SelectItem>
                  {lanes.map((c) => (<SelectItem key={c} value={c}>{c}</SelectItem>))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Forwarder Quality</Label>
              <Select value={fwdq} onValueChange={(v) => setFwdq(v as any)}>
                <SelectTrigger className="w-full"><SelectValue placeholder="ALL"/></SelectTrigger>
                <SelectContent>
                  <SelectItem value="ALL">ALL</SelectItem>
                  {FWDQ.map((b) => (<SelectItem key={b} value={b}>{b}</SelectItem>))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Severity</Label>
              <Select value={severity} onValueChange={(v) => setSeverity(v as any)}>
                <SelectTrigger className="w-full"><SelectValue placeholder="ALL"/></SelectTrigger>
                <SelectContent>
                  <SelectItem value="ALL">ALL</SelectItem>
                  {SEVERITY.map((s) => (<SelectItem key={s} value={s}>{s}</SelectItem>))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Switch checked={nowcast} onCheckedChange={setNowcast} id="nowcast"/>
              <Label htmlFor="nowcast">Use real‑time nowcast for forwarder quality</Label>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <Filter className="h-4 w-4 text-slate-500"/>
              <Input placeholder="Search alert id / tracker / forwarder" value={query} onChange={e=>setQuery(e.target.value)} className="max-w-sm"/>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* KPI Row */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium">Active Shipments</CardTitle>
            <Truck className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">{kpis.active.toLocaleString()}</div>
            <p className="text-xs text-slate-500">across all tenants</p>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium">Open Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">{kpis.open}</div>
            <p className="text-xs text-slate-500">filtered view</p>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium">On‑time Milestones</CardTitle>
            <ShieldCheck className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">{Math.round(kpis.onTime*100)}%</div>
            <p className="text-xs text-slate-500">rolling baseline</p>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium">MTTA</CardTitle>
            <Activity className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">{kpis.mtta}m</div>
            <p className="text-xs text-slate-500">mean time to acknowledge</p>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium">Excursion Minutes / 1k</CardTitle>
            <Thermometer className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">{kpis.excursion}</div>
            <p className="text-xs text-slate-500">temp/shock normalized</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="mt-6 grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="shadow-sm xl:col-span-2">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium">Anomalies & Alerts (last 24h)</CardTitle>
            <Brain className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeseries} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" hide />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="anomalies" dot={false} strokeWidth={2} name="IF anomalies" />
                <Line type="monotone" dataKey="alerts" dot={false} strokeWidth={2} name="Alerts" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Forwarder Quality Distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={fwdqDistribution} dataKey="value" nameKey="name" outerRadius={80}>
                  {fwdqDistribution.map((_, i) => (
                    <Cell key={i} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Alerts by Cargo Type</CardTitle>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cargoAlerts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cargo" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex items-center justify-between">
            <CardTitle className="text-sm font-medium">MTTR Trend (mins)</CardTitle>
            <GaugeCircle className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeseries}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopOpacity={0.4} />
                    <stop offset="95%" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" hide />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="mttr" strokeWidth={2} fillOpacity={1} fill="url(#g1)" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex items-center justify-between">
            <CardTitle className="text-sm font-medium">Map View (placeholder)</CardTitle>
            <MapIcon className="h-4 w-4 text-slate-500"/>
          </CardHeader>
          <CardContent>
            <div className="h-56 rounded-xl bg-gradient-to-br from-slate-100 to-slate-200 grid place-items-center text-slate-500 text-sm">
              Integrate map provider here (deck.gl/Mapbox/etc.)
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Alerts + Details */}
      <div className="mt-6 grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="shadow-sm xl:col-span-2">
          <CardHeader className="pb-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm font-medium">Live Alerts</CardTitle>
              <Badge variant="secondary" className="rounded-full">{filtered.length}</Badge>
            </div>
            <Tabs defaultValue="all">
              <TabsList>
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="anomalous">Anomalous</TabsTrigger>
                <TabsTrigger value="needs-review">Needs Review</TabsTrigger>
              </TabsList>
            </Tabs>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-slate-500">
                  <th className="py-2 pr-4">Time</th>
                  <th className="py-2 pr-4">ID</th>
                  <th className="py-2 pr-4">Tracker</th>
                  <th className="py-2 pr-4">Cargo</th>
                  <th className="py-2 pr-4">Lane</th>
                  <th className="py-2 pr-4">Forwarder</th>
                  <th className="py-2 pr-4">FwdQ</th>
                  <th className="py-2 pr-4">IF</th>
                  <th className="py-2 pr-4">Sev</th>
                  <th className="py-2 pr-4">RL Suggest</th>
                  <th className="py-2 pr-4">Why</th>
                  <th className="py-2 pr-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((a) => (
                  <tr key={a.id} className="border-t">
                    <td className="py-2 pr-4 whitespace-nowrap text-slate-500">{a.time}</td>
                    <td className="py-2 pr-4"><button className="text-violet-700 hover:underline" onClick={()=>setSelected(a)}>{a.id}</button></td>
                    <td className="py-2 pr-4">{a.tracker}</td>
                    <td className="py-2 pr-4"><Badge className="rounded-full" variant="outline">{a.cargo}</Badge></td>
                    <td className="py-2 pr-4">{a.lane}</td>
                    <td className="py-2 pr-4">{a.forwarder.name}</td>
                    <td className="py-2 pr-4"><span className={`px-2 py-1 rounded-full text-xs ${badgeForFwdq(a.fwdq)}`}>{a.fwdq}</span></td>
                    <td className="py-2 pr-4"><span className={`px-2 py-1 rounded-full text-xs ${badgeForIFBucket(a.if_bucket)}`}>{a.if_bucket}</span></td>
                    <td className="py-2 pr-4"><span className={`px-2 py-1 rounded-full text-xs ${badgeForSeverity(a.severity)}`}>{a.severity}</span></td>
                    <td className="py-2 pr-4"><code className="text-xs">{a.suggested}</code></td>
                    <td className="py-2 pr-4 text-slate-500">{a.policy_reason}</td>
                    <td className="py-2 pr-4 text-right">
                      <div className="flex gap-2 justify-end">
                        <Button size="sm" variant="outline" onClick={()=>resolve(a.id, "monitor")}>Monitor</Button>
                        <Button size="sm" variant="outline" onClick={()=>resolve(a.id, "increase_sampling")}>+Sampling</Button>
                        <Button size="sm" variant="outline" onClick={()=>resolve(a.id, "peer_check")}>Peer</Button>
                        <Button size="sm" onClick={()=>resolve(a.id, "escalate")} className="bg-rose-600 hover:bg-rose-700">Escalate</Button>
                      </div>
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={12} className="py-8 text-center text-slate-500">No alerts match current filters.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <Card className="shadow-sm">
          <CardHeader className="pb-2 flex items-center justify-between">
            <CardTitle className="text-sm font-medium">Ticket / Peer Review</CardTitle>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span className="inline-flex items-center gap-1"><ShieldCheck className="h-3 w-3"/> Safety overrides on</span>
            </div>
          </CardHeader>
          <CardContent>
            {selected ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="font-medium">{selected.id}</div>
                  <Badge variant="outline">{selected.tracker}</Badge>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div><span className="text-slate-500">Cargo:</span> {selected.cargo}</div>
                  <div><span className="text-slate-500">Lane:</span> {selected.lane}</div>
                  <div><span className="text-slate-500">Forwarder:</span> {selected.forwarder.name}</div>
                  <div><span className="text-slate-500">FwdQ:</span> <span className={`px-2 py-0.5 rounded-full text-xs ${badgeForFwdq(selected.fwdq)}`}>{selected.fwdq}</span></div>
                  <div><span className="text-slate-500">IF:</span> <span className={`px-2 py-0.5 rounded-full text-xs ${badgeForIFBucket(selected.if_bucket)}`}>{selected.if_bucket}</span></div>
                  <div><span className="text-slate-500">Severity:</span> <span className={`px-2 py-0.5 rounded-full text-xs ${badgeForSeverity(selected.severity)}`}>{selected.severity}</span></div>
                  <div><span className="text-slate-500">Temp SLA mins:</span> {selected.temp_sla_violation_min}</div>
                </div>
                <div className="text-sm text-slate-600">RL suggested: <code>{selected.suggested}</code> · Why: {selected.policy_reason}</div>

                <Tabs defaultValue="review">
                  <TabsList className="w-full">
                    <TabsTrigger value="review" className="flex-1">Review</TabsTrigger>
                    <TabsTrigger value="history" className="flex-1">History</TabsTrigger>
                  </TabsList>
                  <TabsContent value="review" className="space-y-3">
                    <div className="text-xs text-slate-500">Choose outcome (maps to flows):</div>
                    <div className="flex flex-wrap gap-2">
                      <Button variant="outline" size="sm" onClick={()=>resolve(selected.id, "monitor")}>Review OK → Monitor</Button>
                      <Button variant="outline" size="sm" onClick={()=>resolve(selected.id, "increase_sampling")}>Uncertain → +Sampling</Button>
                      <Button size="sm" className="bg-rose-600 hover:bg-rose-700" onClick={()=>resolve(selected.id, "escalate")}>
                        Review NOT OK → Escalate
                      </Button>
                    </div>
                    <div className="text-xs text-slate-500">Safe defaults apply if timeout reached.</div>
                  </TabsContent>
                  <TabsContent value="history">
                    <div className="text-xs text-slate-500">(Mock) Last actions on this tracker:</div>
                    <ul className="text-sm list-disc pl-5 space-y-1">
                      <li>2025-09-15 08:22Z · Calibrate due to pressure drift</li>
                      <li>2025-09-07 19:05Z · Escalate (verified anomaly)</li>
                      <li>2025-08-30 11:44Z · Monitor (verified normal)</li>
                    </ul>
                  </TabsContent>
                </Tabs>
              </div>
            ) : (
              <div className="text-sm text-slate-500 grid place-items-center h-64">
                Select an alert to open peer review panel.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Footer legend */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-500">
        <div className="flex items-center gap-2"><AlertTriangle className="h-4 w-4"/> IF bucket: <Badge className="rounded-full" variant="outline">NORMAL/WARN/ANOMALOUS</Badge></div>
        <div className="flex items-center gap-2"><Brain className="h-4 w-4"/> RL state: (if_bucket, severity, cargo_type, forwarder_quality_bucket)</div>
        <div className="flex items-center gap-2"><Settings2 className="h-4 w-4"/> FwdQ: 90d baseline {nowcast ? "+ 48h nowcast" : "(baseline only)"}; bucketized with hysteresis; per‑leg freeze</div>
      </div>
    </div>
  );
}
