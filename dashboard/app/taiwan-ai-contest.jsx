import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertTriangle, Activity, Radar, MapPin, Rocket, Settings2, LineChart, Play, Pause, RefreshCw, ChevronRight, Shield, Bell, Search, Zap, Users, Target, Radio, Plane } from "lucide-react";

const UAVS = [
  { id: "UAV-Alpha-01", status: "ok", battery: 82, eta: "05:42", pos: [15, 28], mission: "前線補給", cargo: "彈藥" },
  { id: "UAV-Bravo-02", status: "warn", battery: 58, eta: "07:10", pos: [52, 65], mission: "醫療物資", cargo: "血漿" },
  { id: "UAV-Charlie-03", status: "alert", battery: 35, eta: "03:18", pos: [75, 18], mission: "通訊裝備", cargo: "電子設備" },
  { id: "UAV-Delta-04", status: "ok", battery: 91, eta: "09:03", pos: [28, 72], mission: "補給前哨站", cargo: "糧食" },
  { id: "UAV-Echo-05", status: "ok", battery: 76, eta: "04:55", pos: [62, 42], mission: "偵察支援", cargo: "感測器" },
];

const ANOMALIES = [
  { id: "UAV-Bravo-02", type: "GPS 訊號異常跳動", score: 0.87, rl: "Escalate", peer: "緊急處理", threat: "high", detail: "疑似電子戰干擾，已切換 INS 導航", action: "啟動慣性導航備援系統" },
  { id: "UAV-Charlie-03", type: "電池異常耗電 + 溫度過高", score: 0.73, rl: "Escalate", peer: "待審", threat: "high", detail: "電量僅剩 35%，建議緊急降落", action: "計算最近基地降落點" },
  { id: "UAV-Echo-05", type: "通訊訊號間歇中斷", score: 0.68, rl: "Peer Check", peer: "待審", threat: "medium", detail: "與其他 UAV 交叉驗證訊號", action: "增加訊號取樣頻率" },
  { id: "UAV-Alpha-01", type: "航線輕微偏移", score: 0.42, rl: "Monitor", peer: "OK", threat: "low", detail: "陣風影響，自動校正中", action: "持續監控，無需人工介入" },
];

const MISSIONS = [
  { id: "M-2025-001", name: "前線醫療物資運補", status: "進行中", uavs: 3, priority: "緊急", completion: 65 },
  { id: "M-2025-002", name: "彈藥補給任務", status: "進行中", uavs: 2, priority: "高", completion: 42 },
  { id: "M-2025-003", name: "通訊設備部署", status: "待命", uavs: 1, priority: "中", completion: 0 },
];

const KPI = {
  missionSuccess: 98.5,
  aiAccuracy: 94.2,
  detectionLatency: 2.3,
  humanIntervention: 8,
  systemUptime: 99.7,
  uavsOnline: 5,
  totalUavs: 5,
};

const statusColor = (s) => ({ ok: "bg-emerald-500", warn: "bg-amber-400", alert: "bg-rose-500" }[s] || "bg-slate-400");
const statusText = (s) => ({ ok: "正常", warn: "警戒", alert: "異常" }[s] || s);
const threatColor = (t) => ({ low: "text-slate-600", medium: "text-amber-600", high: "text-rose-600" }[t] || "text-slate-600");
const priorityColor = (p) => ({ 緊急: "bg-rose-500", 高: "bg-orange-500", 中: "bg-blue-500", 低: "bg-slate-500" }[p] || "bg-slate-500");

function SectionHeader({ icon: Icon, title, desc, right }) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-xl bg-gradient-to-br from-slate-800 to-slate-600 text-white shadow-md">
          <Icon className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-base font-semibold leading-tight">{title}</h3>
          {desc && <p className="text-xs text-slate-500 mt-0.5">{desc}</p>}
        </div>
      </div>
      <div>{right}</div>
    </div>
  );
}

function MiniLegend() {
  return (
    <div className="flex items-center gap-3 text-xs">
      <div className="flex items-center gap-1">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500" />
        <span className="text-slate-600">正常</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-400" />
        <span className="text-slate-600">警戒</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-rose-500" />
        <span className="text-slate-600">異常</span>
      </div>
    </div>
  );
}

function UavTacticalMap() {
  const [pulse, setPulse] = useState(true);
  
  useEffect(() => {
    const interval = setInterval(() => setPulse(p => !p), 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative w-full h-80 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-slate-700 overflow-hidden shadow-xl">
      <div className="absolute inset-0 opacity-20" style={{
        backgroundImage: 'linear-gradient(#334155 1px, transparent 1px), linear-gradient(90deg, #334155 1px, transparent 1px)',
        backgroundSize: '32px 32px'
      }} />
      
      <div className="absolute inset-0">
        <div className={`absolute inset-0 bg-gradient-conic from-emerald-500/0 via-emerald-500/10 to-emerald-500/0 transition-opacity duration-1000 ${pulse ? 'opacity-100' : 'opacity-30'}`} />
      </div>

      <div className="absolute top-4 left-4 px-3 py-1.5 rounded-lg bg-slate-800/80 backdrop-blur-sm border border-slate-600 text-xs text-emerald-400 font-mono">
        戰區座標: 24°N 121°E
      </div>
      
      {UAVS.map((uav, i) => (
        <div
          key={uav.id}
          className="absolute group cursor-pointer"
          style={{ left: `${uav.pos[0]}%`, top: `${uav.pos[1]}%`, transform: 'translate(-50%, -50%)' }}
        >
          <div className={`absolute inset-0 w-8 h-8 -m-2 rounded-full ${statusColor(uav.status)} opacity-30 animate-ping`} />
          
          <div className={`relative w-4 h-4 rounded-full ${statusColor(uav.status)} shadow-lg border-2 border-white flex items-center justify-center`}>
            <Plane className="w-2.5 h-2.5 text-white" style={{ transform: 'rotate(45deg)' }} />
          </div>
          
          <div className="absolute left-6 top-0 w-48 p-2 rounded-lg bg-slate-800/95 backdrop-blur-sm border border-slate-600 text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-xl">
            <div className="font-bold text-emerald-400 mb-1">{uav.id}</div>
            <div className="space-y-0.5 text-slate-300">
              <div>任務: {uav.mission}</div>
              <div>貨物: {uav.cargo}</div>
              <div>電量: {uav.battery}%</div>
              <div>ETA: {uav.eta}</div>
              <div className="flex items-center gap-1 mt-1">
                <span className={`inline-block w-2 h-2 rounded-full ${statusColor(uav.status)}`} />
                狀態: {statusText(uav.status)}
              </div>
            </div>
          </div>
        </div>
      ))}

      <div className="absolute bottom-3 left-3">
        <MiniLegend />
      </div>
      
      <div className="absolute bottom-3 right-3 flex gap-2">
        <div className="px-2 py-1 rounded bg-slate-800/80 backdrop-blur-sm border border-slate-600 text-xs text-white">
          <Plane className="w-3 h-3 inline mr-1" />
          {UAVS.length} 架運作中
        </div>
        <div className="px-2 py-1 rounded bg-slate-800/80 backdrop-blur-sm border border-slate-600 text-xs text-white">
          <Target className="w-3 h-3 inline mr-1" />
          {MISSIONS.length} 任務
        </div>
      </div>
    </div>
  );
}

function AnomalyTable() {
  return (
    <div className="rounded-xl border-2 border-slate-200 overflow-hidden shadow-sm">
      <div className="grid grid-cols-12 bg-gradient-to-r from-slate-800 to-slate-700 text-white text-xs font-semibold">
        <div className="col-span-2 px-3 py-2.5">UAV 編號</div>
        <div className="col-span-3 px-3 py-2.5">異常類型與描述</div>
        <div className="col-span-2 px-3 py-2.5">威脅 / 信心度</div>
        <div className="col-span-2 px-3 py-2.5">RL 決策 / 審查</div>
        <div className="col-span-3 px-3 py-2.5">系統建議動作</div>
      </div>
      {ANOMALIES.map((a, idx) => (
        <div key={a.id} className={`grid grid-cols-12 text-sm ${idx % 2 ? "bg-white" : "bg-slate-50"} hover:bg-blue-50 transition-colors`}>
          <div className="col-span-2 px-3 py-3 font-mono text-slate-700 font-medium">{a.id}</div>
          <div className="col-span-3 px-3 py-3">
            <div className="font-semibold text-slate-800">{a.type}</div>
            <div className="text-xs text-slate-500 mt-0.5">{a.detail}</div>
          </div>
          <div className="col-span-2 px-3 py-3">
            <Badge className={`${threatColor(a.threat)} bg-transparent border font-semibold mb-1`}>
              {a.threat === 'high' ? '🔴 高威脅' : a.threat === 'medium' ? '🟡 中威脅' : '🟢 低威脅'}
            </Badge>
            <div className="flex items-center gap-2 mt-1">
              <div className="w-16 flex-1"><Progress value={a.score * 100} className="h-1.5" /></div>
              <span className="tabular-nums text-xs font-bold text-slate-700">{(a.score * 100).toFixed(0)}%</span>
            </div>
          </div>
          <div className="col-span-2 px-3 py-3 space-y-1">
            <Badge variant="secondary" className="rounded-full font-medium w-full justify-center">{a.rl}</Badge>
            {a.peer === "OK" ? (
              <Badge className="bg-emerald-500 hover:bg-emerald-600 w-full justify-center">✓ OK</Badge>
            ) : a.peer === "緊急處理" ? (
              <Badge className="bg-rose-600 hover:bg-rose-700 w-full justify-center">🚨 {a.peer}</Badge>
            ) : (
              <Badge className="bg-amber-500 hover:bg-amber-600 w-full justify-center">⏱ {a.peer}</Badge>
            )}
          </div>
          <div className="col-span-3 px-3 py-3">
            <div className="text-xs text-slate-700 leading-relaxed">
              <ChevronRight className="w-3 h-3 inline mr-1 text-blue-600" />
              {a.action}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function PeerReviewFlow() {
  return (
    <div className="flex flex-col gap-3 p-4 rounded-xl bg-slate-50 border-2 border-slate-200">
      <div className="flex items-center gap-2 text-sm">
        <Badge className="bg-slate-800">AI 決策流程 (Human-in-the-Loop)</Badge>
        <span className="text-slate-600">OK → 持續監控 | Not OK → 立即升級 | 不確定 → 增加感測取樣</span>
      </div>
      <div className="flex items-center gap-4 text-xs text-slate-600">
        <span className="inline-flex items-center gap-1">
          <Shield className="w-3.5 h-3.5 text-blue-600" /> 
          <span className="font-medium">稽核軌跡</span>
        </span>
        <span className="inline-flex items-center gap-1">
          <Bell className="w-3.5 h-3.5 text-orange-600" /> 
          <span className="font-medium">即時通知</span>
        </span>
        <span className="inline-flex items-center gap-1">
          <LineChart className="w-3.5 h-3.5 text-emerald-600" /> 
          <span className="font-medium">性能追蹤</span>
        </span>
        <span className="inline-flex items-center gap-1">
          <Users className="w-3.5 h-3.5 text-purple-600" /> 
          <span className="font-medium">多人協同</span>
        </span>
      </div>
    </div>
  );
}

function DigitalTwinPanel() {
  return (
    <div className="rounded-2xl border-2 border-slate-200 p-4 bg-gradient-to-br from-white to-slate-50 shadow-sm">
      <div className="grid grid-cols-1 gap-4">
        <Card className="shadow-md border-2 border-slate-200">
          <CardHeader className="pb-3 bg-gradient-to-r from-rose-50 to-orange-50">
            <CardTitle className="text-base flex items-center gap-2">
              <Radio className="w-5 h-5 text-rose-600" />
              戰術場景：GPS 訊號干擾攻擊
            </CardTitle>
            <CardDescription className="font-medium">
              RL 策略響應：偵測異常 → Peer Check → 確認威脅 → Escalate → 啟動慣性導航備援
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="h-32 rounded-xl bg-gradient-to-br from-rose-100 via-orange-100 to-amber-100 border-2 border-rose-300 flex flex-col items-center justify-center text-rose-800">
              <AlertTriangle className="w-8 h-8 mb-2" />
              <div className="text-sm font-bold">模擬：電子戰 GPS 欺騙 → 航線偏移 → AI 自動切換 INS 導航</div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="shadow-md border-2 border-slate-200">
          <CardHeader className="pb-3 bg-gradient-to-r from-blue-50 to-cyan-50">
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="w-5 h-5 text-blue-600" />
              關鍵場景：醫療冷鏈溫度失控
            </CardTitle>
            <CardDescription className="font-medium">
              RL 策略響應：溫度超標 → 即刻 Escalate → 通知現場 SOP → 建議緊急降落最近基地
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="h-32 rounded-xl bg-gradient-to-br from-blue-100 via-cyan-100 to-teal-100 border-2 border-blue-300 flex flex-col items-center justify-center text-blue-800">
              <Activity className="w-8 h-8 mb-2" />
              <div className="text-sm font-bold">模擬：血漿運輸溫度異常 → RL 決策緊急處置 → 保障戰場醫療</div>
            </div>
          </CardContent>
        </Card>
      </div>
      <div className="mt-4 flex items-center gap-2">
        <Button size="sm" className="rounded-xl bg-emerald-600 hover:bg-emerald-700">
          <Play className="w-4 h-4 mr-1" /> 開始戰術模擬
        </Button>
        <Button variant="secondary" size="sm" className="rounded-xl">
          <Pause className="w-4 h-4 mr-1" /> 暫停
        </Button>
        <Button variant="outline" size="sm" className="rounded-xl">
          <RefreshCw className="w-4 h-4 mr-1" /> 重置場景
        </Button>
      </div>
    </div>
  );
}

function MissionPanel() {
  return (
    <div className="space-y-3">
      {MISSIONS.map((m) => (
        <Card key={m.id} className="shadow-sm hover:shadow-md transition-shadow">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-slate-600" />
                <CardTitle className="text-sm">{m.name}</CardTitle>
              </div>
              <Badge className={`${priorityColor(m.priority)} text-white`}>{m.priority}</Badge>
            </div>
            <CardDescription className="text-xs mt-1">
              任務編號: {m.id} | 部署 UAV: {m.uavs} 架 | 狀態: {m.status}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">任務進度</span>
                <span className="font-bold text-slate-700">{m.completion}%</span>
              </div>
              <Progress value={m.completion} className="h-2" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export default function DefenseAIDashboard() {
  const [isLive, setIsLive] = useState(true);

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-100">
      <div className="sticky top-0 z-40 backdrop-blur-lg bg-white/80 border-b-2 border-slate-200 shadow-sm">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-slate-900 to-slate-700 text-white grid place-items-center font-bold text-lg shadow-md">
              AI
            </div>
            <div>
              <div className="text-sm font-bold leading-tight flex items-center gap-2">
                國防 AI 應用創新競賽 
                <Badge variant="secondary" className="rounded-full bg-blue-100 text-blue-700 font-semibold">Demo</Badge>
              </div>
              <div className="text-xs text-slate-600 font-medium">UAV 智慧後勤 × AIoT × 強化學習 × 多智能體協同</div>
            </div>
            <Badge className="ml-2 rounded-full bg-gradient-to-r from-emerald-600 to-green-600 text-white font-semibold shadow-sm">
              軍民通用 Dual-Use
            </Badge>
            <div className="flex items-center gap-1 ml-2">
              <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'}`} />
              <span className="text-xs font-medium text-slate-600">{isLive ? '即時監控' : '離線'}</span>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-2">
            <div className="relative w-56">
              <Search className="absolute left-2.5 top-2.5 w-4 h-4 text-slate-400" />
              <Input placeholder="搜尋 UAV / 任務 / 異常…" className="pl-9 rounded-xl border-slate-300" />
            </div>
            <Button variant="outline" className="rounded-xl border-slate-300">
              <LineChart className="w-4 h-4 mr-1" />
              匯出報表
            </Button>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 flex flex-col gap-6">
          <Card className="shadow-lg border-2 border-slate-200">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50">
              <SectionHeader 
                icon={Radar} 
                title="戰術態勢圖 (Tactical Situation)" 
                desc="UAV 群集即時位置、任務狀態與預估到達時間" 
                right={<MiniLegend />} 
              />
            </CardHeader>
            <CardContent className="pt-4">
              <UavTacticalMap />
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                {UAVS.slice(0, 4).map((u) => (
                  <div key={u.id} className="rounded-xl border-2 border-slate-200 p-4 bg-white hover:bg-slate-50 transition-colors shadow-sm">
                    <div className="flex items-start justify-between mb-3">
                      <div className="text-sm text-slate-600 font-mono font-bold">{u.id}</div>
                      <Badge className={`${statusColor(u.status)} text-white px-3 py-1 text-sm font-semibold`}>
                        {statusText(u.status)}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm">
                        <Target className="w-4 h-4 text-blue-600 flex-shrink-0" />
                        <span className="text-slate-600">任務:</span>
                        <span className="font-bold text-slate-900">{u.mission}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <MapPin className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                        <span className="text-slate-600">貨物:</span>
                        <span className="font-bold text-slate-900">{u.cargo}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-700 mt-2 pt-2 border-t border-slate-200">
                        <Zap className="w-4 h-4 text-amber-500" />
                        <span className="font-semibold">電量 {u.battery}% · ETA {u.eta}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-lg border-2 border-slate-200">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-orange-50">
              <SectionHeader 
                icon={Activity} 
                title="AI 異常偵測系統 (Anomaly Detection)" 
                desc="Kalman Filter + Isolation Forest + RL 決策引擎" 
                right={
                  <Button variant="outline" size="sm" className="rounded-xl border-slate-300">
                    <Settings2 className="w-4 h-4 mr-1" /> 調整參數
                  </Button>
                } 
              />
            </CardHeader>
            <CardContent className="pt-4">
              <Tabs defaultValue="table" className="w-full">
                <TabsList className="rounded-xl bg-slate-100">
                  <TabsTrigger value="table" className="rounded-lg">異常清單</TabsTrigger>
                  <TabsTrigger value="charts" className="rounded-lg">分析圖表</TabsTrigger>
                </TabsList>
                <TabsContent value="table" className="mt-4 space-y-4">
                  <AnomalyTable />
                  <PeerReviewFlow />
                </TabsContent>
                <TabsContent value="charts" className="mt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="h-40 rounded-xl border-2 border-slate-200 bg-white grid place-items-center text-slate-500 text-sm shadow-sm">
                      <div className="text-center">
                        <LineChart className="w-8 h-8 mx-auto mb-2 text-slate-400" />
                        <div>異常分數時序圖</div>
                      </div>
                    </div>
                    <div className="h-40 rounded-xl border-2 border-slate-200 bg-white grid place-items-center text-slate-500 text-sm shadow-sm">
                      <div className="text-center">
                        <Activity className="w-8 h-8 mx-auto mb-2 text-slate-400" />
                        <div>感測器數據趨勢</div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-5 flex flex-col gap-6">
          <Card className="shadow-lg border-2 border-slate-200">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-emerald-50">
              <SectionHeader 
                icon={Activity} 
                title="即時運作指標 (Real-time Metrics)" 
                desc="當前系統性能與 AI 決策效能監控" 
                right={
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-xs font-medium text-slate-600">即時更新</span>
                  </div>
                } 
              />
            </CardHeader>
            <CardContent className="pt-4">
              <div className="grid grid-cols-2 gap-3">
                <Card className="shadow-sm hover:shadow-md transition-shadow border-2 border-emerald-200">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardDescription className="text-xs font-medium">任務成功率</CardDescription>
                      <Target className="w-4 h-4 text-emerald-600" />
                    </div>
                    <CardTitle className="text-3xl tracking-tight text-emerald-700">
                      {KPI.missionSuccess}
                      <span className="text-base text-slate-500 ml-1">%</span>
                    </CardTitle>
                    <div className="text-xs text-slate-500 mt-1">24 小時滾動平均</div>
                  </CardHeader>
                  <CardContent>
                    <Progress value={KPI.missionSuccess} className="h-2 bg-emerald-100" />
                  </CardContent>
                </Card>

                <Card className="shadow-sm hover:shadow-md transition-shadow border-2 border-blue-200">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardDescription className="text-xs font-medium">AI 決策準確度</CardDescription>
                      <Shield className="w-4 h-4 text-blue-600" />
                    </div>
                    <CardTitle className="text-3xl tracking-tight text-blue-700">
                      {KPI.aiAccuracy}
                      <span className="text-base text-slate-500 ml-1">%</span>
                    </CardTitle>
                    <div className="text-xs text-slate-500 mt-1">本日累計</div>
                  </CardHeader>
                  <CardContent>
                    <Progress value={KPI.aiAccuracy} className="h-2 bg-blue-100" />
                  </CardContent>
                </Card>

                <Card className="shadow-sm hover:shadow-md transition-shadow border-2 border-purple-200">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardDescription className="text-xs font-medium">異常偵測延遲</CardDescription>
                      <Zap className="w-4 h-4 text-purple-600" />
                    </div>
                    <CardTitle className="text-3xl tracking-tight text-purple-700">
                      {KPI.detectionLatency}
                      <span className="text-base text-slate-500 ml-1">秒</span>
                    </CardTitle>
                    <div className="text-xs text-slate-500 mt-1">平均響應時間</div>
                  </CardHeader>
                  <CardContent>
                    <Progress value={100 - (KPI.detectionLatency * 10)} className="h-2 bg-purple-100" />
                  </CardContent>
                </Card>

                <Card className="shadow-sm hover:shadow-md transition-shadow border-2 border-amber-200">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardDescription className="text-xs font-medium">人工介入率</CardDescription>
                      <Users className="w-4 h-4 text-amber-600" />
                    </div>
                    <CardTitle className="text-3xl tracking-tight text-amber-700">
                      {KPI.humanIntervention}
                      <span className="text-base text-slate-500 ml-1">%</span>
                    </CardTitle>
                    <div className="text-xs text-slate-500 mt-1">本週平均</div>
                  </CardHeader>
                  <CardContent>
                    <Progress value={KPI.humanIntervention} className="h-2 bg-amber-100" />
                  </CardContent>
                </Card>

                <Card className="shadow-sm hover:shadow-md transition-shadow border-2 border-cyan-200">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardDescription className="text-xs font-medium">系統可用度</CardDescription>
                      <Activity className="w-4 h-4 text-cyan-600" />
                    </div>
                    <CardTitle className="text-3xl tracking-tight text-cyan-700">
                      {KPI.systemUptime}
                      <span className="text-base text-slate-500 ml-1">%</span>
                    </CardTitle>
                    <div className="text-xs text-slate-500 mt-1">本月累計 Uptime</div>
                  </CardHeader>
                  <CardContent>
                    <Progress value={KPI.systemUptime} className="h-2 bg-cyan-100" />
                  </CardContent>
                </Card>

                <Card className="shadow-sm hover:shadow-md transition-shadow border-2 border-slate-200">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardDescription className="text-xs font-medium">UAV 群集狀態</CardDescription>
                      <Plane className="w-4 h-4 text-slate-600" />
                    </div>
                    <CardTitle className="text-3xl tracking-tight text-slate-700">
                      {KPI.uavsOnline}/{KPI.totalUavs}
                      <span className="text-base text-slate-500 ml-1">架</span>
                    </CardTitle>
                    <div className="text-xs text-slate-500 mt-1">在線運作中</div>
                  </CardHeader>
                  <CardContent>
                    <Progress value={(KPI.uavsOnline / KPI.totalUavs) * 100} className="h-2 bg-slate-100" />
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-lg border-2 border-slate-200">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-purple-50">
              <SectionHeader 
                icon={Target} 
                title="當前任務狀態" 
                desc="即時追蹤各項國防後勤任務進度"
                right={
                  <Button variant="outline" size="sm" className="rounded-xl border-slate-300">
                    <Rocket className="w-3.5 h-3.5 mr-1" /> 新增任務
                  </Button>
                }
              />
            </CardHeader>
            <CardContent className="pt-4">
              <MissionPanel />
            </CardContent>
          </Card>

          <Card className="shadow-lg border-2 border-slate-200">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-indigo-50">
              <SectionHeader 
                icon={Rocket} 
                title="數位分身 (Digital Twin)" 
                desc="戰術場景模擬與 RL 策略驗證" 
                right={
                  <Button variant="outline" size="sm" className="rounded-xl border-slate-300">
                    <Settings2 className="w-3.5 h-3.5 mr-1" /> 場景庫
                  </Button>
                } 
              />
            </CardHeader>
            <CardContent className="pt-4">
              <DigitalTwinPanel />
            </CardContent>
          </Card>

          <Card className="shadow-lg border-2 border-slate-200">
            <CardHeader className="bg-gradient-to-r from-slate-50 to-cyan-50">
              <SectionHeader 
                icon={MapPin} 
                title="情境融合 (Contextual Fusion)" 
                desc="貨品類型 × 承運商品質 → 影響 RL 決策權重"
                right={<Badge variant="secondary" className="rounded-full">One-Hot Encoding</Badge>}
              />
            </CardHeader>
            <CardContent className="pt-4">
              <div className="space-y-3 text-sm">
                <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
                  <div className="font-semibold text-blue-900 mb-1">貨品類型向量</div>
                  <div className="text-blue-700">[易腐品, 電子設備, 危險品, 散裝物資]</div>
                  <div className="text-xs text-blue-600 mt-1">範例：醫療血漿 = [1, 0, 0, 0]</div>
                </div>
                <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                  <div className="font-semibold text-emerald-900 mb-1">承運商品質向量</div>
                  <div className="text-emerald-700">[未知, 低品質, 中品質, 高品質]</div>
                  <div className="text-xs text-emerald-600 mt-1">範例：軍方認證承運商 = [0, 0, 0, 1]</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-xs text-slate-600">
                    <Shield className="w-3.5 h-3.5 inline mr-1" />
                    此上下文向量與感測器數據一併輸入 DQN，影響 AI 對「監控 / 升級 / 校正 / 同儕檢查 / 標記」的策略選擇
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="py-6 border-t-2 border-slate-200 bg-slate-50">
        <div className="mx-auto max-w-7xl px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-600">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-slate-700" />
              <span className="font-semibold">© 2025 國防 AI 應用創新競賽展示系統</span>
              <Separator orientation="vertical" className="h-4" />
              <span>軍民通用 Dual-Use 技術驗證</span>
            </div>
            <div className="flex items-center gap-4">
              <span className="inline-flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-600" /> 
                <span className="font-medium">安全優先</span>
              </span>
              <Separator orientation="vertical" className="h-4" />
              <span className="inline-flex items-center gap-1">
                <Users className="w-3.5 h-3.5 text-blue-600" /> 
                <span className="font-medium">人在迴路 (Human-in-the-Loop)</span>
              </span>
              <Separator orientation="vertical" className="h-4" />
              <span className="inline-flex items-center gap-1">
                <Zap className="w-3.5 h-3.5 text-emerald-600" /> 
                <span className="font-medium">RL 持續學習</span>
              </span>
              <Separator orientation="vertical" className="h-4" />
              <span className="inline-flex items-center gap-1">
                <Shield className="w-3.5 h-3.5 text-purple-600" /> 
                <span className="font-medium">稽核軌跡可追蹤</span>
              </span>
            </div>
          </div>
          <div className="mt-4 text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-slate-800 to-slate-700 text-white text-xs font-medium shadow-md">
              <Rocket className="w-4 h-4" />
              <span>技術堆疊: Kalman Filter + Isolation Forest + DQN/SAC + LangChain/LangGraph</span>
              <ChevronRight className="w-3.5 h-3.5" />
              <span>合作夥伴: ITRI, Arviem</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}