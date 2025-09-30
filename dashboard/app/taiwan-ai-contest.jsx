import React from "react";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertTriangle, Activity, Radar, MapPin, Rocket, Settings2, LineChart, Play, Pause, RefreshCw, ChevronRight, Shield, Bell, Search } from "lucide-react";

// ————————————————————————————————————————————————
// Mock data
// ————————————————————————————————————————————————
const UAVS = [
  { id: "UAV-01", status: "ok", battery: 82, eta: "05:42", pos: [12, 32] },
  { id: "UAV-02", status: "warn", battery: 58, eta: "07:10", pos: [48, 60] },
  { id: "UAV-03", status: "alert", battery: 35, eta: "03:18", pos: [78, 22] },
  { id: "UAV-04", status: "ok", battery: 91, eta: "09:03", pos: [26, 75] },
];

const ANOMALIES = [
  { id: "TRK-3421", type: "溫度跳變", score: 0.82, rl: "Escalate", peer: "待審" },
  { id: "TRK-5528", type: "GPS 漂移", score: 0.44, rl: "Peer Check", peer: "OK" },
  { id: "TRK-1189", type: "壓力異常", score: 0.71, rl: "Calibrate", peer: "待審" },
  { id: "TRK-7760", type: "電量異常", score: 0.63, rl: "Monitor", peer: "OK" },
];

const KPI = {
  falseAlarm: 0.8, // %
  missedAlarm: 0.6, // %
  mttr: 40, // % down
  opex: 27, // % down
  resilience: 30, // % up
  downtime: 20, // % down
};

// 色彩對照
const statusColor = (s) => ({ ok: "bg-emerald-500", warn: "bg-amber-400", alert: "bg-rose-500" }[s] || "bg-slate-400");
const statusText = (s) => ({ ok: "正常", warn: "不確定", alert: "異常" }[s] || s);

// ————————————————————————————————————————————————
// Subcomponents
// ————————————————————————————————————————————————
function SectionHeader({ icon: Icon, title, desc, right }) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-xl bg-slate-100"><Icon className="w-5 h-5" /></div>
        <div>
          <h3 className="text-base font-semibold leading-tight">{title}</h3>
          {desc && <p className="text-xs text-slate-500 mt-0.5">{desc}</p>}
        </div>
      </div>
      <div>{right}</div>
    </div>
  );
}

function KpiCard({ label, value, suffix = "%", trend = "down" }) {
  const icon = trend === "up" ? "▲" : "▼";
  const color = trend === "up" ? "text-emerald-600" : "text-rose-600";
  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardDescription className="text-xs">{label}</CardDescription>
        <CardTitle className="text-2xl tracking-tight">
          <span className={color}>{icon}</span> {value}
          <span className="text-base text-slate-500 ml-1">{suffix}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Progress value={Math.min(100, Math.abs(value))} className="h-2" />
      </CardContent>
    </Card>
  );
}

function MiniLegend() {
  return (
    <div className="flex items-center gap-3 text-xs">
      <div className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500" />正常</div>
      <div className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-400" />不確定</div>
      <div className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full bg-rose-500" />異常</div>
    </div>
  );
}

// UAV map placeholder (純前端動畫模擬)
function UavHeatmap() {
  return (
    <div className="relative w-full h-64 rounded-2xl bg-gradient-to-br from-slate-50 via-white to-slate-100 border border-slate-200 overflow-hidden">
      {/* grid */}
      <div className="absolute inset-0 opacity-50 [background-image:linear-gradient(#e5e7eb_1px,transparent_1px),linear-gradient(90deg,#e5e7eb_1px,transparent_1px)] bg-[length:24px_24px]" />
      {UAVS.map((uav, i) => (
        <motion.div
          key={uav.id}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 * i }}
          className={`absolute w-3.5 h-3.5 rounded-full ${statusColor(uav.status)} shadow`}
          style={{ left: `${uav.pos[0]}%`, top: `${uav.pos[1]}%` }}
          title={`${uav.id}｜電量 ${uav.battery}%｜ETA ${uav.eta}`}
        />
      ))}
      <div className="absolute bottom-2 left-3">
        <MiniLegend />
      </div>
    </div>
  );
}

function AnomalyTable() {
  return (
    <div className="rounded-xl border border-slate-200 overflow-hidden">
      <div className="grid grid-cols-12 bg-slate-50 text-slate-600 text-xs font-medium">
        <div className="col-span-3 px-3 py-2">裝置 ID</div>
        <div className="col-span-3 px-3 py-2">異常類型</div>
        <div className="col-span-2 px-3 py-2">異常分數</div>
        <div className="col-span-2 px-3 py-2">RL 建議</div>
        <div className="col-span-2 px-3 py-2">同儕審查</div>
      </div>
      {ANOMALIES.map((a, idx) => (
        <div key={a.id} className={`grid grid-cols-12 text-sm ${idx % 2 ? "bg-white" : "bg-slate-50/50"}`}>
          <div className="col-span-3 px-3 py-2 font-mono text-slate-700">{a.id}</div>
          <div className="col-span-3 px-3 py-2">{a.type}</div>
          <div className="col-span-2 px-3 py-2">
            <div className="flex items-center gap-2">
              <div className="w-16"><Progress value={a.score * 100} className="h-1.5" /></div>
              <span className="tabular-nums text-xs text-slate-600">{(a.score * 100).toFixed(0)}%</span>
            </div>
          </div>
          <div className="col-span-2 px-3 py-2">
            <Badge variant="secondary" className="rounded-full">{a.rl}</Badge>
          </div>
          <div className="col-span-2 px-3 py-2">
            {a.peer === "OK" ? (
              <Badge className="bg-emerald-500 hover:bg-emerald-600">OK</Badge>
            ) : (
              <Badge className="bg-amber-500 hover:bg-amber-600">待審</Badge>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function PeerReviewFlow() {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm">
        <Badge className="bg-slate-800">流程</Badge>
        <span className="text-slate-600">OK → 監控｜Not OK → 升級｜不確定 → 增加取樣</span>
      </div>
      <div className="flex items-center gap-3 text-xs text-slate-600">
        <span className="inline-flex items-center gap-1"><Shield className="w-3.5 h-3.5" /> 稽核簽章</span>
        <span className="inline-flex items-center gap-1"><Bell className="w-3.5 h-3.5" /> 通知中心</span>
        <span className="inline-flex items-center gap-1"><LineChart className="w-3.5 h-3.5" /> 追蹤 KPI</span>
      </div>
    </div>
  );
}

function DigitalTwinPanel() {
  return (
    <div className="rounded-2xl border border-slate-200 p-4 bg-white">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">UAV 模擬：GPS 偽裝攻擊</CardTitle>
            <CardDescription>RL 策略：先 Peer Check，再視情況 Escalate</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-28 rounded-xl bg-gradient-to-br from-indigo-50 to-indigo-100 border border-indigo-200 flex items-center justify-center text-indigo-700 text-sm">動畫佔位：航跡偏移 → 策略回覆</div>
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">冷鏈 IoT 模擬：溫度失控</CardTitle>
            <CardDescription>RL 策略：即刻 Escalate；並建議現場處置 SOP</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-28 rounded-xl bg-gradient-to-br from-rose-50 to-rose-100 border border-rose-200 flex items-center justify-center text-rose-700 text-sm">動畫佔位：溫度超標 → RL 升級</div>
          </CardContent>
        </Card>
      </div>
      <div className="mt-4 flex items-center gap-2">
        <Button size="sm" className="rounded-xl"><Play className="w-4 h-4 mr-1" /> 開始模擬</Button>
        <Button variant="secondary" size="sm" className="rounded-xl"><Pause className="w-4 h-4 mr-1" /> 暫停</Button>
        <Button variant="outline" size="sm" className="rounded-xl"><RefreshCw className="w-4 h-4 mr-1" /> 重置</Button>
      </div>
    </div>
  );
}

// ————————————————————————————————————————————————
// Main Component
// ————————————————————————————————————————————————
export default function DefenseAIDashboard() {
  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Top Bar */}
      <div className="sticky top-0 z-40 backdrop-blur supports-[backdrop-filter]:bg-white/70 bg-white/60 border-b">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-slate-900 text-white grid place-items-center font-bold">AI</div>
            <div className="">
              <div className="text-sm font-semibold leading-tight">國防 AI 應用創新競賽 Demo</div>
              <div className="text-xs text-slate-500 -mt-0.5">UAV × AIoT × 強化學習 × 多智能體</div>
            </div>
            <Badge variant="secondary" className="ml-2 rounded-full">Dual-Use</Badge>
          </div>
          <div className="hidden md:flex items-center gap-2 w-72">
            <div className="relative w-full">
              <Search className="absolute left-2 top-2.5 w-4 h-4 text-slate-400" />
              <Input placeholder="搜尋裝置 / UAV / 任務…" className="pl-8 rounded-xl" />
            </div>
            <Button variant="outline" className="rounded-xl">匯出報表</Button>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="mx-auto max-w-7xl px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left column */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <Card className="shadow-sm">
            <CardHeader>
              <SectionHeader icon={Radar} title="任務總覽 (Mission Overview)" desc="UAV 群集位置、任務路徑與 ETA" right={<MiniLegend />} />
            </CardHeader>
            <CardContent>
              <UavHeatmap />
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                {UAVS.map((u) => (
                  <div key={u.id} className="rounded-xl border border-slate-200 p-3 bg-white flex items-center justify-between">
                    <div>
                      <div className="text-xs text-slate-500">{u.id}</div>
                      <div className="text-sm font-medium">電量 {u.battery}%</div>
                      <div className="text-xs text-slate-500">ETA {u.eta}</div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full text-white ${statusColor(u.status)}`}>{statusText(u.status)}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader>
              <SectionHeader icon={Activity} title="異常偵測面板 (Anomaly Panel)" desc="Kalman + Isolation Forest + RL 決策" right={<Button variant="outline" size="sm" className="rounded-xl"><Settings2 className="w-4 h-4 mr-1" /> 參數</Button>} />
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="table" className="w-full">
                <TabsList className="rounded-xl">
                  <TabsTrigger value="table">清單</TabsTrigger>
                  <TabsTrigger value="charts">圖表</TabsTrigger>
                </TabsList>
                <TabsContent value="table" className="mt-4">
                  <AnomalyTable />
                  <div className="mt-3"><PeerReviewFlow /></div>
                </TabsContent>
                <TabsContent value="charts" className="mt-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="h-32 rounded-xl border border-slate-200 bg-white grid place-items-center text-slate-500 text-sm">異常分數走勢 (佔位)</div>
                    <div className="h-32 rounded-xl border border-slate-200 bg-white grid place-items-center text-slate-500 text-sm">溫度 / 壓力 / GPS (佔位)</div>
                    <div className="h-32 rounded-xl border border-slate-200 bg-white grid place-items-center text-slate-500 text-sm">RL 動作佔比 (佔位)</div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        {/* Right column */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <Card className="shadow-sm">
            <CardHeader>
              <SectionHeader icon={LineChart} title="KPI 指標" desc="以國防任務韌性與營運效益為核心" right={<Button size="sm" className="rounded-xl">更新</Button>} />
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                <KpiCard label="假警報率" value={KPI.falseAlarm} suffix="%" trend="down" />
                <KpiCard label="漏報率" value={KPI.missedAlarm} suffix="%" trend="down" />
                <KpiCard label="MTTR 降幅" value={KPI.mttr} suffix="%" trend="down" />
                <KpiCard label="OPEX 降幅" value={KPI.opex} suffix="%" trend="down" />
                <KpiCard label="任務韌性提升" value={KPI.resilience} suffix="%" trend="up" />
                <KpiCard label="UAV 停機降低" value={KPI.downtime} suffix="%" trend="down" />
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader>
              <SectionHeader icon={Rocket} title="數位分身 (Digital Twin)" desc="UAV × 冷鏈 IoT 場景模擬" right={<Button variant="outline" size="sm" className="rounded-xl">場景庫</Button>} />
            </CardHeader>
            <CardContent>
              <DigitalTwinPanel />
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader>
              <SectionHeader icon={MapPin} title="情境與上下文 (Contextual Fusion)" desc="貨品類型 × 承運商品質 → 影響 RL 決策" right={<Badge variant="secondary" className="rounded-full">One-Hot</Badge>} />
            </CardHeader>
            <CardContent>
              <div className="text-sm text-slate-700 space-y-2">
                <div>貨品類型向量： [易腐、電子、危險、散裝] → 例如：易腐 = [1,0,0,0]</div>
                <div>承運商品質向量： [UNK, LOW, MED, HIGH] → 例如：HIGH = [0,0,0,1]</div>
                <div className="text-slate-500">此上下文向量會與感測特徵一併輸入 DQN，影響 {"「監控 / 升級 / 校正 / 同儕檢查 / 標記」"} 的策略。</div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Footer */}
      <div className="py-6 border-t">
        <div className="mx-auto max-w-7xl px-4 flex items-center justify-between text-xs text-slate-500">
          <div>© 2025 Phoenix Logistics · Demo 介面僅供競賽展示 · RL/多智能體決策流程皆具稽核軌跡</div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1"><AlertTriangle className="w-3.5 h-3.5" /> 安全優先</span>
            <Separator orientation="vertical" className="h-4" />
            <span className="inline-flex items-center gap-1"><Settings2 className="w-3.5 h-3.5" /> 人在迴路</span>
            <Separator orientation="vertical" className="h-4" />
            <span className="inline-flex items-center gap-1"><ChevronRight className="w-3.5 h-3.5" /> 持續學習 (RL)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
