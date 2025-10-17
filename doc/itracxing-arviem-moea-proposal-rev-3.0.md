

**ITracXing x Arviem — AI Application Acceleration Program (12-Month Plan)**\
*Comprehensive English Version (Merged General Supply Chain Monitoring Proposal for MOEA & Arviem Review)*

---

### 1. Executive Summary

This proposal outlines a comprehensive 12-month collaboration between **ITracXing Co., Ltd.** and **Arviem AG (Switzerland)** under the **MOEA AI Application Acceleration Program (A+ Enterprise Innovation R&D Scheme)**. The initiative integrates explainable AI, reinforcement learning, and multi-agent intelligence into **real-world supply chain monitoring and risk alert systems**, in partnership with **ITRI** and **Taiwan & Amsterdam University AI Collaboration**.

**Objective:** Achieve measurable AI-driven value—at least 20% reduction in operational costs, 18% improvement in analyst efficiency, and a scalable, export-ready AI platform for multi-sector supply chain monitoring and decision automation.

**Funding Structure:** **USD 740K / NT\$ 23.68M** total (**USD 370K / NT\$ 11.84M** co-funded by MOEA; **USD 370K / NT\$ 11.84M** self-funded). This aligns with MOEA's subsidy framework (50/50 split).

**Timeline Overview:**

| Phase     | Duration | Objective                        | Key Focus                                                      |
| --------- | -------- | -------------------------------- | -------------------------------------------------------------- |
| Phase I   | 3 months | Feasibility & Prototype          | AI anomaly detection & LLM-driven alert automation             |
| Phase II  | 5 months | RL Integration & Validation      | RL optimization, ITRI & University collaboration               |
| Phase III | 4 months | Multi-Agent Deployment & Scaling | Multi-agent orchestration + Federated Learning pilot           |

---

### 2. Background & Industry Pain Points

Modern supply chain operations face increasing complexity due to distributed networks, real-time logistics requirements, and dynamic environmental risks. Sensor networks and IoT systems generate massive data volumes that often result in delayed alerts, redundant signals, and missed early warnings, directly impacting delivery reliability and financial efficiency.

**Identified Pain Points:**

- High false-alarm frequency and unverified alerts.
- Overwhelming multi-sensor data volume with inconsistent quality.
- Lack of predictive alert prioritization and risk visualization.
- Fragmented governance of cross-border data and operations.

The ITracXing–Arviem partnership addresses these challenges with an explainable AI architecture combining Kalman-based signal estimation, rule-based scoring, reinforcement learning for adaptive monitoring, and LLM-driven risk summarization for executives.

---

### 3. Phase I — AI Feasibility & Prototype (3 Months)

**Goal:** Build a lightweight, explainable AI model that detects anomalies, ranks risks, and automates alert reporting in near real-time.

**Scope:**

- **Executive AI Dashboard** with Natural Language Query Interface
- IoT Battery Performance and Reliability Monitoring
- AI-Generated "Top-3 Priority" Alerts with Explainable Risk Scoring
- Real-time Supply Chain Risk Visualization

**Approach & Technical Stack:**

- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS
- **AI/ML:** LangChain, OpenAI GPT-4o with structured output
- **Signal Processing:** Kalman Filter (npm: kalman-filter) for noise reduction
- **Analytics:** Z-score residual analysis for statistical anomaly detection
- **Data Layer:** PostgreSQL with time-series battery telemetry
- **Rule Engine:** 10+ predefined battery health rules with multi-severity alerting
 

**Key Features Implemented:**

1. **Natural Language Query System**
   - Executive chat interface: "Show me top 3 risks in Asia-Pacific"
   - Intent classification and parameter extraction
   - LangChain agent with GPT-4o for intelligent responses
   - Structured JSON output with recommendations

2. **Battery Analytics Module**
   - Kalman filtering for voltage and capacity signal smoothing
   - Z-score analysis with configurable thresholds (2σ default)
   - Real-time anomaly detection with severity classification
   - Automated action recommendations

**Deliverables:**

- Production-ready Next.js dashboard with AI assistant
- Battery analytics API with Kalman filtering and anomaly detection
- PostgreSQL database schema for IoT telemetry
- Natural language query API endpoints (/api/query)
- Executive risk visualization with interactive maps and charts
- Feasibility report and ROI validation for MOEA review

---

### 4. Phase II — RL Integration & Validation (5 Months)

**Objective:** Develop reinforcement learning (RL) agents to dynamically optimize alert classification and escalation policies.

**Focus Areas:**

- **RL-Enhanced Alert Classification:** DQN/SAC-based optimization for dynamic threshold adjustment
- **Adaptive Kalman Tuning:** Reinforcement learning to optimize Q (process noise) and R (measurement noise) parameters
- **Multi-Device Policy Learning:** Cross-device learning from IoT battery telemetry patterns
- **RL-Based Adaptive Sampling:** Train RL agents to optimize IoT device sampling rates based on battery state, network conditions, and criticality
- **Edge AI Deployment:** Lightweight RL models for on-device decision-making
- **Dashboard Integration:** Real-time RL-driven recommendations in the executive interface
- **Reward Function Design:** Balancing false positive reduction with critical alert detection and battery life extension
- Joint R&D with ITRI and Taiwan & Amsterdam University AI Collaboration for RL evaluation, federated learning, and adaptive model validation

**Technical Integration:**

- Architecture: Multi-agent orchestration with Model Context Protocol (MCP) for RL tool orchestration and decision pipelines
- IoT Firmware: Embedded C/C++ and FreeRTOS prototype implementing AI-driven adaptive sampling for power optimization

- Multi-Agent System:
  - Tool orchestration via Model Context Protocol (MCP)
  - Agent functions: getTopRisks, getRiskTrends, getRiskSummary
  - Database-backed risk repository with time-series queries

- IoT Firmware with AI-Driven Adaptive Sampling:
  - Embedded AI agent for dynamic sensor sampling rate adjustment
  - Battery-aware data collection (reduce sampling when battery < 30%)
  - Edge intelligence: local anomaly detection before cloud transmission
  - Power optimization: 40-60% battery life extension through intelligent sampling
  

- Extend battery analytics module with RL-based parameter tuning
- Integrate RL agent outputs into LangChain decision pipeline
- **Deploy RL models to IoT firmware for edge-based adaptive sampling**
- Add training data collection from dashboard user feedback
- Implement A/B testing framework for rule-based vs. RL-based alerting
- **Field testing: Compare battery life with fixed vs. adaptive sampling**

**Institutional Partners:**

- **ITRI AI Center** — RL model verification, data pipeline integration, and performance benchmarking
- **Taiwan & Amsterdam University AI Collaboration** — Research on federated optimization, multi-agent coordination, and cross-domain learning ethics

**Deliverables:**

- RL-enhanced battery monitoring module integrated into Next.js dashboard
- Adaptive alert prioritization system with 30% reduction in false alarms
- Self-tuning Kalman filter with RL-optimized parameters
- **RL-optimized IoT firmware with 40-60% battery life extension**
- Edge AI models deployed to IoT devices
- Performance comparison dashboard: Rule-based vs. RL vs. Hybrid approaches
- **Field test results: Fixed vs. Adaptive sampling comparison**
- Joint validation report co-developed with ITRI and university partners

 


---

### 5. Phase III — Multi-Agent Deployment & Scaling (4 Months)

**Objective:** Deploy a production-ready multi-agent AI platform capable of managing supply chain risk monitoring and automated decision support.

**Key Capabilities:**

- **Multi-Agent Orchestration Dashboard:** Production deployment using LangChain and Model Context Protocol (MCP)
- **Federated Learning POC:** Privacy-preserving multi-partner battery analytics without data sharing
- **Executive Command Center:** Unified dashboard integrating risk monitoring, natural language queries, and automated recommendations
- **API Gateway:** RESTful APIs for third-party integration (Arviem IoT devices, external monitoring systems)
- **Real-time Monitoring:** WebSocket-based live updates for critical alerts
- Final validation with ITRI and university partners for system resilience and compliance readiness

**Technical Architecture:**

- Next.js 14 production build with edge deployment optimization
- PostgreSQL with TimescaleDB extension for time-series optimization
- Redis caching layer for API response optimization
- Kubernetes deployment with horizontal pod autoscaling
- Authentication via NextAuth.js with role-based access control
- Monitoring: OpenTelemetry + Grafana + Prometheus stack

**Deliverables:**

- **Production AI Dashboard** deployed on cloud infrastructure (AWS/Azure/GCP)
- Multi-agent coordination system with MCP server for tool orchestration
 
- Federated learning framework with differential privacy guarantees
- **Validated 40-60% battery life extension through AI-driven adaptive sampling**
- API documentation (OpenAPI/Swagger) and integration SDKs
- Deployment guide, operations runbook, and scaling playbook
- Technical architecture document and source code repository
- User training materials and executive demo videos

---

### 6. R&D Team Description

| Role                          | Organization                                         | Expertise                                      | Responsibility                                        |
| ----------------------------- | ---------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| Principal Investigator        | ITracXing                                            | RL & Explainable AI Systems                    | Technical leadership and architecture design          |
| AI Agent Engineers            | ITracXing                                            | LangChain, MCP, multi-agent orchestration      | AI agent development and LLM integration              |
| IoT Firmware Engineer         | ITracXing                                            | Embedded C/C++, RTOS, on-device AI             | RL-based adaptive sampling firmware                   |
| Full-Stack Engineers          | ITracXing                                            | Next.js, TypeScript, FastAPI                   | Dashboard and API development                         |
| Python Backend Engineers      | ITracXing                                            | FastAPI, PostgreSQL, microservices             | Analytics API and RL integration                      |
| Data Scientists               | ITracXing                                            | Kalman filtering, Z-score analysis, RL         | Battery analytics and RL model development            |
| DevOps Engineers              | ITracXing                                            | Kubernetes, CI/CD, monitoring                  | Deployment automation and infrastructure              |
| Research Partners             | ITRI, Taiwan & Amsterdam University AI Collaboration | Model validation, federated learning           | Joint research, RL evaluation, and system testing     |
| Operations & Integration Team | Arviem                                               | Supply chain operations, IoT devices           | Field data validation, pilot execution, and feedback  |

**Governance:** Monthly steering meetings, bi-weekly progress reviews.

---

### 7. Risk Management & IP Strategy

| Risk                    | Mitigation Strategy                                                   |
| ----------------------- | --------------------------------------------------------------------- |
| Data Quality Issues     | Kalman filtering + adaptive weighting models                          |
| Model Drift             | Regular retraining validated by ITRI                                  |
| Funding Constraints     | Stage-based MOEA subsidy applications                                 |
| Data Privacy Compliance | Federated learning with encryption and access control                 |
| IP Rights               | Shared IP ownership with export licensing options for global partners |

---

### 8. Expected Outcomes

- **Technical:** 
  - 90%+ anomaly detection accuracy with Kalman-filtered signals
  - Natural language query response time < 2 seconds
  - Self-learning alert prioritization via RL agents
  - Multi-agent orchestration handling 1000+ concurrent queries
  - Real-time dashboard updates with < 500ms latency
  - **40-60% battery life extension through AI-driven adaptive sampling**
  - **Edge AI inference < 10ms latency on IoT devices**
  
- **Operational:** 
  - 20–30% reduction in false alarms through intelligent filtering
  - 40% reduction in analyst query time via natural language interface
  - 50% faster incident response through automated recommendations
  - **60% reduction in data transmission costs via intelligent sampling**
  - **70% reduction in battery replacement frequency**
  - Executive decision support with AI-generated insights
  
- **Economic:** 
  - Reduced operational costs through automation (target: 20%)
  - **Significant battery cost savings: 70% fewer replacements**
  - **Lower network costs: 60% reduced data transmission**
  - Improved supply chain uptime and reliability
  - Cost savings from predictive maintenance alerts
  - Exportable SaaS platform with recurring revenue potential
  
- **User Experience:**
  - Intuitive executive dashboard accessible without technical training
  - Mobile-responsive design for on-the-go monitoring
  - Interactive visualizations with drill-down capabilities
  - Multilingual support (English, Chinese)
  
- **Academic:** 
  - Strengthened Taiwan–EU AI collaboration ecosystem
  - Published research papers on RL-enhanced IoT monitoring
  - Open-source contributions to LangChain and MCP communities

---

### 9. Budget Summary (USD / NT\$)

| Phase     | Duration     | Budget (USD / NT\$) | Funding Split | Key Outputs                       |
| --------- | ------------ | ------------------- | ------------- | --------------------------------- |
| Phase I   | 3 months     | 200K / 6.4M         | 50/50         | Feasibility prototype             |
| Phase II  | 5 months     | 350K / 11.2M        | 50/50         | RL pilot + validation             |
| Phase III | 4 months     | 250K / 8.0M         | 50/50         | Multi-agent deployment system     |
| **Total** | **12 months** | **740K / 23.68M**    | **50/50**     | Production-ready AI platform      |

---

### 10. Detailed Budget Breakdown

**Phase I — USD 200K / NT$ 6.4M**

- Personnel (AI Agent, Full-Stack, Python Backend, Data Scientist): 95K / 3.0M
  - AI Agent Engineer: LangChain/MCP orchestration (1 engineer × 3 months)
  - Full-Stack Engineer: Next.js/TypeScript dashboard (1 engineer × 3 months)
  - Python Backend Engineer: FastAPI/PostgreSQL analytics API (1 engineer × 3 months)
  - Data Scientist: Kalman filtering, Z-score analysis (1 engineer × 3 months)
- Cloud Infrastructure & APIs: 45K / 1.4M
  - AWS/GCP hosting, PostgreSQL database, OpenAI API credits
  - Development and staging environments
- Testing & Validation: 35K / 1.1M
  - UI/UX testing, API load testing, security audits
- Travel & Collaboration: 15K / 0.5M
  - Arviem coordination meetings, ITRI partnership setup
- Contingency: 10K / 0.4M

**Phase II — USD 350K / NT$ 11.2M**

- Personnel (RL Engineers, Python Backend, Full-Stack, Data Pipeline): 165K / 5.3M
  - RL/ML Engineers: DQN/SAC with stable-baselines3 (2 engineers × 5 months)
  - Python Backend Engineer: RL API integration, model serving (1 engineer × 5 months)
  - Full-Stack Engineer: RL dashboard integration, A/B testing UI (1 engineer × 5 months)
  - Data Pipeline Engineer: Time-series optimization, ETL (1 engineer × 5 months)
- Compute & Cloud: 75K / 2.4M
  - GPU instances for RL training, increased database storage
  - Production environment scaling, CDN for global access
- Verification & Tools: 60K / 1.9M
  - RL validation framework, A/B testing infrastructure
  - Performance monitoring tools (Grafana, Prometheus)
- Collaboration (ITRI + Universities): 25K / 0.8M
  - Joint RL validation experiments, federated learning research
- Travel & Compliance: 15K / 0.5M
  - Research collaboration visits, compliance audits
- Contingency: 10K / 0.3M

**Phase III — USD 250K / NT$ 8.0M**

- Personnel (DevOps, Python Backend, Documentation): 115K / 3.7M
  - DevOps Engineers: Kubernetes, CI/CD, monitoring (2 engineers × 4 months)
  - Python Backend Engineer: Federated learning backend, API gateway (1 engineer × 4 months)
  - Technical Writer: API docs, training materials, videos (1 engineer × 4 months)
- Cloud + Production Infrastructure: 55K / 1.8M
  - Kubernetes cluster setup, load balancers, autoscaling
  - Production database (PostgreSQL with TimescaleDB)
  - CDN, monitoring, backup systems
- Simulation & Digital Twin: 30K / 1.0M
  - 3D visualization framework (Three.js/React Three Fiber)
  - Predictive simulation engine development
- Research Collaboration & Final Validation: 20K / 0.6M
  - ITRI final system validation, university partnership reporting
- Deployment, Documentation & Training: 20K / 0.6M
  - API documentation (Swagger/OpenAPI), user guides
  - Video tutorials, training sessions for Arviem team
- Contingency: 10K / 0.3M

**Total:** USD 740K / NT\$ 23.68M — 50% MOEA subsidy / 50% self-funded.

---

### 11. MOEA AI應用躍昇計畫經費需求明細表

**申請補助總金額：NT$ 11,840,000（USD 370K）**  
**自籌配合款：NT$ 11,840,000（USD 370K）**  
**計畫總經費：NT$ 23,680,000（USD 740K）**

#### 11.1 iTX公司研發人事費用（R&D Personnel Costs）

| 階段 | 職位 | 人數 | 期間 | 月薪 (NT\$) | 小計 (NT\$) | 補助額 (NT\$) | 自籌額 (NT\$) |
|------|------|------|------|------------|------------|-------------|-------------|
| **Phase I** | 全端工程師 (Full-Stack Engineers) | 2 | 3個月 | 150,000 | 900,000 | 450,000 | 450,000 |
| | 資料科學家 (Data Scientist) | 1 | 3個月 | 180,000 | 540,000 | 270,000 | 270,000 |
| | AI/ML工程師 (AI/ML Engineer) | 1 | 3個月 | 170,000 | 510,000 | 255,000 | 255,000 |
| **Phase II** | ML工程師 (ML Engineers) | 2 | 5個月 | 170,000 | 1,700,000 | 850,000 | 850,000 |
| | 全端工程師 (Full-Stack Engineers) | 2 | 5個月 | 150,000 | 1,500,000 | 750,000 | 750,000 |
| | 資料管道工程師 (Data Pipeline Engineer) | 1 | 5個月 | 160,000 | 800,000 | 400,000 | 400,000 |
| **Phase III** | DevOps工程師 (DevOps Engineers) | 2 | 4個月 | 160,000 | 1,280,000 | 640,000 | 640,000 |
| | 全端工程師 (Full-Stack Engineers) | 2 | 4個月 | 150,000 | 1,200,000 | 600,000 | 600,000 |
| | 技術文件工程師 (Technical Writer) | 1 | 4個月 | 120,000 | 480,000 | 240,000 | 240,000 |
| **小計** | | | | | **8,910,000** | **4,455,000** | **4,455,000** |

#### 11.2 添購研發設備（R&D Equipment Purchase）

| 項目 | 規格/用途 | 數量 | 單價 (NT\$) | 小計 (NT\$) | 補助額 (NT\$) | 自籌額 (NT\$) |
|------|----------|------|------------|------------|-------------|-------------|
| 高效能工作站 | Intel i9/64GB RAM/RTX 4090, AI模型開發 | 3 | 180,000 | 540,000 | 270,000 | 270,000 |
| 開發筆電 | MacBook Pro M3 Max, 前端開發 | 5 | 120,000 | 600,000 | 300,000 | 300,000 |
| 測試伺服器 | Dell PowerEdge R750, 整合測試環境 | 1 | 350,000 | 350,000 | 175,000 | 175,000 |
| 網路設備 | 10GbE交換器、防火牆 | 1套 | 150,000 | 150,000 | 75,000 | 75,000 |
| IoT測試設備 | GPS追蹤器、感測器套件（模擬Arviem設備） | 1套 | 200,000 | 200,000 | 100,000 | 100,000 |
| 顯示設備 | 4K顯示器（開發與展示用） | 8 | 25,000 | 200,000 | 100,000 | 100,000 |
| **小計** | | | | **2,040,000** | **1,020,000** | **1,020,000** |

#### 11.3 研發設備維修費（Equipment Maintenance）

| 項目 | 說明 | 期間 | 費用 (NT\$) | 補助額 (NT\$) | 自籌額 (NT\$) |
|------|------|------|------------|-------------|-------------|
| 硬體維護合約 | 伺服器、工作站年度維護 | 12個月 | 120,000 | 60,000 | 60,000 |
| 設備保險 | 全部研發設備保險 | 12個月 | 80,000 | 40,000 | 40,000 |
| **小計** | | | **200,000** | **100,000** | **100,000** |

#### 11.4 無形資產及驗證費用（Intangible Assets & Validation）

| 項目 | 說明 | 數量/期間 | 費用 (NT\$) | 補助額 (NT\$) | 自籌額 (NT\$) |
|------|------|----------|------------|-------------|-------------|
| **軟體授權** | | | | | |
| OpenAI API額度 | GPT-4o API使用費（預估300M tokens） | 12個月 | 960,000 | 480,000 | 480,000 |
| GitHub Enterprise | 原始碼管理與協作 | 12個月 | 120,000 | 60,000 | 60,000 |
| JetBrains全系列 | IDE授權（WebStorm, PyCharm等） | 10授權×12個月 | 150,000 | 75,000 | 75,000 |
| Figma Enterprise | UI/UX設計協作 | 12個月 | 90,000 | 45,000 | 45,000 |
| **驗證與認證** | | | | | |
| 資安滲透測試 | 第三方資安驗證（2次） | 2次 | 400,000 | 200,000 | 200,000 |
| ISO 27001顧問 | 資訊安全管理系統導入 | 服務 | 300,000 | 150,000 | 150,000 |
| 效能負載測試 | 第三方效能驗證與壓力測試 | 2次 | 200,000 | 100,000 | 100,000 |
| AI模型驗證 | 與ITRI聯合驗證實驗費用 | 服務 | 500,000 | 250,000 | 250,000 |
| 法規合規諮詢 | GDPR、PDPA合規顧問 | 服務 | 280,000 | 140,000 | 140,000 |
| **小計** | | | **3,000,000** | **1,500,000** | **1,500,000** |

#### 11.5 消耗性器材及原材料（Consumables & Materials）

| 項目 | 說明 | 期間/數量 | 費用 (NT\$) | 補助額 (NT\$) | 自籌額 (NT\$) |
|------|------|----------|------------|-------------|-------------|
| **雲端服務** | | | | | |
| AWS/GCP雲端 | EC2/Compute Engine實例、S3/Cloud Storage | 12個月 | 2,400,000 | 1,200,000 | 1,200,000 |
| GPU運算 | RL訓練用GPU實例（NVIDIA A100） | Phase II-III (9個月) | 1,800,000 | 900,000 | 900,000 |
| 資料庫服務 | PostgreSQL RDS + TimescaleDB | 12個月 | 600,000 | 300,000 | 300,000 |
| CDN服務 | CloudFront/Cloud CDN全球加速 | 12個月 | 360,000 | 180,000 | 180,000 |
| 監控服務 | Grafana Cloud, Datadog, Sentry | 12個月 | 240,000 | 120,000 | 120,000 |
| **辦公耗材** | | | | | |
| 文具、列印耗材 | 日常辦公用品 | 12個月 | 60,000 | 30,000 | 30,000 |
| 電力與網路 | 研發團隊辦公室電費、網路費 | 12個月 | 180,000 | 90,000 | 90,000 |
| **測試資料** | | | | | |
| 模擬資料集 | 供應鏈IoT模擬資料採購 | 1次 | 150,000 | 75,000 | 75,000 |
| **小計** | | | **5,790,000** | **2,895,000** | **2,895,000** |

#### 11.6 專利申請費（Patent Application）

| 項目 | 說明 | 數量 | 費用 (NT\$) | 補助額 (NT\$) | 自籌額 (NT\$) |
|------|------|------|------------|-------------|-------------|
| 台灣發明專利申請 | "多智能體IoT監控系統"、"自適應Kalman濾波方法" | 2件 | 150,000 | 75,000 | 75,000 |
| 美國發明專利申請 | Multi-Agent IoT Monitoring System | 1件 | 350,000 | 175,000 | 175,000 |
| 歐盟發明專利申請 | Adaptive Kalman Filtering for IoT | 1件 | 400,000 | 200,000 | 200,000 |
| 專利代理人費用 | 專利撰寫、答辯服務 | 服務 | 300,000 | 150,000 | 150,000 |
| **小計** | | | **1,200,000** | **600,000** | **600,000** |

#### 11.7 國內差旅費（Domestic Travel）

| 項目 | 說明 | 次數/人次 | 費用 (NT\$) | 補助額 (NT\$) | 自籌額 (NT\$) |
|------|------|----------|------------|-------------|-------------|
| 台北-新竹（ITRI訪問） | 與工研院AI中心協作會議 | 20人次 | 60,000 | 30,000 | 30,000 |
| 台北-台南/高雄 | 南部合作夥伴、場域驗證 | 12人次 | 90,000 | 45,000 | 45,000 |
| 住宿費 | 跨縣市會議住宿 | 15晚 | 60,000 | 30,000 | 30,000 |
| 交通租車 | 場域測試交通 | 10天 | 50,000 | 25,000 | 25,000 |
| 餐費（協作會議） | 與ITRI、大學合作餐敘 | 30場次 | 90,000 | 45,000 | 45,000 |
| 研討會註冊費 | 台灣AI年會、TAAI會議 | 5人次 | 50,000 | 25,000 | 25,000 |
| **小計** | | | **400,000** | **200,000** | **200,000** |

---

#### 11.8 經費總表（Total Budget Summary）

| 項目 | 總金額 (NT\$) | 申請補助額 (NT\$) | 自籌配合款 (NT\$) | 補助比例 |
|------|-------------|----------------|----------------|---------|
| 1. 研發人事費用 | 8,910,000 | 4,455,000 | 4,455,000 | 50% |
| 2. 研發設備購置 | 2,040,000 | 1,020,000 | 1,020,000 | 50% |
| 3. 設備維修費用 | 200,000 | 100,000 | 100,000 | 50% |
| 4. 無形資產及驗證 | 3,000,000 | 1,500,000 | 1,500,000 | 50% |
| 5. 消耗性器材及原材料 | 5,790,000 | 2,895,000 | 2,895,000 | 50% |
| 6. 專利申請費 | 1,200,000 | 600,000 | 600,000 | 50% |
| 7. 國內差旅費 | 400,000 | 200,000 | 200,000 | 50% |
| **總計** | **21,540,000** | **10,770,000** | **10,770,000** | **50%** |

**註：**
- 以上經費依據MOEA AI應用躍昇計畫補助要點編列
- 補助比例為50%，符合A+企業創新研發淬鍊計畫規範
- 匯率以1 USD = 32 TWD計算（實際以申請時匯率為準）
- 人事費用依據台灣AI產業薪資行情編列
- 雲端服務費用依據AWS/GCP定價與預估使用量計算
- 專利費用包含申請費、代理費及可能的答辯費用

---

### 12. Contact

**ITracXing Co., Ltd.**\
Email: [contact@itracxing.com](mailto\:contact@itracxing.com)\
**Arviem AG** — Strategic Partner for AI Supply Chain Deployment\
**Research Partners:** ITRI, Taiwan & Amsterdam University AI Collaboration

---

*Prepared by ITracXing Co., Ltd., October 2025 — Final General Supply Chain Proposal for MOEA & Arviem CEO Review.*

---

# 中文版本

**ITracXing x Arviem — AI應用躍昇計畫（12個月計畫）**\
*中文版（供MOEA與Arviem高層審閱）*

---

### 一、計畫摘要

本計畫由 **ITracXing Co., Ltd.** 與 **Arviem AG（瑞士）** 共同執行，依據 **經濟部A+企業創新研發淬鍊計畫—AI應用躍昇計畫** 推動，目標在於將可解釋式AI、強化學習（RL）與多智能體技術應用於**供應鏈監控與風險預警系統**，並與 **工研院（ITRI）** 及 **台灣暨阿姆斯特丹大學AI合作中心** 共同研發。

**計畫目標：** 以AI技術實現供應鏈營運智慧化，達成營運成本降低20%、分析效率提升18%、並打造具出口潛力之AI供應鏈決策平台。

**經費結構：** 總經費 **USD 740K / NT\$ 23.68M**（補助 **USD 370K / NT\$ 11.84M**；自籌 **USD 370K / NT\$ 11.84M**），符合MOEA補助架構（50/50分攤）。

**期程概述：**

| 階段        | 期間   | 目標          | 核心重點              |
| --------- | ---- | ----------- | ----------------- |
| Phase I   | 3 個月 | 可行性驗證與原型    | AI異常偵測與LLM警示自動化   |
| Phase II  | 5 個月 | RL整合與驗證     | RL模型優化、工研院與大學合作驗證 |
| Phase III | 4 個月 | 多智能體部署與擴展 | 多智能體協作與聯邦學習試驗     |

---

### 二、產業背景與痛點

現代供應鏈系統跨足全球，面臨即時監控與資料爆量挑戰：

- 警示過多、誤報頻繁。
- 多感測資料品質不一、難以即時反應。
- 缺乏預測性風險排序與警示可視化。
- 跨國資料治理不一致，合規風險高。

本計畫結合 **Kalman濾波、規則引擎、強化學習與LLM摘要報告**，構建可解釋、可自學的AI供應鏈監控架構，協助高層快速掌握營運風險與決策依據。

---

### 三、第一階段：AI可行性驗證與原型（3個月）

**目標：** 建立可解釋式AI模型，實現即時異常偵測、風險排序與報告自動化。

**工作內容：**

- **高層AI管理儀表板** 具自然語言查詢介面
- IoT設備電池效能與可靠性監控
- 生成式AI風險報告與可解釋性風險評分（前三名優先警示）
- 即時供應鏈風險視覺化

**方法與技術架構：**

- **前端框架：** Next.js 14、React 18、TypeScript、Tailwind CSS
- **AI/ML：** LangChain、OpenAI GPT-4o結構化輸出
- **訊號處理：** Kalman濾波器（npm: kalman-filter）降噪
- **分析技術：** Z-score殘差分析進行統計異常偵測
- **資料層：** PostgreSQL時序電池遙測資料庫
- **架構：** 多智能體協作與模型上下文協定（MCP）
- **規則引擎：** 10+電池健康規則與多級別警示系統

**已實現的關鍵功能：**

1. **自然語言查詢系統**
   - 高層聊天介面：「顯示亞太地區前三名風險」
   - 意圖分類與參數提取
   - LangChain代理與GPT-4o智能回應
   - 結構化JSON輸出與建議

2. **電池分析模組**
   - Kalman濾波電壓與容量訊號平滑
   - Z-score分析與可調閥值（預設2σ）
   - 即時異常偵測與嚴重性分類
   - 自動化行動建議

3. **多智能體系統**
   - 透過模型上下文協定進行工具協作
   - 代理功能：getTopRisks、getRiskTrends、getRiskSummary
   - 資料庫支援的風險儲存庫與時序查詢

**成果：**

- 生產級Next.js儀表板與AI助手
- 電池分析API含Kalman濾波與異常偵測
- PostgreSQL IoT遙測資料庫架構
- 自然語言查詢API端點（/api/query）
- 高層風險視覺化含互動式地圖與圖表
- 可行性驗證與ROI報告提交MOEA

---

### 四、第二階段：RL整合與驗證（5個月）

**目標：** 建立RL代理模型，動態優化警示分類與回應策略。

**重點項目：**

- **RL增強警示分類：** DQN/SAC演算法優化動態閥值調整
- **自適應Kalman調校：** 強化學習優化Q（過程噪聲）和R（測量噪聲）參數
- **多設備策略學習：** 從IoT電池遙測模式進行跨設備學習
- **儀表板整合：** 高層介面中的即時RL驅動建議
- **獎勵函數設計：** 平衡誤報減少與關鍵警示偵測
- **與ITRI及台灣暨阿姆斯特丹大學AI合作中心聯合開發RL與聯邦學習模型**

**技術整合：**

- 擴展電池分析模組含RL參數調校
- 整合RL代理輸出至LangChain決策管道
- 從儀表板使用者回饋收集訓練資料
- 實現規則式vs. RL式警示的A/B測試框架

**合作單位：**

- **ITRI AI中心**：RL模型驗證、資料管道整合與效能基準測試
- **台灣暨阿姆斯特丹大學AI合作中心**：聯邦優化、多智能體協作與演算法倫理研究

**成果：**

- RL增強電池監控模組整合至Next.js儀表板
- 自適應警示優先級系統，誤報減少30%
- 具RL優化參數的自調整Kalman濾波器
- 效能比較儀表板：規則式 vs. RL式 vs. 混合式
- 聯合驗證報告提交MOEA

---

### 五、第三階段：多智能體部署與擴展（4個月）

**目標：** 部署生產級多智能體AI平台，實現供應鏈風險監控與自動化決策支援。

**主要技術：**

- **多智能體協作儀表板：** 使用LangChain與模型上下文協定（MCP）的生產部署
- **聯邦學習POC：** 保護隱私的多夥伴電池分析，無需資料共享
- **高層指揮中心：** 整合風險監控、自然語言查詢與自動化建議的統一儀表板
- **API閘道：** 用於第三方整合的RESTful APIs（Arviem IoT設備、外部監控系統）
- **即時監控：** 基於WebSocket的關鍵警示即時更新
- **ITRI與學研單位進行最終系統驗證與合規準備**

**技術架構：**

- Next.js 14生產構建與邊緣部署優化
- PostgreSQL含TimescaleDB擴展進行時序優化
- Redis快取層進行API回應優化
- Kubernetes部署與水平Pod自動擴展
- 透過NextAuth.js的身份驗證與角色權限控制
- 監控：OpenTelemetry + Grafana + Prometheus堆疊

**成果：**

- **生產級AI儀表板** 部署於雲端基礎設施（AWS/Azure/GCP）
- 多智能體協作系統含MCP伺服器進行工具協作
- 聯邦學習框架含差分隱私保證
- API文件（OpenAPI/Swagger）與整合SDK
- 部署指南、操作手冊與擴展策略書
- 技術架構文件與原始碼儲存庫
- 使用者培訓材料與高層示範影片

---

### 六、研發團隊說明

| 角色        | 單位              | 專長                   | 職責                  |
| --------- | --------------- | -------------------- | ------------------- |
| 計畫主持      | ITracXing       | 強化學習與可解釋AI系統         | 架構設計與技術總管           |
| AI/ML工程師  | ITracXing       | LangChain、OpenAI GPT-4、異常偵測 | AI代理開發與模型整合         |
| 全端工程師     | ITracXing       | Next.js、TypeScript、PostgreSQL | 儀表板開發與API實現         |
| 資料科學家     | ITracXing       | Kalman濾波、Z-score分析、時序   | 電池分析與訊號處理           |
| 前端工程師     | ITracXing       | React、Tailwind CSS、資料視覺化 | UI/UX設計與互動元件         |
| DevOps工程師 | ITracXing       | Kubernetes、CI/CD、監控   | 部署自動化與基礎設施          |
| 研究合作      | ITRI、台灣暨阿姆斯特丹大學 | 模型驗證與聯邦學習            | 聯合研發、RL評估與系統測試      |
| 營運與整合團隊   | Arviem          | 供應鏈營運、IoT設備          | 場域資料驗證、試驗執行與回饋      |

治理機制：每月召開指導委員會、雙週技術進度檢討會。

---

### 七、風險管理與智慧財產策略

| 風險     | 對策                                 |
| ------ | ---------------------------------- |
| 資料品質問題 | Kalman過濾與權重調整模型                    |
| 模型漂移   | 由ITRI定期再訓練與驗證                      |
| 經費限制   | 採階段式申請MOEA補助策略                     |
| 隱私合規   | 聯邦學習與加密資料交換                        |
| 智慧財產權  | 由ITracXing、Arviem與合作單位共同持有，具出口授權選項 |

---

### 八、預期效益

- **技術面：** 
  - 具Kalman濾波訊號的異常偵測準確率達90%以上
  - 自然語言查詢回應時間 < 2秒
  - 透過RL代理實現自學能力的警示優先級排序
  - 多智能體協作處理1000+併發查詢
  - 即時儀表板更新延遲 < 500毫秒
  
- **營運面：** 
  - 透過智能過濾，誤報率降低20–30%
  - 透過自然語言介面，分析師查詢時間減少40%
  - 透過自動化建議，事件回應速度加快50%
  - 提供AI生成洞察的高層決策支援
  
- **經濟面：** 
  - 透過自動化降低營運成本（目標：20%）
  - 提升供應鏈運作穩定性與可靠性
  - 預測性維護警示帶來的成本節省
  - 具出口潛力的SaaS平台與經常性收入
  
- **使用者體驗：**
  - 直觀的高層儀表板，無需技術培訓即可使用
  - 行動響應式設計，支援隨時監控
  - 互動式視覺化與深入分析功能
  - 多語言支援（英文、中文）
  
- **學研面：** 
  - 強化台歐AI合作生態系
  - 發表RL增強IoT監控研究論文
  - 對LangChain與MCP社群的開源貢獻

---

### 九、經費摘要（USD / NT\$）

| 階段        | 期間       | 預算 (USD / NT\$) | 補助比例    | 主要成果       |
| --------- | -------- | --------------- | ------- | ---------- |
| Phase I   | 3 個月     | 200K / 6.4M     | 50/50   | 可行性原型      |
| Phase II  | 5 個月     | 350K / 11.2M    | 50/50   | RL試驗與驗證    |
| Phase III | 4 個月     | 250K / 8.0M     | 50/50   | 多智能體部署系統   |
| **合計**    | **12 個月** | **740K / 23.68M** | **50/50** | 生產級AI平台 |

---

### 十、詳細經費配置

**Phase I — USD 200K / NT\$ 6.4M**\
人事費（AI智能體、全端、Python後端、資料科學家）：95K / 3.0M
  - AI智能體工程師：LangChain/MCP協作（1位工程師 × 3個月）
  - 全端工程師：Next.js/TypeScript儀表板（1位工程師 × 3個月）
  - Python後端工程師：FastAPI/PostgreSQL分析API（1位工程師 × 3個月）
  
  - 資料科學家：Kalman濾波、Z-score分析（1位工程師 × 3個月）\
雲端基礎設施與API：45K / 1.4M
  - AWS/GCP託管、PostgreSQL資料庫、OpenAI API額度
  - 開發與測試環境\
驗證測試：35K / 1.1M
  - UI/UX測試、API負載測試、安全稽核\
差旅與協作：15K / 0.5M
  - Arviem協調會議、ITRI夥伴關係建立\
預備金：10K / 0.4M

**Phase II — USD 350K / NT\$ 11.2M**\
人事費（RL工程師、Python後端、全端、資料管道）：165K / 5.3M
  - RL/ML工程師：DQN/SAC與stable-baselines3（2位工程師 × 5個月）
  - Python後端工程師：RL API整合、模型服務（1位工程師 × 5個月）
  - 全端工程師：RL儀表板整合、A/B測試UI（1位工程師 × 5個月）
  - IoT韌體工程師：RL自適應採樣（AI邊緣決策）（1位工程師 × 5個月）
  
  - 資料管道工程師：時序優化、ETL（1位工程師 × 5個月）\
運算與雲端：75K / 2.4M
  - RL訓練GPU實例、增加資料庫儲存
  - 生產環境擴展、全球存取CDN\
驗證與工具：60K / 1.9M
  - RL驗證框架、A/B測試基礎設施
  - 效能監控工具（Grafana、Prometheus）\
合作研究（ITRI + 大學）：25K / 0.8M
  - 聯合RL驗證實驗、聯邦學習研究\
差旅與合規：15K / 0.5M
  - 研究協作訪問、合規稽核\
預備金：10K / 0.3M

**Phase III — USD 250K / NT$ 8.0M**\
人事費（DevOps、Python後端、文件）：115K / 3.7M
  - DevOps工程師：Kubernetes、CI/CD、監控（2位工程師 × 4個月）
  - Python後端工程師：聯邦學習後端、API閘道（1位工程師 × 4個月）
  - 技術文件工程師：API文件、培訓材料、影片（1位工程師 × 4個月）\
雲端與生產基礎設施：55K / 1.8M
  - Kubernetes叢集設置、負載平衡器、自動擴展
  - 生產資料庫（PostgreSQL含TimescaleDB）
  - CDN、監控、備份系統\
模擬與數位分身：30K / 1.0M
  - 3D視覺化框架（Three.js/React Three Fiber）
  - 預測模擬引擎開發\
研究合作與最終驗證：20K / 0.6M
  - ITRI最終系統驗證、大學夥伴關係報告\
部署、文件與培訓：20K / 0.6M
  - API文件（Swagger/OpenAPI）、使用者指南
  - 影片教程、Arviem團隊培訓課程\
預備金：10K / 0.3M

**總計：USD 740K / NT\$ 23.68M — 50% MOEA補助 / 50% 自籌。**

---

<!-- Removed duplicate Chinese budget section to avoid conflicting totals; authoritative tables remain above with NT$ 23,680,000 total and 50/50 split. -->
