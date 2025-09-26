# AIoT Multi-Agent Logistics Monitoring Platform (Supply Chain Monitoring as a Service)

## System Overview

This document describes an AIoT-based monitoring platform tailored for supply‑chain and logistics operations. It focuses on real‑time telemetry, operational anomaly detection, automated orchestration, and human oversight — delivered as Monitoring as a Service (MaaS) for logistics customers.

## Purpose and Scope

- Purpose: provide continuous visibility, anomaly detection, and automated operational responses for assets in transit.
- Scope: telemetry processing, feature extraction, ML anomaly scoring, contextual fusion (cargo type, forwarder quality), decision automation, and human escalation. This is an operational monitoring and logistics command capability.

## Logistics Operations Monitoring as a Service (MaaS)

- Primary focus is physical and operational telemetry (temperature, geolocation, battery percentage) and asset health
- Actions (increase sampling, calibrate, peer check, flag, escalate to operations) address data quality and operational continuity rather than malware, network intrusions, or forensics.
- The RL decision engine and multi‑agent orchestration automate operational workflows and triage; human‑in‑the‑loop handles peer review and escalations.


## Key Components (summary)

- Data Ingestion: IoT sensors and Kalman filters for denoising and state estimation. The platform collects per-asset telemetry and derived quality/health metrics, for example:
  - `temp_sla_violation`, `temp_jump_rate` — temperature SLA violations and sudden temperature changes
  - `press_residual_proxy`, `pressure_jump_rate` — pressure residual proxies and frequency of pressure jumps
  - `route_corridor_dev_km` — deviation from planned route in kilometers (location provided by router)
  - `speed_spike_rate`, `accel_spike_rate` — sudden speed or acceleration events that may indicate incidents
  - `ts_jitter_sec`, `non_monotonic_ts_rate`, `missing_frac` — timestamp jitter, out-of-order timestamps, and fraction of missing samples (data quality indicators)
  - `battery_pct`, `cal_age_hours` — battery percentage and sensor calibration age (device health)
  - `router_location_quality` — location accuracy and update frequency from router
- Processing Pipeline: feature engineering, anomaly scoring (e.g., Isolation Forest), and context fusion.
- Forwarder Quality Assessment: continuous evaluation of freight forwarder performance per forwarder × lane × cargo combination, with quality bucketization (UNK/LOW/MED/HIGH) feeding into decision logic.
- Decision Engine: RL-based action selection with safety overrides and policy constraints.
- Action & Orchestration: automated actions (monitor, increase sampling, calibrate, peer check, escalate, flag) coordinated by agents with human oversight.
- Operations Dashboard: alerts, tickets, and visualization tailored to logistics operators.

## Context Integration

As shown in the architecture diagram, the contextual fusion layer feeds additional context into the RL state builder:

- **Cargo Type**: Influences decision thresholds (perishable, electronics, hazardous, bulk)
- **Forwarder Quality**: UNK/LOW/MED/HIGH quality buckets affecting trust levels

This context helps the DQN make more informed decisions based on business logic and operational constraints, while the core state space focuses on the technical telemetry and anomaly detection features.

## Data Flow

1. **Ingestion**: IoT sensors collect data → Kalman filters process signals
2. **Analysis**: Feature engineering → ML anomaly detection → Context integration (cargo type, forwarder quality)
3. **Decision**: RL decision engine (with safety overrides) selects appropriate actions
4. **Execution**: Multi-agent system executes actions and coordinates responses
5. **Output**: Results displayed on the Operations Dashboard
6. **Learning**: System continuously learns from actions to improve future decisions

## Key Features

- **Real-time Processing**: Continuous monitoring and immediate response capabilities
- **Safety-First Design**: Safety overrides ensure critical decisions aren't fully automated
- **Human Oversight**: Human-in-the-loop design for escalations and peer reviews
- **Adaptive Learning**: RL training loop enables the system to improve over time
- **Multi-Modal Integration**: Combines sensor data with contextual business information

This architecture represents a modern approach to IoT operational monitoring that balances automation with human oversight, using advanced AI techniques while maintaining safety and explainability through multi-agent coordination.

## Peer Review Decision Flows

The diagram shows three key outcomes from human peer review:

1. **Review OK** → **Monitor Action**: When peer review validates the anomaly as a false positive or resolved issue:
   - Tracker returns to baseline monitoring mode
   - Trust/confidence score increases for the device
   - Event labeled as "verified_normal" for training data
   - Status updates flow to Operations Dashboard for visibility

2. **Review NOT OK** → **Escalate Action**: When peer review confirms a genuine anomaly or operational issue:
   - High-priority alert generated on Operations Dashboard
   - Immediate notification to logistics operators and on-call staff
   - Safety policies may trigger additional protective actions
   - Event labeled as "verified_anomaly" for model improvement

3. **Review Uncertain** → **Increase Sampling Action**: When peer review needs more data for assessment:
   - Temporarily boost sensor data collection frequency
   - Gather additional telemetry for better root cause analysis
   - Set timeout for human decision with safe defaults
   - Route to human escalation queue if uncertainty persists

## Cargo Type Integration Scenarios

Cargo type information flows through Context Fusion and influences decisions at multiple points:

### Decision Engine Impact
- **Perishable goods**: Lower temperature thresholds, faster escalation on temp_sla_violation
- **Electronics**: Tighter shock/acceleration limits, immediate response to accel_spike_rate
- **Hazardous materials**: Strictest safety overrides, mandatory human approval for certain actions
- **Bulk commodities**: Relaxed thresholds, prefer monitoring over costly escalations

### Peer Review Context  
- Reviewers see cargo type in their assessment interface
- Cargo-specific checklists and inspection criteria
- Different escalation priorities based on cargo value and risk

### Action Selection
- High-value cargo → prefer Escalate over Monitor for uncertain cases
- Temperature-sensitive cargo → automatic Increase Sampling on temperature anomalies
- Fragile cargo → immediate Calibrate action on pressure/vibration issues

## Freight Forwarder Quality Integration

The system maintains continuous assessment of freight forwarder performance through quality snapshots that track performance across multiple dimensions:

### Quality Snapshot Metrics
- **Per-forwarder performance**: Historical reliability, on-time delivery, damage rates
- **Lane-specific performance**: Route efficiency, typical transit times, incident frequency
- **Cargo-type expertise**: Specialized handling capabilities, temperature control effectiveness

### Quality Bucketization
Forwarder quality is categorized into discrete levels:
- **UNK**: Unknown or insufficient data for assessment
- **LOW**: Below-average performance, frequent issues or delays
- **MED**: Standard performance meeting basic requirements
- **HIGH**: Excellent performance, reliable and efficient operations

### Decision Engine Impact
- **LOW quality forwarders**: More aggressive monitoring, faster escalation thresholds
- **HIGH quality forwarders**: Extended monitoring intervals, higher anomaly thresholds
- **UNK quality forwarders**: Baseline monitoring with moderate sensitivity

### Continuous Learning Loop
- Outcome logging feeds back into forwarder quality assessments
- KPI updates refine quality bucketization over time
- Reward signals help adjust forwarder performance rankings

## Architecture Diagram


```mermaid
flowchart TD
  %% Data Ingestion Layer
  subgraph INGESTION["📡 Data Ingestion"]
    direction TB
    SENSORS["🌡️ IoT Sensors<br/>Temp, Pressure, battery percentage"]:::sensor
    FILTER["🔄 Kalman Filters<br/>Signal Processing"]:::filter
    SENSORS --> FILTER
  end
  
  %% Feature Processing
  FEATURES["📋 Feature Engineering<br/>Residuals & Metrics"]:::feature
  
  %% AI/ML Layer  
  subgraph ML["🤖 ML Detection"]
    direction LR
    ANOMALY["🌲 Isolation Forest<br/>Anomaly Scoring"]:::ml
    STATES["🎯 State Classification<br/>Normal → Critical"]:::states
    ANOMALY --> STATES
  end
  
  %% Forwarder Quality Assessment
  subgraph FORWARDER["🚚 Forwarder Quality"]
    direction TB
    SNAPSHOT["📊 Quality Snapshot<br/>per forwarder × lane × cargo"]:::forwarder
    BUCKET["🗂️ Bucketizer<br/>UNK/LOW/MED/HIGH"]:::bucket
    SNAPSHOT --> BUCKET
  end
  
  %% Context Integration
  CONTEXT["📝 Context Fusion<br/>Cargo Type, Forwarder Quality"]:::context
  
  %% Decision Layer
  subgraph DECISION["🧠 RL Decision Engine"]
    direction TB
    STATE["🔗 State Builder<br/>Context Integration"]:::state
    DQN["🔗 Deep Q-Network<br/>Action Selection"]:::neural
    SAFETY["🛡️ Safety Override<br/>Rules & Policies"]:::safety
    STATE --> DQN
    DQN --> SAFETY
  end
  
  %% Action Execution
  subgraph ACTIONS["🎬 Action Space"]
    direction TB
    A_MON["👀 Monitor"]:::actionItem
    A_INC["📈 Increase Sampling"]:::actionItem
    A_PEER["👥 Peer Check"]:::actionItem
    A_CAL["⚙️ Calibrate"]:::actionItem
    A_ESC["🚨 Escalate"]:::actionItem
    A_FLAG["🏳️ Flag"]:::actionItem
  end
  
  %% Multi-Agent Orchestration
  subgraph AGENTS["🤖 Multi-Agent System"]
    direction TB
    DETECT["🕵️ Anomaly Agent<br/>Root Cause Analysis"]:::agent
    DECIDE["🧠 Decision Agent<br/>Policy Reasoning"]:::agent
    HUMAN["👤 Human-in-Loop<br/>Peer Review"]:::human
  end
  
  %% Output Dashboard
  OUTPUT["📊 Operations Dashboard<br/>Alerts & Tickets"]:::output
  
  %% Training Loop (Background)
  TRAIN["🎓 RL Training<br/>Offline Learning"]:::train
  
  %% Main Flow
  INGESTION --> FEATURES
  FEATURES --> ML
  ML --> CONTEXT
  FORWARDER --> CONTEXT
  CONTEXT --> STATE
  STATE --> DECISION
  DECISION --> ACTIONS
  ACTIONS --> AGENTS
  AGENTS --> OUTPUT
  
  %% Training Feedback Loop
  ACTIONS -.->|experience| TRAIN
  TRAIN -.->|updated model| DECISION
  
  %% Forwarder Quality Feedback Loop
  OUTPUT -.->|KPI updates| SNAPSHOT
  ACTIONS -.->|outcome logging| SNAPSHOT
  
  %% Conditional Flows
  CONTEXT -->|high risk| SAFETY
  A_PEER -->|peer review| HUMAN
  A_ESC -->|escalation| HUMAN
  A_FLAG -->|flagged items| OUTPUT
  
  %% Peer Review Decision Flow
  HUMAN -.->|review OK| A_MON
  HUMAN -.->|review NOT OK| A_ESC
  HUMAN -.->|uncertain| A_INC
  
  %% Dashboard Display Flows
  A_ESC -->|escalated alerts| OUTPUT
  A_MON -.->|status updates| OUTPUT

  %% Enhanced Style Definitions
  classDef sensor fill:#E8F6F3,stroke:#16A085,stroke-width:2px,color:#0E4B99
  classDef filter fill:#EBF5FB,stroke:#3498DB,stroke-width:2px,color:#1B4F72
  classDef feature fill:#FDF2E9,stroke:#E67E22,stroke-width:2px,color:#B7611A
  classDef ml fill:#F4ECF7,stroke:#8E44AD,stroke-width:2px,color:#6C3483
  classDef states fill:#FEF9E7,stroke:#F39C12,stroke-width:2px,color:#B7950B
  classDef context fill:#F8F9FA,stroke:#6C757D,stroke-width:2px,color:#495057
  classDef state fill:#E1F5FE,stroke:#0277BD,stroke-width:2px,color:#01579B
  classDef neural fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C
  classDef safety fill:#FFEBEE,stroke:#D32F2F,stroke-width:3px,color:#B71C1C,font-weight:bold
  classDef action fill:#F1F8E9,stroke:#689F38,stroke-width:2px,color:#33691E
  classDef agent fill:#E8EAF6,stroke:#3F51B5,stroke-width:2px,color:#1A237E
  classDef human fill:#FFF3E0,stroke:#FF9800,stroke-width:2px,color:#E65100
  classDef output fill:#E0F2F1,stroke:#00695C,stroke-width:3px,color:#004D40,font-weight:bold
  classDef train fill:#FFEBEE,stroke:#E57373,stroke-width:2px,stroke-dasharray: 5 5,color:#C62828
  classDef actionItem fill:#F9FBE7,stroke:#827717,stroke-width:1px,color:#33691E
  classDef forwarder fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
  classDef bucket fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C