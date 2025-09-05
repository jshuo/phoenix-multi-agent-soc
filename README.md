# Phoenix Multi‑Agent SOC

Multi‑agent demo for SOC triage with **ReAct** + **AutoGPT**‑style planner orchestrated by **LangGraph**, plus **MCP‑style audit envelopes** and optional **HSM/PKCS#11 signing stubs**.

> Tailored for TSMC Senior Agent AI Engineer: agent architectures, tool use, multi‑agent collaboration, safety/ethics guardrails, and SOC‑style log triage.

## Features
- **ReAct triage agent** (tool‑use) — searches logs, looks up CTI, emits JSON with severity, summary, MITRE placeholders, and recommendations.
- **AutoGPT‑style planner** — decomposes tasks into steps (deterministic stub, interview‑friendly).
- **LangGraph coordinator** — planner → executor orchestration.
- **Auditability & safety** — MCP‑style envelope with hashes and demo signature; deny‑list policy gate.
- **SOC context** — tiny Next.js dashboard to visualize triage output.

## Quickstart (Python)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install langchain langgraph langchain-openai pydantic tiktoken python-dotenv
# Single‑agent ReAct triage on sample logs
python examples/run_triage.py
# Multi‑agent (planner → executor)
python examples/run_multi_agent.py
# Sign + verify an audit envelope (stub)
python examples/sign_and_verify.py
```

### Environment
Copy `.env.example` to `.env` and set your keys (or swap to a local LLM).
```bash
cp .env.example .env
```

## Quickstart (Dashboard)
```bash
cd dashboard
npm i
npm run dev
```
Visit http://localhost:3000 and paste JSON from `examples/run_multi_agent.py` to visualize alerts.

## Mapping to TSMC JD
- **Agent architectures**: ReAct tool‑use + AutoGPT‑style planner + LangGraph orchestration.
- **Tool integration**: mock SIEM (Wazuh/OpenSearch) and CTI; **PKCS#11** signer stub.
- **Safety/Ethics**: deny‑listed actions, audit envelope with verifiable hashes/signature.
- **Multi‑agent collaboration**: planner → executor hand‑off with shared state.
- **Evaluation hooks**: precision/recall, MTTD/MTTR stubs.

## Repo Layout
```
src/phoenix_soc/
  agents/ (react_triage_agent, autogpt_planner, coordinator_graph, safety_policies)
  tools/  (wazuh_search, cti_lookup, pkcs11_signer)
  audit/  (mcp_envelope, verifier)
  data/   (samples, schemas)
  evaluation/ (metrics)
examples/ (run_triage, run_multi_agent, sign_and_verify)
dashboard/ (Next.js mock UI)
tests/
```

## License
MIT — use freely for interviews and demos.
