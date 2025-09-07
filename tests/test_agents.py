# tests/test_agents.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain_community.vectorstores import FAISS
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun  # <- explicit import path
from langchain.agents import Tool

import json, re
from collections import Counter
from pathlib import Path

def build_memory():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_texts(["AutoGPT scratchpad initialized."], embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": 4})

def _load_fab_logs(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"fab logs not found at {path}")
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _summarize_anomalies(rows: list[dict]) -> str:
    msgs = [r["msg"] for r in rows]
    fails = sum("login failure" in m.lower() for m in msgs)
    unsigned_fw = [m for m in msgs if re.search(r"(signature:\s*MISSING|verify signature FAILED)", m, re.I)]
    exfil = [m for m in msgs if re.search(r"(SCP transfer|GB to )", m, re.I)]
    param_drifts = [m for m in msgs if re.search(r"(baseline\s*\d+)", m, re.I)]

    hosts = Counter(r["host"] for r in rows)
    top_host, _ = hosts.most_common(1)[0]

    mitigations = []
    if fails:
        mitigations.append("enforce FIDO2 + lockout on repeated failures")
    if unsigned_fw:
        mitigations.append("block unsigned firmware via HSM-signed allowlist")
    if exfil:
        mitigations.append("rate-limit and DLP on MES gateways; alert on large egress")
    if param_drifts:
        mitigations.append("require signed param changes + operator dual-control")

    one_liner = (
        f"Detected {fails} login failures, {len(unsigned_fw)} unsigned/failed-signature firmware events, "
        f"{len(exfil)} large egress transfers, and {len(param_drifts)} process-parameter drift(s); "
        f"mitigate with {', '.join(mitigations)}."
    )
    detail = {
        "top_host": top_host,
        "counts": {
            "login_failures": fails,
            "unsigned_fw_events": len(unsigned_fw),
            "large_egress_events": len(exfil),
            "param_drifts": len(param_drifts),
        },
        "examples": {
            "unsigned_fw": unsigned_fw[:2],
            "egress": exfil[:2],
            "param_drifts": param_drifts[:2],
        }
    }
    return one_liner + "\nDETAIL:\n" + json.dumps(detail, indent=2)

def analyze_fab_logs_tool(tool_input: str) -> str:
    """
    Tool signature: takes a path to a JSONL logs file and returns anomaly summary + mitigation.
    """
    path = tool_input.strip() or "tests/data/fab_logs.jsonl"
    rows = _load_fab_logs(path)
    return _summarize_anomalies(rows)

def build_tools():
    ddg = DuckDuckGoSearchRun()
    return [
        Tool(
            name="analyze_fab_logs",
            func=analyze_fab_logs_tool,
            description="Analyze a fab JSONL log file and return anomalies + mitigations. Input = file path (default tests/data/fab_logs.jsonl)."
        ),
        Tool(
            name="web-search",
            func=ddg.run,  # if you're on LC 0.3, ddg.invoke also works
            description="Search the web for up-to-date facts and URLs."
        ),
    ]

def build_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def main():
    llm = build_llm()
    tools = build_tools()
    memory = build_memory()

    agent = AutoGPT.from_llm_and_tools(
        ai_name="FabSecBuddy",
        ai_role=(
            "An autonomous fab security assistant that inspects OT/IT logs, "
            "flags anomalies (auth failures, unsigned firmware, data egress, parameter drifts), "
            "and suggests mitigations aligned with zero-trust and HSM-signed controls."
        ),
        tools=tools,
        llm=llm,
        memory=memory,
    )

    goals = [
        "Use analyze_fab_logs on tests/data/fab_logs.jsonl",
        "Summarize anomalies and mitigation steps in one sentence",
        "If helpful, do a quick web-search for best practices on firmware signing and DLP"
    ]

    # result = agent.run(goals, max_iterations=5)  # limit to keep runs snappy
    result = agent.run(goals)  # limit to keep runs snappy
    print("\n=== FINAL ANSWER ===\n", result)

if __name__ == "__main__":
    main()
