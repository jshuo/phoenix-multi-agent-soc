# tests/test_agents_react.py
from dotenv import load_dotenv
load_dotenv()

import json, re
from collections import Counter
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

# --------------------------
# Your log-analysis helpers
# --------------------------
def _load_fab_logs(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return [{"host": "N/A", "msg": f"ERROR: fab logs not found at {path}"}]
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _summarize_anomalies(rows: list[dict]) -> str:
    if len(rows) == 1 and rows[0].get("msg", "").startswith("ERROR:"):
        return rows[0]["msg"]

    msgs = [r.get("msg","") for r in rows]
    fails = sum("login failure" in m.lower() for m in msgs)
    unsigned_fw = [m for m in msgs if re.search(r"(signature:\s*MISSING|verify signature FAILED)", m, re.I)]
    exfil = [m for m in msgs if re.search(r"(SCP transfer|GB to )", m, re.I)]
    param_drifts = [m for m in msgs if re.search(r"(baseline\s*\d+)", m, re.I)]

    hosts = Counter(r.get("host","unknown") for r in rows if r.get("host"))
    top_host = hosts.most_common(1)[0][0] if hosts else "unknown"

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
        f"mitigate with {', '.join(mitigations) or 'standard SOC playbooks'}."
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
    return "[OK] Analysis complete.\n" + one_liner + "\nDETAIL:\n" + json.dumps(detail, indent=2)

def analyze_fab_logs_tool(tool_input: str) -> str:
    path = (tool_input or "").strip() or "tests/data/fab_logs.jsonl"
    rows = _load_fab_logs(path)
    return _summarize_anomalies(rows)

# --------------------------
# Tool call limiter
# --------------------------
def limit_calls(func, *, name: str, max_calls: int = 1):
    calls = {"n": 0}
    def wrapped(x):
        if calls["n"] >= max_calls:
            return f"[{name}] call limit reached ({max_calls}). Provide the FINAL ANSWER now."
        calls["n"] += 1
        return func(x)
    return wrapped

def build_tools():
    ddg = DuckDuckGoSearchRun()
    return [
        Tool(
            name="analyze_fab_logs",
            func=limit_calls(analyze_fab_logs_tool, name="analyze_fab_logs", max_calls=1),
            description="Analyze a fab JSONL log file and return anomalies + mitigations. Input = file path (default tests/data/fab_logs.jsonl)."
        ),
        Tool(
            name="web-search",
            func=limit_calls(ddg.run, name="web-search", max_calls=1),
            description="Search the web for best practices (firmware signing, DLP). Use at most once."
        ),
    ]

def build_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

REACT_PROMPT = PromptTemplate.from_template(
"""You are a SOC triage assistant for semiconductor fabs.

You have access to the following tools:
{tools}

Tool names you can call: {tool_names}

Use the ReAct format exactly. When you take a step, use:
Thought: <your reasoning>
Action: <one of {tool_names}>
Action Input: <JSON or string input for the tool>
Observation: <tool result>

When you are ready to finish, output exactly:
Final Answer: <one-sentence summary of anomalies and mitigations>
DETAIL:
<JSON with keys: top_host, counts, examples>

Hard rules:
- Call `analyze_fab_logs` FIRST with the provided path.
- Only if it materially improves mitigations, call `web-search` AT MOST ONCE.
- If the log file is missing, say so plainly in the Final Answer and STOP.
- Do not call a tool more than once if it already succeeded.

Question: {input}
{agent_scratchpad}
"""
)


def main():
    llm = build_llm()
    tools = build_tools()
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=4,
        early_stopping_method="generate",
        handle_parsing_errors=(
            "Your last message was not in ReAct format. "
            "If you intended to finish, reply now using exactly:\n"
            "Final Answer: <one sentence>\nDETAIL:\n<JSON>\n"
            "Otherwise, continue with:\nThought:\nAction: <tool name>\nAction Input: <input>"
        ),
    )

    user_task = (
        "Analyze tests/data/fab_logs.jsonl and summarize anomalies; include mitigations; "
        "if helpful, do one web-search for firmware signing or DLP best practices."
    )
    result = executor.invoke({"input": user_task})
    print("\n=== FINAL ANSWER ===\n", result["output"])

if __name__ == "__main__":
    main()
