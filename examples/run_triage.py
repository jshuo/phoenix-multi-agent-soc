from src.phoenix_soc.agents.react_triage_agent import run_react_triage
from src.phoenix_soc.audit.mcp_envelope import build_envelope
from src.phoenix_soc.tools.pkcs11_signer import sign_payload

if __name__ == "__main__":
    task = "Investigate failed logins on fab network in last 5 minutes"
    triage = run_react_triage(task)
    env = build_envelope(prompt="ReAct triage", model="gpt-4o-mini", inputs={"task": task}, outputs=triage)
    env.signature = sign_payload(env.to_json())
    print(env.to_json())
