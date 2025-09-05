from src.phoenix_soc.agents.react_triage_agent import run_react_triage
from src.phoenix_soc.agents.autogpt_planner import plan
from src.phoenix_soc.agents.coordinator_graph import build_graph

if __name__ == "__main__":
    app = build_graph(run_react_triage, plan)
    out = app.invoke({"task": "ssh brute force from unknown IPs in OT"})
    # In practice, you would parse out["triage"]["output"] (string) and forward to dashboard
    print(out["triage"])
