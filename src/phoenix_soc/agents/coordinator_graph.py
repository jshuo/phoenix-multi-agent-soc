from typing import Dict, Any
from langgraph.graph import StateGraph, END

# shared state for agents
class S(dict):
    pass

def build_graph(run_triage, plan_func):
    g = StateGraph(S)

    def n_plan(state: S):
        state["plan"] = plan_func(state["task"])
        return state

    def n_act(state: S):
        # execute the simple plan using provided runner
        out = run_triage(state["task"])
        state["triage"] = out
        return state

    def n_finish(state: S):
        return state

    g.add_node("plan", n_plan)
    g.add_node("act", n_act)
    g.add_node("finish", n_finish)

    g.set_entry_point("plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "finish")
    return g.compile()
