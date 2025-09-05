from typing import List, Dict

GOALS = """- Determine whether an incident is active.
- Gather all evidence required for audit.
- Produce an action plan that avoids unsafe steps and respects RBAC.
""".strip()

def plan(task: str) -> List[Dict]:
    # Minimal, deterministic stub for interviews (no API key needed)
    steps = [
        {"id": 1, "action": "search_logs", "args": {"query": task}},
        {"id": 2, "action": "lookup_cti", "args": {"indicator": "203.0.113.10"}},
        {"id": 3, "action": "summarize", "args": {"policy": "no destructive ops"}},
    ]
    return steps
