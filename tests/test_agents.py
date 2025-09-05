from src.phoenix_soc.agents.autogpt_planner import plan

def test_plan_structure():
    steps = plan("test")
    assert isinstance(steps, list) and len(steps) >= 1
    assert {"id","action","args"}.issubset(steps[0].keys())
