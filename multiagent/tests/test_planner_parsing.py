from multiagent.agents.planner import Planner, Plan, SubTask


SAMPLE = """
```json
{
  "summary": "retry on fetch",
  "complexity": "simple",
  "subtasks": [
    {"id": "t1", "title": "wrap fetch", "description": "add retry",
     "depends_on": [], "parallel_group": "g1", "files_hint": ["a.py"]},
    {"id": "t2", "title": "wrap upload", "description": "add retry",
     "depends_on": [], "parallel_group": "g1", "files_hint": ["b.py"]},
    {"id": "t3", "title": "docs", "description": "update",
     "depends_on": ["t1","t2"], "files_hint": ["README.md"]}
  ]
}
```
"""


def test_parse_plan() -> None:
    plan = Planner._parse(SAMPLE)
    assert plan.summary == "retry on fetch"
    assert len(plan.subtasks) == 3
    batches = plan.batches()
    assert [t.id for t in batches[0]] == ["t1", "t2"]
    assert batches[1][0].id == "t3"


def test_plan_fallback_on_garbage() -> None:
    plan = Planner._parse("not json at all")
    assert plan.subtasks
    assert plan.complexity == "simple"
