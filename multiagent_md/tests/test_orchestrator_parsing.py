from multiagent_md.harness.orchestrator import Orchestrator


SAMPLE = """
prefix prose
```json
{
  "summary": "retry",
  "complexity": "simple",
  "subtasks": [
    {"id":"t1","title":"a","description":"do a","parallel_group":"g"},
    {"id":"t2","title":"b","description":"do b","parallel_group":"g"},
    {"id":"t3","title":"c","description":"c","depends_on":["t1","t2"]}
  ]
}
```
"""


def test_plan_parse_and_batches() -> None:
    plan = Orchestrator.__dict__["_parse_plan"](
        object.__new__(Orchestrator), SAMPLE
    )
    assert plan.summary == "retry"
    assert len(plan.subtasks) == 3
    batches = plan.batches()
    assert {t.id for t in batches[0]} == {"t1", "t2"}
    assert batches[1][0].id == "t3"


def test_json_fallback() -> None:
    empty = Orchestrator._parse_json("not json")
    assert empty == {}
