from pathlib import Path
from typing import Any

from multiagent_md.harness.budget import BudgetTracker
from multiagent_md.harness.cache import PromptCache
from multiagent_md.harness.router import ModelResponse, ModelRouter
from multiagent_md.harness.runner import AgentRunner
from multiagent_md.harness.tool_registry import ToolRegistry
from multiagent_md.memory.wiki import WikiMemory


class FakeRouter(ModelRouter):
    def __init__(self, scripted: list[ModelResponse]) -> None:
        super().__init__({"cheap": "c", "mid": "m", "strong": "s"})
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def call(self, model, system, messages, tools=None, max_tokens=4096):
        self.calls.append({"model": model, "system": system,
                           "messages": messages, "tools": tools})
        return self.scripted.pop(0)


SUBAGENTS_ROOT = Path(__file__).resolve().parent.parent / "subagents"


def _runner(tmp_path: Path, scripted: list[ModelResponse]) -> AgentRunner:
    return AgentRunner(
        subagents_root=SUBAGENTS_ROOT,
        router=FakeRouter(scripted),
        budget=BudgetTracker(cap_usd=1.0),
        cache=PromptCache(tmp_path / "cache"),
        memory=WikiMemory(tmp_path / "wiki"),
        tools=ToolRegistry(tmp_path),
    )


def test_planner_simple_end_turn(tmp_path: Path) -> None:
    scripted = [ModelResponse(
        text='```json\n{"summary":"x","complexity":"simple","subtasks":[]}\n```',
        tool_use=[], stop_reason="end_turn",
        tokens_in=10, tokens_out=20, cost_usd=0.001, model="c",
        raw_content=[{"type": "text", "text": "ok"}],
    )]
    r = _runner(tmp_path, scripted)
    res = r.run("planner", "do something")
    assert res.ok
    assert "summary" in res.text


def test_tool_use_loop_executes_tool(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("print('hi')\n")
    scripted = [
        ModelResponse(
            text="", tool_use=[{"id": "tu1", "name": "read_file",
                                  "input": {"path": "src.py"}}],
            stop_reason="tool_use", tokens_in=5, tokens_out=5,
            cost_usd=0.0001, model="c",
            raw_content=[{"type": "tool_use", "id": "tu1",
                           "name": "read_file", "input": {"path": "src.py"}}],
        ),
        ModelResponse(
            text="done reading", tool_use=[], stop_reason="end_turn",
            tokens_in=5, tokens_out=5, cost_usd=0.0001, model="c",
            raw_content=[{"type": "text", "text": "done reading"}],
        ),
    ]
    r = _runner(tmp_path, scripted)
    res = r.run("coder", "inspect src.py")
    assert res.ok
    assert res.iters == 2
    # Second call should have a tool_result in the user message.
    last_call = r.router.calls[-1]
    last_user = last_call["messages"][-1]
    assert last_user["role"] == "user"
    assert any(c.get("type") == "tool_result" for c in last_user["content"])


def test_force_finalize_when_iters_cap_hit(tmp_path: Path) -> None:
    """If agent keeps calling tools, runner forces a final no-tools turn."""
    # Coder has max_tool_iters from md (20). Make a tiny agent: override via
    # a different subagent with low cap isn't necessary — we just simulate
    # a tool_use loop that never stops and rely on the force-finalize branch
    # by checking runner terminates.
    tool_call = ModelResponse(
        text="", tool_use=[{"id": "x", "name": "list_files", "input": {}}],
        stop_reason="tool_use", tokens_in=1, tokens_out=1,
        cost_usd=0.0, model="c",
        raw_content=[{"type": "tool_use", "id": "x",
                       "name": "list_files", "input": {}}],
    )
    # planner has max_tool_iters = default (10); supply enough to force finalize.
    scripted = [tool_call] * 10 + [ModelResponse(
        text="forced summary", tool_use=[], stop_reason="end_turn",
        tokens_in=1, tokens_out=1, cost_usd=0.0, model="c",
        raw_content=[{"type": "text", "text": "forced summary"}],
    )]
    r = _runner(tmp_path, scripted)
    # Use reviewer since its md sets max_tool_iters=10.
    res = r.run("reviewer", "dummy", {"diff": ""})
    assert res.text == "forced summary"
