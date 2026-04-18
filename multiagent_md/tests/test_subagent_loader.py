from pathlib import Path

from multiagent_md.subagents.loader import list_subagents, load_subagent, parse_md


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_all_six_subagents_present() -> None:
    agents = list_subagents(REPO_ROOT / "subagents")
    assert set(agents) == {
        "planner", "coder", "reviewer", "tester", "debugger", "gatekeeper",
    }


def test_frontmatter_parse_roundtrip() -> None:
    text = """---
name: linter
model_tier: cheap
tools: [read_file]
---
body text
"""
    fm, body = parse_md(text)
    assert fm["name"] == "linter"
    assert fm["tools"] == ["read_file"]
    assert body.strip() == "body text"


def test_coder_has_expected_tools() -> None:
    sa = load_subagent(REPO_ROOT / "subagents" / "coder.md")
    assert sa.model_tier == "cheap"
    assert "write_file" in sa.tools
    assert "apply_diff" in sa.tools


def test_gatekeeper_has_run_tests() -> None:
    sa = load_subagent(REPO_ROOT / "subagents" / "gatekeeper.md")
    assert "run_tests" in sa.tools
    assert sa.max_tool_iters >= 10


def test_no_md_missing_name_or_tier() -> None:
    for name, sa in list_subagents(REPO_ROOT / "subagents").items():
        assert sa.name == name
        assert sa.model_tier in {"cheap", "mid", "strong"}
