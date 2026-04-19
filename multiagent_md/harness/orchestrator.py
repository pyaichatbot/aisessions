from __future__ import annotations

import concurrent.futures as cf
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..memory.wiki import WikiMemory
from ..tools.fs import apply_patch
from .budget import BudgetTracker
from .cache import PromptCache
from .git_env import GitEnv
from .router import ModelRouter
from .runner import AgentRunner, RunResult
from .state import SharedState
from .tool_registry import ToolRegistry


@dataclass
class SubTask:
    id: str
    title: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    parallel_group: str | None = None
    files_hint: list[str] = field(default_factory=list)


@dataclass
class Plan:
    summary: str
    complexity: str
    subtasks: list[SubTask]
    raw: str = ""

    def batches(self) -> list[list[SubTask]]:
        remaining = {t.id: t for t in self.subtasks}
        done: set[str] = set()
        batches: list[list[SubTask]] = []
        while remaining:
            ready = [t for t in remaining.values() if all(d in done for d in t.depends_on)]
            if not ready:
                batches.append(list(remaining.values()))
                break
            groups: dict[str, list[SubTask]] = {}
            for t in ready:
                key = t.parallel_group or t.id
                groups.setdefault(key, []).append(t)
            chosen = max(groups.values(), key=len)
            batches.append(chosen)
            for t in chosen:
                done.add(t.id)
                remaining.pop(t.id, None)
        return batches


@dataclass
class OrchestratorResult:
    ok: bool
    summary: str
    budget: dict[str, Any]
    gate: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0


class Orchestrator:
    """Lead agent. Wires md subagents via AgentRunner."""

    def __init__(self, cfg: dict[str, Any], git_env: GitEnv) -> None:
        self.cfg = cfg
        self.git = git_env
        self.workdir = git_env.workdir
        run_dir = self.workdir / ".multiagent_md" / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.state = SharedState(run_dir / "state.json")
        self.cache = PromptCache(
            root=self.workdir / ".multiagent_md" / "cache",
            enabled=cfg["harness"]["enable_prompt_cache"],
        )
        self.budget = BudgetTracker(cap_usd=float(cfg["harness"]["cost_budget_usd"]))
        self.router = ModelRouter(cfg["router"])
        self.memory = WikiMemory(
            root=self.workdir / cfg["memory"]["wiki_root"],
            page_bytes_max=int(cfg["memory"]["page_bytes_max"]),
            topk=int(cfg["memory"]["topk_retrieval"]),
        )
        self.tools = ToolRegistry(self.workdir)
        self.runner = AgentRunner(
            subagents_root=Path(__file__).resolve().parent.parent / "subagents",
            router=self.router, budget=self.budget,
            cache=self.cache, memory=self.memory, tools=self.tools,
        )

    # --- JSON parsing helpers -------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        payload = m.group(1) if m else text
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    def _parse_plan(self, text: str) -> Plan:
        data = self._parse_json(text)
        subs = [
            SubTask(
                id=s["id"],
                title=s.get("title", s["id"]),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
                parallel_group=s.get("parallel_group"),
                files_hint=s.get("files_hint", []),
            )
            for s in data.get("subtasks", [])
        ]
        if not subs:
            subs = [SubTask(id="t1", title="root", description=text)]
        return Plan(
            summary=data.get("summary", text[:120]),
            complexity=data.get("complexity", "simple"),
            subtasks=subs,
            raw=text,
        )

    # --- steps ----------------------------------------------------------------

    def plan(self, task: str) -> Plan:
        res = self.runner.run(
            "planner", task,
            context={"repo_tree_hint": _tree(self.workdir)},
        )
        plan = self._parse_plan(res.text)
        self.state.set("plan", {
            "summary": plan.summary, "complexity": plan.complexity,
            "subtasks": [s.__dict__ for s in plan.subtasks],
        })
        return plan

    def run_batches(self, plan: Plan, escalate: bool = False) -> list[SubTask]:
        failed: list[SubTask] = []
        max_par = int(self.cfg["harness"]["max_parallel_coders"])
        for batch in plan.batches():
            if len(batch) == 1:
                failed += self._run_coder(batch[0], escalate)
            else:
                with cf.ThreadPoolExecutor(max_workers=max_par) as ex:
                    futures = [ex.submit(self._run_coder, t, escalate) for t in batch]
                    for f in cf.as_completed(futures):
                        failed += f.result()
        return failed

    def _run_coder(self, sub: SubTask, escalate: bool) -> list[SubTask]:
        res = self.runner.run(
            "coder", sub.description or sub.title,
            context={
                "subtask_id": sub.id,
                "files_hint": ", ".join(sub.files_hint),
            },
            escalate=escalate,
        )
        apply_patch(self.workdir, res.text)  # handle any fenced blocks in summary
        self.state.append("coder_runs", {"id": sub.id, "ok": res.ok,
                                          "model": res.model})
        return [] if res.ok else [sub]

    def review_and_fix(self, base: str, max_iters: int) -> tuple[bool, int]:
        for i in range(max_iters):
            diff = self.git.current_diff(base)
            res = self.runner.run("reviewer", "Review this diff.",
                                   context={"diff": diff, "iteration": i})
            verdict = self._parse_json(res.text)
            approved = bool(verdict.get("approved", False))
            self.state.append("reviews", {
                "iter": i, "approved": approved,
                "severity": verdict.get("severity", "low"),
            })
            if approved:
                return True, i + 1
            issues = "\n".join(verdict.get("issues", []))
            fix = self.runner.run(
                "coder", "Address reviewer issues.",
                context={"issues": issues}, escalate=(i >= 1),
            )
            apply_patch(self.workdir, fix.text)
        return False, max_iters

    def tests_step(self, spec: str) -> bool:
        res = self.runner.run("tester", spec, context={})
        apply_patch(self.workdir, res.text)
        self.state.append("tester_runs", {"ok": res.ok})
        return res.ok

    def gate_step(self) -> dict[str, Any]:
        thr = float(self.cfg["harness"]["coverage_threshold"])
        res = self.runner.run(
            "gatekeeper",
            "Make tests green and coverage >= threshold.",
            context={"threshold": thr},
        )
        apply_patch(self.workdir, res.text)
        # Gatekeeper uses run_tests tool itself; final truth = one more run.
        from ..tools.test_runner import run_full_suite
        rep = run_full_suite(self.workdir)
        payload = {
            "passed": rep.passed and rep.coverage >= thr,
            "coverage": rep.coverage, "threshold": thr,
            "failing": rep.failing,
        }
        self.state.set("gate", payload)
        return payload

    def debug(self, failing_logs: str, failed_job: str) -> Plan:
        res = self.runner.run(
            "debugger", "Diagnose and produce fix plan.",
            context={"failing_logs": failing_logs, "job": failed_job},
        )
        data = self._parse_json(res.text)
        subs = [
            SubTask(
                id=s["id"], title=s.get("title", s["id"]),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
                parallel_group=s.get("parallel_group"),
                files_hint=s.get("files_hint", []),
            )
            for s in data.get("subtasks", [])
        ] or [SubTask(id="t1", title="root", description=res.text)]
        plan = Plan(
            summary=data.get("root_cause", "")[:200],
            complexity=data.get("complexity", "simple"),
            subtasks=subs, raw=res.text,
        )
        self.state.set("fix_plan", {
            "root_cause": plan.summary,
            "affected": data.get("affected", []),
            "complexity": plan.complexity,
            "subtasks": [s.__dict__ for s in plan.subtasks],
        })
        return plan


def _tree(root: Path, limit: int = 120) -> str:
    out = []
    for p in sorted(root.rglob("*")):
        if ".multiagent_md" in p.parts or ".git" in p.parts:
            continue
        if p.is_file():
            out.append(str(p.relative_to(root)))
            if len(out) >= limit:
                out.append("...")
                break
    return "\n".join(out)
