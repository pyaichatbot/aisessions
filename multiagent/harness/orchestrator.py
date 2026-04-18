from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import BudgetTracker
from .cache import PromptCache
from .git_env import GitEnv
from .router import ModelRouter
from .state import SharedState
from ..agents import (
    Coder,
    Debugger,
    Gatekeeper,
    Planner,
    Reviewer,
    Tester,
)
from ..agents.planner import Plan, SubTask
from ..memory.wiki import WikiMemory


@dataclass
class OrchestratorResult:
    ok: bool
    summary: str
    budget: dict[str, Any]
    gate: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0


class Orchestrator:
    """Lead agent. Wires subagents together for a workflow run."""

    def __init__(
        self,
        cfg: dict[str, Any],
        git_env: GitEnv,
    ) -> None:
        self.cfg = cfg
        self.git = git_env
        self.workdir = git_env.workdir
        state_dir = self.workdir / ".multiagent" / "run"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state = SharedState(state_dir / "state.json")
        self.cache = PromptCache(
            root=self.workdir / ".multiagent" / "cache",
            enabled=cfg["harness"]["enable_prompt_cache"],
        )
        self.budget = BudgetTracker(cap_usd=float(cfg["harness"]["cost_budget_usd"]))
        self.router = ModelRouter(cfg["router"])
        self.memory = WikiMemory(
            root=self.workdir / cfg["memory"]["wiki_root"],
            page_bytes_max=int(cfg["memory"]["page_bytes_max"]),
            topk=int(cfg["memory"]["topk_retrieval"]),
        )

    def _make(self, cls):
        return cls(self.router, self.budget, self.cache, self.memory, self.workdir)

    # --- workflow building blocks ------------------------------------------------

    def run_batches(self, plan_like, escalate: bool = False) -> list[SubTask]:
        failed: list[SubTask] = []
        batches = plan_like.batches()
        max_par = int(self.cfg["harness"]["max_parallel_coders"])
        for batch in batches:
            if len(batch) == 1:
                failed += self._run_one(batch[0], escalate)
            else:
                with cf.ThreadPoolExecutor(max_workers=max_par) as ex:
                    futures = [ex.submit(self._run_one, t, escalate) for t in batch]
                    for f in cf.as_completed(futures):
                        failed += f.result()
        return failed

    def _run_one(self, sub: SubTask, escalate: bool) -> list[SubTask]:
        coder = self._make(Coder)
        ctx = {
            "files_hint": ", ".join(sub.files_hint),
            "subtask_id": sub.id,
            "escalate": escalate,
        }
        res = coder.code(sub.description or sub.title, ctx)
        self.state.append("coder_runs", {"id": sub.id, "ok": res.ok, "model": res.model})
        return [] if res.ok else [sub]

    def review_and_fix(
        self,
        base_branch: str,
        max_iters: int,
    ) -> tuple[bool, int]:
        reviewer = self._make(Reviewer)
        coder = self._make(Coder)
        for i in range(max_iters):
            diff = self.git.current_diff(base_branch)
            verdict = reviewer.review(diff, {"iteration": i})
            self.state.append("reviews", {"iter": i, "approved": verdict.approved,
                                          "severity": verdict.severity})
            if verdict.approved:
                return True, i + 1
            ctx = {"issues": "\n".join(verdict.issues), "escalate": i >= 1}
            fix = coder.code("Address reviewer issues.", ctx)
            self.state.append("review_fixes", {"iter": i, "ok": fix.ok})
        return False, max_iters

    def tests_step(self, spec: str) -> bool:
        tester = self._make(Tester)
        res = tester.write_tests(spec, {})
        self.state.append("tester_runs", {"ok": res.ok})
        return res.ok

    def gate_step(self) -> dict[str, Any]:
        gk = self._make(Gatekeeper)
        thr = float(self.cfg["harness"]["coverage_threshold"])
        report = gk.gate(thr, {}, max_fix_iters=int(self.cfg["harness"]["max_iterations"]))
        payload = {
            "passed": report.passed,
            "coverage": report.coverage,
            "threshold": report.threshold,
            "failing": report.failing,
        }
        self.state.set("gate", payload)
        return payload

    # --- public entrypoints -----------------------------------------------------

    def planner(self) -> Planner:
        return self._make(Planner)

    def debugger(self) -> Debugger:
        return self._make(Debugger)
