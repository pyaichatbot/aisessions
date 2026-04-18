from __future__ import annotations

from pathlib import Path
from typing import Any

from ..harness.git_env import GitEnv
from ..harness.orchestrator import Orchestrator, OrchestratorResult, Plan
from ..tools.git_ops import open_mr


def run_development(
    cfg: dict[str, Any],
    repo_root: Path,
    task: str,
    branch: str,
    base: str = "main",
    local: bool | None = None,
) -> OrchestratorResult:
    git = GitEnv.prepare(repo_root, branch, base=base, local=local)
    orch = Orchestrator(cfg, git)

    plan = orch.plan(task)

    failed = orch.run_batches(plan)
    if failed:
        failed = orch.run_batches(
            Plan(summary=plan.summary, complexity=plan.complexity, subtasks=failed),
            escalate=True,
        )
    git.commit_all(f"agent(coder): {plan.summary}")

    review_ok, iters = orch.review_and_fix(base, cfg["harness"]["max_iterations"])
    git.commit_all("agent(review): address issues")

    orch.tests_step(f"Write tests for: {task}\nPlan: {plan.summary}")
    git.commit_all("agent(tester): add tests")

    gate = orch.gate_step()
    git.commit_all(f"agent(gatekeeper): coverage={gate['coverage']}")

    git.push()
    mr_url = open_mr(
        git.workdir,
        title=f"[agent/dev] {plan.summary[:60]}",
        description=_desc(task, plan, gate, orch.budget.snapshot()),
        source=branch, target=base,
        labels=[cfg["gitlab"]["mr_label_dev"]],
    )
    orch.state.set("mr_url", mr_url)

    ok = review_ok and gate["passed"]
    return OrchestratorResult(
        ok=ok,
        summary=f"review_ok={review_ok} gate_ok={gate['passed']} mr={mr_url}",
        budget=orch.budget.snapshot(), gate=gate, iterations=iters,
    )


def _desc(task, plan, gate, budget) -> str:
    subs = "\n".join(f"- {s.id}: {s.title}" for s in plan.subtasks)
    return (
        f"## Task\n{task}\n\n"
        f"## Plan ({plan.complexity})\n{plan.summary}\n\n{subs}\n\n"
        f"## Gate\ncoverage={gate['coverage']} threshold={gate['threshold']} "
        f"passed={gate['passed']}\n\n"
        f"## Budget\n{budget}\n"
    )
