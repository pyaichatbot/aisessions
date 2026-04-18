from __future__ import annotations

from pathlib import Path
from typing import Any

from ..harness.git_env import GitEnv
from ..harness.orchestrator import Orchestrator, OrchestratorResult
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

    # 1) plan
    plan = orch.planner().plan(task, {"repo_tree_hint": _tree(git.workdir)})
    orch.state.set("plan", {"summary": plan.summary, "complexity": plan.complexity,
                             "subtasks": [s.__dict__ for s in plan.subtasks]})

    # 2) coders (sequential/parallel by batches)
    failed = orch.run_batches(plan)
    if failed:
        # Retry once with escalation.
        failed = orch.run_batches(_subplan(plan, failed), escalate=True)

    git.commit_all(f"agent(coder): {plan.summary}")

    # 3) review loop
    review_ok, iters = orch.review_and_fix(base, cfg["harness"]["max_iterations"])
    git.commit_all("agent(review): address issues")

    # 4) tests
    orch.tests_step(f"Write tests for: {task}\n\nPlan: {plan.summary}")
    git.commit_all("agent(tester): add tests")

    # 5) gate
    gate = orch.gate_step()
    git.commit_all(f"agent(gatekeeper): coverage={gate['coverage']}")

    # 6) push + single MR (idempotent)
    git.push()
    mr_url = open_mr(
        git.workdir,
        title=f"[agent/dev] {plan.summary[:60]}",
        description=_mr_description(task, plan, gate, orch.budget.snapshot()),
        source=branch,
        target=base,
        labels=[cfg["gitlab"]["mr_label_dev"]],
    )
    orch.state.set("mr_url", mr_url)

    ok = review_ok and gate["passed"]
    return OrchestratorResult(
        ok=ok,
        summary=f"review_ok={review_ok} gate_ok={gate['passed']} mr={mr_url}",
        budget=orch.budget.snapshot(),
        gate=gate,
        iterations=iters,
    )


def _subplan(plan, failed):
    from ..agents.planner import Plan
    return Plan(summary=plan.summary, complexity=plan.complexity, subtasks=failed)


def _tree(root: Path, limit: int = 120) -> str:
    out = []
    for p in sorted(root.rglob("*")):
        if ".multiagent" in p.parts or ".git" in p.parts:
            continue
        if p.is_file():
            out.append(str(p.relative_to(root)))
            if len(out) >= limit:
                out.append("...")
                break
    return "\n".join(out)


def _mr_description(task, plan, gate, budget) -> str:
    subs = "\n".join(f"- {s.id}: {s.title}" for s in plan.subtasks)
    return (
        f"## Task\n{task}\n\n"
        f"## Plan ({plan.complexity})\n{plan.summary}\n\n{subs}\n\n"
        f"## Gate\ncoverage={gate['coverage']} threshold={gate['threshold']} "
        f"passed={gate['passed']}\n\n"
        f"## Budget\n{budget}\n"
    )
