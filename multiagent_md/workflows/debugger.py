from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..harness.git_env import GitEnv
from ..harness.orchestrator import Orchestrator, OrchestratorResult, Plan
from ..tools.git_ops import open_mr


def run_debugger(
    cfg: dict[str, Any],
    repo_root: Path,
    branch: str,
    failing_logs: str,
    failed_job: str | None = None,
    base: str = "main",
    local: bool | None = None,
) -> OrchestratorResult:
    allowed = set(cfg["workflows"]["debug"]["trigger_jobs"])
    env_allowed = os.environ.get("MULTIAGENT_DEBUG_JOBS")
    if env_allowed:
        allowed = {j.strip() for j in env_allowed.split(",") if j.strip()}
    if failed_job and allowed and failed_job not in allowed:
        return OrchestratorResult(
            ok=True,
            summary=f"skip: job {failed_job} not in {sorted(allowed)}",
            budget={},
        )

    git = GitEnv.prepare(repo_root, branch, base=base, local=local)
    orch = Orchestrator(cfg, git)

    plan = orch.debug(failing_logs, failed_job or "")
    failed = orch.run_batches(plan)
    if failed:
        failed = orch.run_batches(
            Plan(summary=plan.summary, complexity=plan.complexity, subtasks=failed),
            escalate=True,
        )
    git.commit_all(f"agent(debug): {plan.summary[:60]}")

    review_ok, iters = orch.review_and_fix(base, cfg["harness"]["max_iterations"])
    git.commit_all("agent(review): address issues")

    orch.tests_step(f"Add/extend tests for fix: {plan.summary}")
    git.commit_all("agent(tester): stabilize")

    gate = orch.gate_step()
    git.commit_all(f"agent(gatekeeper): coverage={gate['coverage']}")

    git.push()
    mr_url = open_mr(
        git.workdir,
        title=f"[agent/debug] {plan.summary[:60]}",
        description=f"## Fix\n{plan.summary}\n\n## Gate\n{gate}\n",
        source=branch, target=base,
        labels=[cfg["gitlab"]["mr_label_debug"]],
    )
    orch.state.set("mr_url", mr_url)

    ok = review_ok and gate["passed"]
    return OrchestratorResult(
        ok=ok,
        summary=f"debug ok={ok} mr={mr_url}",
        budget=orch.budget.snapshot(), gate=gate, iterations=iters,
    )
