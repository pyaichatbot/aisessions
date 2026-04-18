from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..harness.git_env import GitEnv
from ..harness.orchestrator import Orchestrator, OrchestratorResult
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
    # Gate: only run for configured jobs.
    allowed = set(cfg["workflows"]["debug"]["trigger_jobs"])
    env_allowed = os.environ.get("MULTIAGENT_DEBUG_JOBS")
    if env_allowed:
        allowed = set([j.strip() for j in env_allowed.split(",") if j.strip()])
    if failed_job and allowed and failed_job not in allowed:
        return OrchestratorResult(
            ok=True,
            summary=f"skip: job {failed_job} not in {sorted(allowed)}",
            budget={},
        )

    # Current MR branch — no new branch or worktree switch.
    git = GitEnv.prepare(repo_root, branch, base=base, local=local)
    orch = Orchestrator(cfg, git)

    fix = orch.debugger().diagnose(failing_logs, {"job": failed_job or ""})
    orch.state.set("fix_plan", {
        "root_cause": fix.root_cause, "affected": fix.affected,
        "complexity": fix.complexity,
        "subtasks": [s.__dict__ for s in fix.subtasks],
    })

    failed = orch.run_batches(fix)
    if failed:
        from ..agents.debugger import FixPlan
        failed = orch.run_batches(
            FixPlan(root_cause=fix.root_cause, affected=fix.affected,
                    subtasks=failed, complexity=fix.complexity),
            escalate=True,
        )
    git.commit_all(f"agent(debug): {fix.root_cause[:60]}")

    review_ok, iters = orch.review_and_fix(base, cfg["harness"]["max_iterations"])
    git.commit_all("agent(review): address issues")

    orch.tests_step(f"Add/extend tests for fix: {fix.root_cause}")
    git.commit_all("agent(tester): stabilize")

    gate = orch.gate_step()
    git.commit_all(f"agent(gatekeeper): coverage={gate['coverage']}")

    git.push()
    # Update the SAME MR (source=branch,target=base) rather than create new.
    mr_url = open_mr(
        git.workdir,
        title=f"[agent/debug] {fix.root_cause[:60]}",
        description=f"## Fix\n{fix.root_cause}\n\n## Gate\n{gate}\n",
        source=branch,
        target=base,
        labels=[cfg["gitlab"]["mr_label_debug"]],
    )
    orch.state.set("mr_url", mr_url)

    ok = review_ok and gate["passed"]
    return OrchestratorResult(
        ok=ok,
        summary=f"debug ok={ok} mr={mr_url}",
        budget=orch.budget.snapshot(),
        gate=gate,
        iterations=iters,
    )
