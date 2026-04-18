from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Agent, AgentResult
from ..tools.test_runner import run_full_suite, CoverageReport
from ..tools.fs import apply_patch


@dataclass
class GateReport:
    passed: bool
    coverage: float
    threshold: float
    failing: list[str]
    raw_logs: str


class Gatekeeper(Agent):
    role = "gatekeeper"
    tier = "mid"
    prompt_file = "gatekeeper.md"

    def gate(
        self,
        threshold: float,
        context: dict[str, Any],
        max_fix_iters: int = 3,
    ) -> GateReport:
        report: CoverageReport | None = None
        for _ in range(max_fix_iters + 1):
            report = run_full_suite(self.workdir)
            if report.passed and report.coverage >= threshold:
                return GateReport(True, report.coverage, threshold, [], report.logs)
            ctx = dict(context)
            ctx["failing_tests"] = "\n".join(report.failing)
            ctx["logs_tail"] = report.logs[-4000:]
            ctx["coverage"] = report.coverage
            ctx["threshold"] = threshold
            fix = self.run("Fix failing tests and raise coverage.", ctx)
            apply_patch(self.workdir, fix.output)
        final = report or run_full_suite(self.workdir)
        return GateReport(
            passed=final.passed and final.coverage >= threshold,
            coverage=final.coverage,
            threshold=threshold,
            failing=final.failing,
            raw_logs=final.logs,
        )
