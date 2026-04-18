from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .shell import run_shell


@dataclass
class CoverageReport:
    passed: bool
    coverage: float
    failing: list[str] = field(default_factory=list)
    logs: str = ""


_COVERAGE_RE = re.compile(r"TOTAL\s+\d+\s+\d+\s+(\d+)%")
_PYTEST_FAIL_RE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)


def run_full_suite(workdir: Path) -> CoverageReport:
    """Detect project type and run matching test suite. Pytest + coverage
    by default; Node (jest) and Go auto-detected.
    """
    test_cmd = os.environ.get("MULTIAGENT_TEST_CMD")
    if test_cmd:
        r = run_shell(workdir, test_cmd)
        return _parse_generic(r.ok, r.stdout + r.stderr)
    if (workdir / "package.json").exists():
        r = run_shell(workdir, "npm test --silent -- --coverage")
        return _parse_generic(r.ok, r.stdout + r.stderr)
    if (workdir / "go.mod").exists():
        r = run_shell(workdir, "go test ./... -cover")
        return _parse_generic(r.ok, r.stdout + r.stderr)
    # Default: Python + pytest.
    r = run_shell(
        workdir,
        "python -m pytest --maxfail=20 --cov --cov-report=term-missing",
    )
    logs = r.stdout + r.stderr
    cov = 0.0
    m = _COVERAGE_RE.search(logs)
    if m:
        cov = float(m.group(1))
    failing = _PYTEST_FAIL_RE.findall(logs)
    return CoverageReport(passed=r.ok, coverage=cov, failing=failing, logs=logs)


def _parse_generic(ok: bool, logs: str) -> CoverageReport:
    cov = 0.0
    m = _COVERAGE_RE.search(logs)
    if m:
        cov = float(m.group(1))
    else:
        m2 = re.search(r"All files\s*\|\s*([\d.]+)", logs)
        if m2:
            cov = float(m2.group(1))
        else:
            m3 = re.search(r"coverage:\s*([\d.]+)%", logs)
            if m3:
                cov = float(m3.group(1))
    failing = _PYTEST_FAIL_RE.findall(logs)
    return CoverageReport(passed=ok, coverage=cov, failing=failing, logs=logs)
