from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


_DENY = ("rm -rf /", "mkfs", "shutdown", ":(){", "dd if=", "> /dev/sda")


@dataclass
class ShellResult:
    ok: bool
    code: int
    stdout: str
    stderr: str


def run_shell(cwd: Path, cmd: str, timeout: int = 600,
              env: dict[str, str] | None = None) -> ShellResult:
    if any(bad in cmd for bad in _DENY):
        return ShellResult(False, 126, "", f"blocked: {cmd}")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        p = subprocess.run(
            shlex.split(cmd) if " " in cmd and not cmd.startswith("bash ") else cmd,
            shell=isinstance(cmd, str) and cmd.startswith("bash "),
            cwd=str(cwd),
            env=merged,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return ShellResult(p.returncode == 0, p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as e:
        return ShellResult(False, 124, e.stdout or "", f"timeout: {cmd}")
    except FileNotFoundError as e:
        return ShellResult(False, 127, "", str(e))
