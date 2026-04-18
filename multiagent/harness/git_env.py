from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitEnv:
    """Abstraction over local worktree vs GitLab CI branch."""
    repo_root: Path
    branch: str
    workdir: Path
    mode: str  # "worktree" | "branch"

    @classmethod
    def prepare(
        cls,
        repo_root: Path,
        branch: str,
        base: str = "main",
        local: bool | None = None,
    ) -> "GitEnv":
        in_ci = os.environ.get("GITLAB_CI") == "true"
        use_worktree = (not in_ci) if local is None else local
        if use_worktree:
            wt_root = repo_root / ".multiagent" / "worktrees" / branch.replace("/", "_")
            wt_root.parent.mkdir(parents=True, exist_ok=True)
            if not wt_root.exists():
                cls._run(repo_root, ["git", "fetch", "origin", base])
                # Reuse branch if already exists.
                rev = cls._run(repo_root, ["git", "rev-parse", "--verify", branch],
                               check=False)
                if rev.returncode != 0:
                    cls._run(repo_root, ["git", "worktree", "add", "-b", branch,
                                         str(wt_root), f"origin/{base}"])
                else:
                    cls._run(repo_root, ["git", "worktree", "add", str(wt_root), branch])
            return cls(repo_root, branch, wt_root, "worktree")
        # CI branch mode — operate in repo root.
        rev = cls._run(repo_root, ["git", "rev-parse", "--verify", branch], check=False)
        if rev.returncode != 0:
            cls._run(repo_root, ["git", "checkout", "-b", branch])
        else:
            cls._run(repo_root, ["git", "checkout", branch])
        return cls(repo_root, branch, repo_root, "branch")

    def commit_all(self, message: str) -> None:
        self._run(self.workdir, ["git", "add", "-A"])
        status = self._run(self.workdir, ["git", "status", "--porcelain"], capture=True)
        if not status.stdout.strip():
            return
        self._run(self.workdir, ["git", "commit", "-m", message])

    def push(self) -> None:
        self._run(self.workdir, ["git", "push", "-u", "origin", self.branch])

    def current_diff(self, base: str = "main") -> str:
        r = self._run(self.workdir, ["git", "diff", f"origin/{base}...HEAD"], capture=True)
        return r.stdout

    def cleanup_worktree(self) -> None:
        if self.mode != "worktree":
            return
        self._run(self.repo_root,
                  ["git", "worktree", "remove", "--force", str(self.workdir)],
                  check=False)

    @staticmethod
    def _run(cwd: Path, cmd: list[str], check: bool = True, capture: bool = False):
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            check=check,
            text=True,
            capture_output=True if capture else False,
        )
