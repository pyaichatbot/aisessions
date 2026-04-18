from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path


def current_diff(workdir: Path, base: str = "main") -> str:
    r = subprocess.run(
        ["git", "diff", f"origin/{base}...HEAD"],
        cwd=str(workdir), text=True, capture_output=True, check=False,
    )
    return r.stdout


def open_mr(
    workdir: Path,
    title: str,
    description: str,
    source: str,
    target: str = "main",
    labels: list[str] | None = None,
) -> str | None:
    """Open/update a single GitLab MR for this branch. Returns MR URL or None.

    Idempotent: reuses existing MR with same source/target.
    """
    token = os.environ.get("GITLAB_TOKEN") or os.environ.get("CI_JOB_TOKEN")
    project = os.environ.get("CI_PROJECT_ID") or os.environ.get("MULTIAGENT_PROJECT_ID")
    api = os.environ.get("CI_API_V4_URL", "https://gitlab.com/api/v4")
    if not token or not project:
        return None
    base = f"{api}/projects/{project}"
    hdr = {"PRIVATE-TOKEN": token, "Content-Type": "application/json"}
    q = (f"{base}/merge_requests?source_branch={source}&target_branch={target}"
         "&state=opened")
    req = urllib.request.Request(q, headers=hdr)
    with urllib.request.urlopen(req, timeout=30) as resp:
        existing = json.loads(resp.read())
    body = {
        "title": title,
        "description": description,
        "labels": ",".join(labels or []),
    }
    if existing:
        iid = existing[0]["iid"]
        req = urllib.request.Request(
            f"{base}/merge_requests/{iid}",
            data=json.dumps(body).encode("utf-8"),
            headers=hdr, method="PUT",
        )
    else:
        body.update({"source_branch": source, "target_branch": target})
        req = urllib.request.Request(
            f"{base}/merge_requests",
            data=json.dumps(body).encode("utf-8"),
            headers=hdr, method="POST",
        )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("web_url")
