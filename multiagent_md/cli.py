from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .workflows import run_debugger, run_development


def _load_cfg(path: Path | None) -> dict:
    default = Path(__file__).resolve().parent / "config" / "default.yaml"
    base = yaml.safe_load(default.read_text("utf-8"))
    if path and path.exists():
        user = yaml.safe_load(path.read_text("utf-8")) or {}
        base = _merge(base, user)
    return base


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="multiagent-md")
    sub = p.add_subparsers(dest="cmd", required=True)

    dev = sub.add_parser("dev")
    dev.add_argument("--task", required=True)
    dev.add_argument("--branch", required=True)
    dev.add_argument("--base", default="main")
    dev.add_argument("--repo", default=".")
    dev.add_argument("--config", default=None)
    dev.add_argument("--local", action="store_true")
    dev.add_argument("--ci", action="store_true")

    dbg = sub.add_parser("debug")
    dbg.add_argument("--branch", required=True)
    dbg.add_argument("--base", default="main")
    dbg.add_argument("--repo", default=".")
    dbg.add_argument("--config", default=None)
    dbg.add_argument("--local", action="store_true")
    dbg.add_argument("--ci", action="store_true")
    dbg.add_argument("--failed-job", default=os.environ.get("FAILED_JOB_NAME"))
    dbg.add_argument("--logs-file", default=os.environ.get("FAILED_JOB_LOGS_FILE"))
    dbg.add_argument("--logs", default=None)

    args = p.parse_args(argv)
    cfg = _load_cfg(Path(args.config)) if args.config else _load_cfg(None)
    repo = Path(args.repo).resolve()
    local = True if args.local else (False if args.ci else None)

    if args.cmd == "dev":
        res = run_development(cfg, repo, args.task, args.branch, args.base, local=local)
    else:
        logs = args.logs
        if not logs and args.logs_file and Path(args.logs_file).exists():
            logs = Path(args.logs_file).read_text("utf-8", errors="replace")
        logs = logs or ""
        res = run_debugger(cfg, repo, args.branch, logs, args.failed_job,
                           args.base, local=local)

    print(json.dumps({
        "ok": res.ok, "summary": res.summary, "budget": res.budget,
        "gate": res.gate, "iterations": res.iterations,
    }, indent=2, default=str))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
