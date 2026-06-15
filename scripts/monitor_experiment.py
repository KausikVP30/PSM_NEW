#!/usr/bin/env python3
"""
Simple monitor that checks every N seconds (default 600) and reports:
- elapsed wall time for the running `run_experiment.py` process
- number of processed samples (from outputs/metrics/progress.json or outputs/predictions/predictions.csv)
- total samples (from data/*.jsonl or progress.json)
- percent complete and ETA (best-effort)

Usage:
  python scripts/monitor_experiment.py        # run in foreground
  python scripts/monitor_experiment.py --once # run single check

Run in background:
  nohup python scripts/monitor_experiment.py &

"""
from __future__ import annotations

import time
import subprocess
import json
import os
import argparse
from datetime import timedelta, datetime

CHECK_INTERVAL = 600  # seconds (10 minutes)


def human(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def find_pid() -> str | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", "run_experiment.py"]).decode().strip()
        pids = [l for l in out.splitlines() if l.strip()]
        return pids[0] if pids else None
    except subprocess.CalledProcessError:
        return None


def elapsed_seconds(pid: str) -> int | None:
    try:
        out = subprocess.check_output(["ps", "-p", pid, "-o", "etimes="]).decode().strip()
        return int(out)
    except Exception:
        return None


def read_progress_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def total_from_data() -> int | None:
    # look for common dataset files
    candidates = ["data/triviaqa_full.jsonl", "data/triviaqa.jsonl", "data/triviaqa_sample.jsonl"]
    for p in candidates:
        if os.path.exists(p):
            try:
                out = subprocess.check_output(["wc", "-l", p]).decode().strip()
                return int(out.split()[0])
            except Exception:
                pass
    return None


def processed_from_predictions(predictions_path: str) -> int:
    try:
        if os.path.exists(predictions_path):
            with open(predictions_path, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            # subtract header if present
            return max(0, lines - 1)
    except Exception:
        pass
    return 0


def single_check() -> None:
    now = datetime.utcnow().isoformat()
    pid = find_pid()
    if not pid:
        print(f"[{now}] No running run_experiment.py process found.")
        return

    et = elapsed_seconds(pid)
    et_str = human(et) if et is not None else "N/A"

    # try progress.json first
    progress_path = "outputs/metrics/progress.json"
    progress = read_progress_json(progress_path) if os.path.exists(progress_path) else None

    if progress:
        total = progress.get("total_samples")
        processed = progress.get("processed", 0)
        avg_latency = progress.get("avg_latency_seconds")
    else:
        total = total_from_data()
        processed = processed_from_predictions("outputs/predictions/predictions.csv")
        avg_latency = None

    pct = None
    eta = None
    if total and total > 0:
        pct = (processed / total) * 100.0
        if avg_latency:
            remaining = max(0, total - processed)
            eta_seconds = remaining * float(avg_latency)
            eta = human(eta_seconds)
        else:
            # estimate avg from elapsed if possible
            if et is not None and processed > 0:
                avg = float(et) / float(processed)
                remaining = max(0, total - processed)
                eta_seconds = remaining * avg
                eta = human(eta_seconds)

    print(f"[{now}] PID={pid} | elapsed={et_str} | processed={processed} | total={total} | percent={pct:.2f}% if pct else 'N/A' | ETA={eta}")


def loop(interval: int) -> None:
    try:
        while True:
            single_check()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("monitor stopped")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL, help="check interval in seconds")
    parser.add_argument("--once", action="store_true", help="run one check and exit")
    args = parser.parse_args()

    if args.once:
        single_check()
    else:
        loop(args.interval)
