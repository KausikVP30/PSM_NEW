#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--config",
        "config_triviaqa_full.yaml",
        "--mode",
        "full",
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
