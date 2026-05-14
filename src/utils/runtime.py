from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def kaggle_input_root() -> Optional[Path]:
    root = Path("/kaggle/input")
    return root if root.exists() else None


def resolve_output_path(path: str, default_filename: str) -> str:
    if path:
        return path
    if is_kaggle():
        return str(Path("/kaggle/working") / "outputs" / default_filename)
    return str(Path("outputs") / default_filename)


def _candidate_files(root: Path, stem: str) -> list[Path]:
    patterns = [f"**/{stem}.jsonl", f"**/{stem}.json", f"**/{stem}.csv"]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    return matches


def resolve_dataset_paths(dataset_paths: Dict[str, str], dataset_name: str = "") -> Dict[str, str]:
    """Resolve missing dataset paths automatically on Kaggle.

    If a path is already provided, it is preserved. Otherwise we search under
    /kaggle/input for common filenames like nq.jsonl, triviaqa.jsonl and
    hotpotqa.jsonl. This makes the notebook usable with only a mounted dataset.
    """

    resolved = dict(dataset_paths)
    root = kaggle_input_root()
    if root is None:
        return resolved

    for key in ("nq", "triviaqa", "hotpotqa"):
        if resolved.get(key):
            continue
        candidates = _candidate_files(root, key)
        if candidates:
            resolved[key] = str(candidates[0])

    # If a dataset name is supplied, also try that folder directly.
    if dataset_name and root.exists():
        named_root = root / dataset_name
        if named_root.exists():
            for key in ("nq", "triviaqa", "hotpotqa"):
                if not resolved.get(key):
                    candidates = _candidate_files(named_root, key)
                    if candidates:
                        resolved[key] = str(candidates[0])

    return resolved