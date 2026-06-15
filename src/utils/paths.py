from __future__ import annotations

from pathlib import Path


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def resolve_device(setting: str) -> str:
    normalized = str(setting).strip().lower()
    if normalized in {"gpu", "cuda"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("CUDA is not available. GPU-only execution is required.")
        except Exception:
            raise RuntimeError("CUDA is not available or could not be initialized. GPU-only execution is required.")
    if normalized.startswith("cuda:"):
        try:
            import torch

            if torch.cuda.is_available():
                return normalized
            raise RuntimeError("CUDA is not available. GPU-only execution is required.")
        except Exception:
            raise RuntimeError("CUDA is not available or could not be initialized. GPU-only execution is required.")
    if normalized == "cpu":
        raise RuntimeError("CPU execution is disabled for this pipeline. Please provide a CUDA device.")
    if normalized != "auto":
        raise RuntimeError(f"Unsupported device setting: {setting}")
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA is not available. GPU-only execution is required.")
    except Exception:
        raise RuntimeError("CUDA is not available or could not be initialized. GPU-only execution is required.")
