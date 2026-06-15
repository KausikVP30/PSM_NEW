"""Utilities for semantic memory with embeddings."""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Tuple


def get_max_free_gpu() -> Optional[int]:
    """Automatically select GPU with maximum free memory.
    
    Returns:
        GPU index (0, 1, ...) or None if no GPU available or query fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return None

        gpu_memory = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    gpu_id = int(parts[0].strip())
                    free_mem = float(parts[1].strip())
                    gpu_memory.append((gpu_id, free_mem))
                except ValueError:
                    continue

        if not gpu_memory:
            return None

        # Return GPU with max free memory
        best_gpu = max(gpu_memory, key=lambda x: x[1])
        return best_gpu[0]

    except Exception:
        return None


def resolve_embedding_device(device: str) -> str:
    """Resolve embedding device (auto/gpu/cpu).
    
    Args:
        device: "auto", "gpu", or "cpu"
        
    Returns:
        "cuda:X" for GPU, "cpu" for CPU
    """
    normalized = device.lower().strip()
    if normalized == "cpu":
        raise RuntimeError("CPU execution is disabled for memory embeddings. GPU-only execution is required.")

    # Try GPU
    if normalized in ("auto", "gpu", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                # If CUDA_VISIBLE_DEVICES is set, respect it and use cuda:0
                if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                    return "cuda:0"
                if normalized == "auto":
                    # Auto-select GPU with max free memory
                    gpu_id = get_max_free_gpu()
                    if gpu_id is not None:
                        return f"cuda:{gpu_id}"
                else:
                    # User explicitly requested GPU
                    return "cuda:0"
        except Exception:
            pass

    raise RuntimeError("CUDA is not available or could not be initialized. GPU-only execution is required.")


def normalize_device(device_str: str) -> str:
    """Normalize device string to proper format.
    
    Args:
        device_str: Device string like "gpu", "cuda", "cuda:0", "cpu", "auto"
        
    Returns:
        Normalized device string for torch/transformers
    """
    d = device_str.lower().strip()

    if d in ("gpu", "cuda"):
        return resolve_embedding_device("gpu")
    elif d == "auto":
        return resolve_embedding_device("auto")
    elif d == "cpu":
        raise RuntimeError("CPU execution is disabled. GPU-only execution is required.")
    elif d.startswith("cuda:"):
        return d
    else:
        raise RuntimeError(f"Unsupported device string: {device_str}")
