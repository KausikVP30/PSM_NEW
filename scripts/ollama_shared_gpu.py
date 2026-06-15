from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GpuInfo:
    index: int
    free_memory_mb: int


def parse_nvidia_smi_output(output: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
            free_memory_mb = int(parts[1])
        except ValueError:
            continue
        gpus.append(GpuInfo(index=index, free_memory_mb=free_memory_mb))
    return gpus


def select_gpu_with_most_free_memory() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES is already set; GPU auto-selection is disabled.")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus = parse_nvidia_smi_output(result.stdout)
    if not gpus:
        raise RuntimeError("No GPUs were reported by nvidia-smi.")

    best_gpu = max(gpus, key=lambda gpu: gpu.free_memory_mb)
    return best_gpu.index


def select_cuda_visible_devices_if_unassigned() -> Optional[str]:
    current = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if current:
        return None
    selected_gpu = select_gpu_with_most_free_memory()
    return str(selected_gpu)