from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Settings:
    raw: Dict[str, Any]

    @property
    def run(self) -> Dict[str, Any]:
        return self.raw.get("run", {})

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self.raw.get("retrieval", {})

    @property
    def reranking(self) -> Dict[str, Any]:
        return self.raw.get("reranking", {})

    @property
    def gating(self) -> Dict[str, Any]:
        return self.raw.get("gating", {})

    @property
    def generation(self) -> Dict[str, Any]:
        return self.raw.get("generation", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw.get("output", {})


def load_settings(config_path: str | Path) -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings(raw=raw)
