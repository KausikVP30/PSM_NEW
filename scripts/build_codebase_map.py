from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "codebase_map.md"

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "kaggle_dataset_repo",
    "kaggle_kernel_output",
    "kaggle_kernel_output_latest",
    "logs",
    "outputs",
    "repo",
    "=0.7.0",
}

ROOT_FILES = [
    "README.md",
    "config.yaml",
    "config_gpu_run.yaml",
    "config_triviaqa_full.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-py311.txt",
    "run_experiment.py",
    "run_experiment_gpu.py",
    "run_triviaqa_full.py",
    "test_generator.py",
    "test_ollama_generator.py",
]

TREE_DIRS = ["data", "scripts", "src", "tests"]


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def iter_python_files() -> Iterable[Path]:
    for path in ROOT.rglob("*.py"):
        if is_excluded(path):
            continue
        if path.parent == ROOT or path.parts[-2] in {"src", "scripts", "tests"} or "src" in path.parts:
            yield path


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if len(parts) == 1:
        return path.stem
    if parts[0] in {"src", "scripts", "tests"}:
        module_parts = [parts[0], *parts[1:-1]]
        if path.name != "__init__.py":
            module_parts.append(path.stem)
        return ".".join(module_parts)
    return path.stem


def package_name(path: Path) -> str:
    module = module_name(path)
    if path.name == "__init__.py":
        return module
    if "." not in module:
        return module
    return module.rsplit(".", 1)[0]


def resolve_relative(current_package: str, level: int, module: str | None) -> str | None:
    parts = [part for part in current_package.split(".") if part]
    base_len = len(parts) - (level - 1)
    if base_len < 0:
        return None
    base = parts[:base_len]
    if module:
        base.extend(module.split("."))
    return ".".join(base) if base else None


def parse_internal_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    current_package = package_name(path)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                resolved = resolve_relative(current_package, node.level, node.module)
                if resolved:
                    imports.add(resolved)
            elif node.module:
                imports.add(node.module)

    return imports


def internal_target(module: str) -> str | None:
    if module == "src" or module.startswith("src."):
        return module
    if module == "scripts" or module.startswith("scripts."):
        return module
    if module == "tests" or module.startswith("tests."):
        return module
    if module in {p.stem for p in ROOT.glob("*.py")}:
        return module
    return None


def build_edges() -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for path in iter_python_files():
        source = module_name(path)
        for imported in parse_internal_imports(path):
            target = internal_target(imported)
            if target and target != source:
                edges[source].add(target)
    return edges


def render_tree() -> list[str]:
    lines: list[str] = [f"{ROOT.name}/"]

    for name in ROOT_FILES:
        file_path = ROOT / name
        if file_path.exists():
            lines.append(f"├── {name}")

    for dirname in TREE_DIRS:
        directory = ROOT / dirname
        if not directory.exists():
            continue
        lines.append(f"├── {dirname}/")
        entries = sorted(
            [p for p in directory.rglob("*") if not is_excluded(p) and p.is_file()],
            key=lambda p: p.relative_to(directory).parts,
        )
        rendered: set[str] = set()
        for entry in entries:
            rel = entry.relative_to(directory)
            if len(rel.parts) == 1:
                rendered.add(entry.name)
                lines.append(f"│   ├── {entry.name}")
                continue
            top = rel.parts[0]
            if top not in rendered:
                rendered.add(top)
                lines.append(f"│   ├── {top}/")
            tail = " / ".join(rel.parts[1:])
            lines.append(f"│   │   └── {tail}")

    lines.append("└── (excluded: kaggle_*, repo/, outputs/, logs/, .venv/)")
    return lines


def render_graph(edges: dict[str, set[str]]) -> list[str]:
    nodes = sorted({node for node in edges} | {target for targets in edges.values() for target in targets})
    lines = ["```mermaid", "flowchart LR"]
    for node in nodes:
        safe = node.replace(".", "_")
        label = node
        lines.append(f'  {safe}["{label}"]')
    lines.append("")
    for source in sorted(edges):
        for target in sorted(edges[source]):
            lines.append(f"  {source.replace('.', '_')} --> {target.replace('.', '_')}")
    lines.append("```")
    return lines


def render_markdown() -> str:
    edges = build_edges()
    tree = render_tree()

    core_packages = [
        "src.config: YAML-backed settings loader.",
        "src.data: dataset loading and schema objects.",
        "src.evaluation: answer scoring and metrics.",
        "src.generation: generator backends for T5 and Ollama.",
        "src.logging_utils: structured logging helpers.",
        "src.memory: semantic memory store and persistence helpers.",
        "src.pipeline: end-to-end RAG orchestration.",
        "src.prompt: prompt assembly.",
        "src.retrieval: BM25, dense HNSW, hybrid retrieval, reranking.",
        "src.router: confidence gate for memory-vs-retrieval routing.",
        "src.utils: IO, runtime, path, and seeding helpers.",
    ]

    entrypoints = [
        "run_experiment.py: primary main-project runner.",
        "run_experiment_gpu.py: GPU-oriented variant.",
        "run_triviaqa_full.py: full TriviaQA runner.",
        "scripts/*.py: local utilities for monitoring, Ollama, and dataset prep.",
    ]

    lines: list[str] = [
        "# Codebase Map",
        "",
        "This snapshot covers the active main project in the workspace and excludes Kaggle mirrors, generated outputs, and the nested `repo/` copy.",
        "",
        "## Structure",
        "",
        "```text",
        *tree,
        "```",
        "",
        "## Core Packages",
        "",
        *[f"- {item}" for item in core_packages],
        "",
        "## Entry Points",
        "",
        *[f"- {item}" for item in entrypoints],
        "",
        "## Import Graph",
        "",
        *render_graph(edges),
        "",
        "## Refresh",
        "",
        "Run `source .venv/bin/activate && python scripts/build_codebase_map.py` from the project root to regenerate this file.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Markdown map of the active codebase.")
    parser.add_argument("--output", default=str(OUTPUT), help="Markdown output path")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()