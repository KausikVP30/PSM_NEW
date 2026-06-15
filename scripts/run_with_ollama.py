#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from urllib import request
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ollama_shared_gpu import select_cuda_visible_devices_if_unassigned


def discover_ollama_binary() -> str | None:
    candidates = [
        shutil.which("ollama"),
        str(Path.home() / "local" / "ollama" / "bin" / "ollama"),
        str(Path.home() / "local" / "ollama" / "usr" / "local" / "bin" / "ollama"),
        str(Path.home() / "local" / "ollama" / "usr" / "bin" / "ollama"),
        str(Path.home() / "local" / "bin" / "ollama"),
        str(Path.home() / ".local" / "bin" / "ollama"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def wait_for_ollama(host: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    url = f"http://{host}/api/tags"
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Ollama did not become ready at {url} within {timeout_seconds} seconds")


def find_free_local_host() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
    return f"127.0.0.1:{port}"


def is_host_in_use(host: str) -> bool:
    hostname, port_text = host.rsplit(":", 1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((hostname, int(port_text))) == 0


def ensure_model_pulled(ollama_bin: str, env: dict[str, str], model_name: str) -> None:
    subprocess.run([ollama_bin, "pull", model_name], check=True, env=env)


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    with suppress(Exception):
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with suppress(Exception):
            process.kill()
        with suppress(Exception):
            process.wait(timeout=10)


def read_progress(progress_path: Path) -> tuple[int | None, int | None, float | None]:
    try:
        if not progress_path.exists():
            return None, None, None
        with progress_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        processed = data.get("processed")
        total = data.get("total_samples")
        avg_latency = data.get("avg_latency_seconds")
        return (
            int(processed) if processed is not None else None,
            int(total) if total is not None else None,
            float(avg_latency) if avg_latency is not None else None,
        )
    except Exception:
        return None, None, None


def archive_outputs(repo_root: Path, archive_dir: Path, gpu_label: str) -> tuple[Path, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    predictions_src = repo_root / "outputs" / "predictions" / "predictions.csv"
    metrics_src = repo_root / "outputs" / "metrics" / "metrics.json"
    predictions_dst = archive_dir / f"predictions_{gpu_label}_{timestamp}.csv"
    metrics_dst = archive_dir / f"metrics_{gpu_label}_{timestamp}.json"
    if predictions_src.exists():
        shutil.copy2(predictions_src, predictions_dst)
    if metrics_src.exists():
        shutil.copy2(metrics_src, metrics_dst)
    return predictions_dst, metrics_dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiment with a locally managed Ollama server.")
    parser.add_argument("--config", default="config_triviaqa_full.yaml", help="Path to YAML config")
    parser.add_argument(
        "--runner",
        default="run_experiment.py",
        help="Experiment entry script (e.g. run_triviaqa_raw_paraphrase_experiment.py)",
    )
    parser.add_argument("--mode", default=None, choices=["smoke", "subset", "full"], help="Override run mode")
    parser.add_argument("--model", default="llama3", help="Ollama model name to pull and use")
    parser.add_argument("--host", default="auto", help="Ollama host:port, or auto for a private free local port")
    parser.add_argument("--ollama-bin", default=None, help="Path to ollama executable if not on PATH")
    parser.add_argument("--monitor-interval-seconds", type=int, default=300, help="How often to print progress updates")
    parser.add_argument("--archive-dir", default="outputs/archive", help="Directory for timestamped final outputs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ollama_bin = args.ollama_bin or discover_ollama_binary()
    if not ollama_bin:
        raise FileNotFoundError("Could not find the ollama executable. Install it in your user environment first.")

    env = os.environ.copy()
    selected_gpu = select_cuda_visible_devices_if_unassigned()
    if selected_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = selected_gpu
    gpu_label = env.get("CUDA_VISIBLE_DEVICES", selected_gpu or "gpu")
    print(f"Selected CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'unset')} (max-free GPU when auto-selected)")

    host = find_free_local_host() if args.host == "auto" else args.host
    if is_host_in_use(host):
        raise RuntimeError(f"Ollama host {host} is already in use. Choose another --host to avoid sharing another process.")

    env["OLLAMA_HOST"] = host
    env["OLLAMA_ENDPOINT"] = f"http://{host}"
    env["OLLAMA_MODELS"] = str(Path.home() / "local" / "ollama" / "models")
    print(f"Using isolated Ollama endpoint http://{host}")

    Path(env["OLLAMA_MODELS"]).mkdir(parents=True, exist_ok=True)
    log_dir = repo_root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ollama_log_path = log_dir / "ollama_server.log"

    process: subprocess.Popen[str] | None = None
    log_handle = open(ollama_log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [ollama_bin, "serve"],
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if process.poll() is not None:
            raise RuntimeError(f"Ollama server exited early. Check {ollama_log_path}.")

        wait_for_ollama(host)
        ensure_model_pulled(ollama_bin, env, args.model)

        run_cmd = [sys.executable, args.runner, "--config", args.config]
        if args.mode:
            run_cmd.extend(["--mode", args.mode])

        progress_candidates = [
            repo_root / "outputs" / "metrics" / "progress_raw.json",
            repo_root / "outputs" / "metrics" / "progress_paraphrased.json",
            repo_root / "outputs" / "metrics" / "progress.json",
        ]
        progress_path = progress_candidates[0]
        progress_log_path = repo_root / "outputs" / "logs" / f"progress_{gpu_label}.log"
        run_process = subprocess.Popen(run_cmd, cwd=repo_root, env=env)
        last_report = 0.0
        with open(progress_log_path, "a", encoding="utf-8") as progress_log:
            while True:
                rc = run_process.poll()
                now = time.time()
                if now - last_report >= max(1, args.monitor_interval_seconds):
                    processed, total, avg_latency = (None, None, None)
                    for candidate in progress_candidates:
                        processed, total, avg_latency = read_progress(candidate)
                        if processed is not None:
                            progress_path = candidate
                            break
                    if processed is not None and total:
                        percent = (processed / total) * 100.0
                        message = f"Progress: {processed}/{total} ({percent:.1f}%)"
                        if avg_latency is not None:
                            message += f" | avg_latency={avg_latency:.2f}s"
                        print(message, flush=True)
                        progress_log.write(message + "\n")
                        progress_log.flush()
                    last_report = now
                if rc is not None:
                    if rc != 0:
                        raise subprocess.CalledProcessError(rc, run_cmd)
                    break
                time.sleep(5)

        archived_predictions, archived_metrics = archive_outputs(repo_root, Path(args.archive_dir), str(gpu_label))
        print(f"GPU selected via CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'unset')}")
        print(f"Ollama endpoint: http://{host}")
        print(f"Archived predictions: {archived_predictions}")
        print(f"Archived metrics: {archived_metrics}")
    finally:
        if process is not None:
            terminate_process(process)
        log_handle.close()


if __name__ == "__main__":
    main()
