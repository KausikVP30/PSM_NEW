"""Simple helper to prepare a Kaggle dataset directory by zipping a source folder.

Usage:
    python scripts/package_dataset.py --src path/to/corpus_folder --out out_dataset.zip

This script is optional — Kaggle's web UI can also be used to upload datasets.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def package(src: str, out: str) -> None:
    src_p = Path(src)
    if not src_p.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    out_p = Path(out)
    if out_p.exists():
        out_p.unlink()
    shutil.make_archive(str(out_p.with_suffix('')), 'zip', root_dir=str(src_p))
    print(f"Created {out_p} from {src_p}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source folder to include in dataset')
    parser.add_argument('--out', required=True, help='Output zip path (ends with .zip)')
    args = parser.parse_args()
    package(args.src, args.out)
