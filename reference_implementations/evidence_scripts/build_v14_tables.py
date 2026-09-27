#!/usr/bin/env python3
"""Regenerate all LaTeX tables used by the V14 manuscript."""
from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-root", type=Path, default=root / "completed_v11_results")
    parser.add_argument("--out-dir", type=Path, default=root / "paper_tables")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "python", str(root / "scripts" / "build_v13_tables.py"),
        "--results-root", str(args.results_root), "--out-dir", str(args.out_dir),
    ], check=True)
    with tempfile.TemporaryDirectory(prefix="v14_figures_") as tmp:
        subprocess.run([
            "python", str(root / "scripts" / "build_v14_figures.py"),
            "--outdir", tmp, "--table-out", str(args.out_dir / "table_helpsteer.tex"),
        ], check=True)


if __name__ == "__main__":
    main()
