#!/usr/bin/env python3
"""Validate frozen V11 benchmark evidence plus V14 HelpSteer2 artifacts."""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
import pandas as pd


def fail(msg: str) -> None:
    raise AssertionError(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root / "completed_v11_results")
    args = parser.parse_args()

    base = subprocess.run([
        "python", str(root / "scripts" / "validate_v13_results.py"),
        "--root", str(args.root), "--skip-table-check",
    ], check=True, capture_output=True, text=True)
    base_summary = json.loads(base.stdout)

    help_dir = root / "helpsteer2_realistic_case"
    required_sources = {
        "detailed_experiment_report.md",
        "paper_ready_experiment_setting.md",
        "helpsteer2_llm_pilot_report.html",
        "helpsteer2_llm_pilot_report.ipynb",
    }
    missing_sources = sorted(name for name in required_sources if not (help_dir / name).exists())
    if missing_sources:
        fail(f"Missing HelpSteer2 source files: {missing_sources}")

    figures = root / "figures"
    down = pd.read_csv(figures / "helpsteer2_downstream_data.csv")
    paired = pd.read_csv(figures / "helpsteer2_paired_effects.csv")
    boundary = pd.read_csv(figures / "helpsteer2_boundary_data.csv")
    pairing = pd.read_csv(figures / "pairing_ablation_data.csv")
    if len(down) != 6 or len(paired) != 3 or len(boundary) != 3 or len(pairing) != 5:
        fail("Unexpected V14 figure-data row counts")
    checks = {
        "flash_msce": float(down.loc[down.method == "Flash-PPI", "msce"].iloc[0]),
        "pro_msce": float(down.loc[down.method == "Pro-PPI", "msce"].iloc[0]),
        "costmatched_n": 54.8,
        "flash_boundary_reduction": float(boundary.loc[boundary.labeler == "Flash", "reduction_percent"].iloc[0]),
        "pro_boundary_reduction": float(boundary.loc[boundary.labeler == "Pro", "reduction_percent"].iloc[0]),
    }
    expected = {
        "flash_msce": 0.005016,
        "pro_msce": 0.003998,
        "costmatched_n": 54.8,
        "flash_boundary_reduction": 32.5,
        "pro_boundary_reduction": 39.5,
    }
    for key, value in checks.items():
        if not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-10):
            fail(f"HelpSteer2 value mismatch for {key}: {value}")

    canonical_tables = root / "paper_tables"
    expected_names = {
        "tables_main.tex", "table_ablation.tex", "tables_appendix.tex",
        "tables_sensitivity.tex", "table_pinball.tex", "table_varying_N.tex",
        "table_helpsteer.tex",
    }
    if {p.name for p in canonical_tables.glob("*.tex")} != expected_names:
        fail("Canonical paper_tables directory has an unexpected file set")
    with tempfile.TemporaryDirectory(prefix="v14_tables_") as tmp:
        tmp_path = Path(tmp)
        subprocess.run([
            "python", str(root / "scripts" / "build_v14_tables.py"),
            "--results-root", str(args.root), "--out-dir", str(tmp_path),
        ], check=True, capture_output=True, text=True)
        for name in expected_names:
            if (tmp_path / name).read_text() != (canonical_tables / name).read_text():
                fail(f"Regenerated table differs from canonical fragment: {name}")

    for name in ["pairing_ablation.pdf", "helpsteer2_downstream.pdf"]:
        path = figures / name
        if not path.exists() or path.stat().st_size < 5000:
            fail(f"Missing or unexpectedly small vector figure: {name}")

    base_core = {
        key: base_summary[key]
        for key in [
            "planned_tasks", "observed_tasks", "task_method_rows", "main_tasks",
            "main_datasets", "main_methods_per_task", "skipped_rows",
            "nonfinite_required_metrics", "infinite_calibrations",
            "rows_with_full_sets", "ordinary_pool_true_outcome_access",
            "supplementary_cv_pairs",
        ]
    }
    summary = {
        "status": "PASS",
        "base_evidence": base_core,
        "helpsteer2_methods": len(down),
        "helpsteer2_paired_comparisons": len(paired),
        "helpsteer2_boundary_rows": len(boundary),
        "pairing_ablation_datasets": len(pairing),
        "canonical_tables": len(expected_names),
        "vector_figures": 2,
        "helpsteer2_source_files": len(required_sources),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
