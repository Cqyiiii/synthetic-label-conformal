#!/usr/bin/env python3
"""Validate the frozen production evidence and V13 table inputs.

Historical suite identifiers retain their original ``v11_*`` names so task IDs
remain immutable. They are execution provenance, not manuscript-facing labels.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REQUIRED_METRICS = [
    "coverage",
    "coverage_error",
    "wsc",
    "ert_l1_gain",
    "mean_log_volume",
    "median_log_volume",
    "empty_frequency",
    "full_frequency",
    "msce_10_raw",
    "msce_10_debiased",
    "msce_10_split_half",
    "msce_30_raw",
    "msce_30_debiased",
    "msce_30_split_half",
]

EXPECTED_TASKS = 740
EXPECTED_METHOD_ROWS = 6900
EXPECTED_MAIN_TASKS = 240
EXPECTED_MAIN_METHODS = {
    "Split",
    "SPPI",
    "NNM",
    "CQR",
    "CQR-PPI",
    "RCP",
    "RCP-PTFT",
    "RCP-PPI",
    "RCP-PPI-CV",
    "RCP-Aug",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "completed_v11_results",
        help="Root containing plans/, outputs/, and reports/.",
    )
    parser.add_argument(
        "--skip-table-check",
        action="store_true",
        help="Skip deterministic regeneration and comparison of paper table fragments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    plan_path = root / "plans" / "v11_all.jsonl"
    rows_path = root / "reports" / "v11" / "task_method_rows.csv"
    qc_path = root / "reports" / "v11" / "qc.json"

    if not plan_path.exists():
        fail(f"Missing plan: {plan_path}")
    if not rows_path.exists():
        fail(f"Missing method rows: {rows_path}")

    plans = [json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
    plan_ids = {row["task_id"] for row in plans}
    if len(plans) != EXPECTED_TASKS or len(plan_ids) != EXPECTED_TASKS:
        fail(f"Expected {EXPECTED_TASKS} unique planned tasks, found {len(plans)} rows / {len(plan_ids)} IDs")

    task_files = sorted((root / "outputs").glob("*/tasks/*.json"))
    observed_ids = {p.stem for p in task_files}
    if len(task_files) != EXPECTED_TASKS or len(observed_ids) != EXPECTED_TASKS:
        fail(f"Expected {EXPECTED_TASKS} unique task files, found {len(task_files)} files / {len(observed_ids)} IDs")
    missing = sorted(plan_ids - observed_ids)
    extra = sorted(observed_ids - plan_ids)
    if missing or extra:
        fail(f"Task reconciliation failed: missing={len(missing)}, extra={len(extra)}")

    df = pd.read_csv(rows_path)
    if len(df) != EXPECTED_METHOD_ROWS:
        fail(f"Expected {EXPECTED_METHOD_ROWS} task-method rows, found {len(df)}")
    if bool(df["skipped"].fillna(False).astype(bool).any()):
        fail("At least one scientific method row is marked skipped")
    if bool(df["calibration_infinite"].fillna(False).astype(bool).any()):
        fail("At least one row has an infinite calibration correction")
    observed_fit_flags = df["fit_finite"].dropna().astype(bool)
    if not bool(observed_fit_flags.all()):
        fail("At least one fitted method row has a non-finite fit")
    if float(df["full_frequency"].fillna(0).abs().max()) != 0.0:
        fail("At least one method row produced a full prediction set")
    if int(df["outcome_access_ordinary"].fillna(0).sum()) != 0:
        fail("Ordinary synthetic methods accessed hidden true pool outcomes")

    for metric in REQUIRED_METRICS:
        if metric not in df.columns:
            fail(f"Required metric column missing: {metric}")
        values = pd.to_numeric(df[metric], errors="coerce")
        if not values.map(math.isfinite).all():
            fail(f"Non-finite values detected in required metric: {metric}")

    main_df = df[df["suite"] == "v11_main"].copy()
    main_task_ids = set(main_df["task_id"])
    if len(main_task_ids) != EXPECTED_MAIN_TASKS:
        fail(f"Expected {EXPECTED_MAIN_TASKS} main tasks, found {len(main_task_ids)}")
    if len(main_df) != EXPECTED_MAIN_TASKS * len(EXPECTED_MAIN_METHODS):
        fail(f"Expected {EXPECTED_MAIN_TASKS * len(EXPECTED_MAIN_METHODS)} main method rows, found {len(main_df)}")
    method_sets = main_df.groupby("task_id")["method"].apply(set)
    bad = method_sets[method_sets.map(lambda x: x != EXPECTED_MAIN_METHODS)]
    if len(bad):
        fail(f"Main tasks with incorrect method set: {len(bad)}")
    datasets = sorted(main_df["dataset"].unique().tolist())
    if len(datasets) != 8:
        fail(f"Expected 8 main datasets, found {datasets}")
    seed_counts = main_df.groupby("dataset")["seed"].nunique()
    if not (seed_counts == 30).all():
        fail(f"Main benchmark does not have 30 common seeds per dataset: {seed_counts.to_dict()}")

    # Inputs required by the two restored supplementary experiments.
    main_task_files = sorted((root / "outputs" / "v11_main" / "tasks").glob("*.json"))
    cv_pairs = 0
    for task_path in main_task_files:
        payload = json.loads(task_path.read_text())["result"]
        rows = {row["method"]: row for row in payload["methods"]}
        losses = rows["RCP-PPI-CV"].get("cv_losses") or {}
        if "0.0" not in losses or "1.0" not in losses:
            fail(f"Missing lambda=0/1 CV losses in {task_path}")
        if not (math.isfinite(float(losses["0.0"])) and math.isfinite(float(losses["1.0"]))):
            fail(f"Non-finite CV losses in {task_path}")
        cv_pairs += 1
    if cv_pairs != EXPECTED_MAIN_TASKS:
        fail(f"Expected {EXPECTED_MAIN_TASKS} main-task CV-loss pairs, found {cv_pairs}")

    pool_df = df[df["suite"] == "v11_pool_size"]
    expected_pool_shares = {0.05, 0.10, 0.20, 0.40}
    if set(round(float(x), 2) for x in pool_df["synthetic_prop"].unique()) != expected_pool_shares:
        fail("Unexpected synthetic-pool grid")

    table_check = "SKIPPED"
    if not args.skip_table_check:
        package_root = Path(__file__).resolve().parents[1]
        canonical = package_root / "paper_tables"
        with tempfile.TemporaryDirectory(prefix="v13_tables_") as tmp:
            tmp_path = Path(tmp)
            subprocess.run(
                [
                    "python",
                    str(package_root / "scripts" / "build_v13_tables.py"),
                    "--results-root",
                    str(root),
                    "--out-dir",
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            expected_names = {
                "tables_main.tex",
                "table_ablation.tex",
                "tables_appendix.tex",
                "tables_sensitivity.tex",
                "table_pinball.tex",
                "table_varying_N.tex",
            }
            if {p.name for p in canonical.glob("*.tex")} != expected_names:
                fail("Canonical paper_tables directory has an unexpected file set")
            for name in expected_names:
                if (tmp_path / name).read_text() != (canonical / name).read_text():
                    fail(f"Regenerated table differs from canonical fragment: {name}")
        table_check = "PASS"

    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    summary = {
        "status": "PASS",
        "root": str(root),
        "planned_tasks": len(plan_ids),
        "observed_tasks": len(observed_ids),
        "task_method_rows": len(df),
        "main_tasks": len(main_task_ids),
        "main_datasets": datasets,
        "main_methods_per_task": len(EXPECTED_MAIN_METHODS),
        "skipped_rows": int(df["skipped"].fillna(False).astype(bool).sum()),
        "nonfinite_required_metrics": 0,
        "infinite_calibrations": int(df["calibration_infinite"].fillna(False).astype(bool).sum()),
        "rows_with_full_sets": int((df["full_frequency"].fillna(0) > 0).sum()),
        "ordinary_pool_true_outcome_access": int(df["outcome_access_ordinary"].fillna(0).sum()),
        "supplementary_cv_pairs": cv_pairs,
        "table_regeneration": table_check,
        "reported_qc": qc,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
