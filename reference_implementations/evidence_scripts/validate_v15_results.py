#!/usr/bin/env python3
"""Validate V14 frozen evidence plus V15 theory-alignment additions."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = subprocess.run(
        ["python", str(root / "scripts" / "validate_v14_results.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    base_summary = json.loads(base.stdout)

    theory = root / "theory_diagnostics_frozen"
    required = [
        theory / "docs" / "PRODUCTION_RESULTS_V5.md",
        theory / "configs" / "conformal_null.json",
        theory / "configs" / "mechanism_2x2.json",
        theory / "configs" / "paired_score_oracle.json",
        theory / "configs" / "calibration_density.json",
        theory / "configs" / "signed_power.json",
        theory / "outputs" / "conformal_null" / "aggregate" / "conformal_null_rows.csv",
        theory / "outputs" / "mechanism_2x2" / "aggregate" / "mechanism_decomposition.csv",
        theory / "outputs" / "paired_score_oracle" / "aggregate" / "lambda_curve_summary.csv",
        theory / "outputs" / "calibration_density" / "aggregate" / "procedure_decomposition.csv",
        theory / "outputs" / "signed_power" / "aggregate" / "lambda_curve_summary.csv",
        root / "paper_tables_v15_additions" / "table_theory_experiment_map.tex",
        root / "docs" / "V15_THEORY_EXPERIMENT_HOLD_PLAN_CN.md",
        root / "plans" / "v15_theory_followup_hold.json",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f"Missing V15 files: {missing}")

    map_df = pd.read_csv(theory / "theory_experiment_map.csv")
    if len(map_df) != 5:
        raise AssertionError(f"Expected 5 theory-map rows, found {len(map_df)}")

    plan = json.loads((root / "plans" / "v15_theory_followup_hold.json").read_text())
    if plan.get("status") != "HOLD_NOT_RUN":
        raise AssertionError("Follow-up plan must remain HOLD_NOT_RUN")
    critical = plan.get("submission_critical", [])
    randomized = sum(int(x.get("new_randomized_tasks", 0)) for x in critical)
    if randomized != 270:
        raise AssertionError(f"Unexpected held randomized task count: {randomized}")

    summary = {
        "status": "PASS",
        "base_v14": base_summary,
        "theory_map_rows": len(map_df),
        "theory_required_files": len(required),
        "held_plan_status": plan["status"],
        "held_submission_critical_randomized_tasks": randomized,
        "held_deterministic_derivative_check": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
