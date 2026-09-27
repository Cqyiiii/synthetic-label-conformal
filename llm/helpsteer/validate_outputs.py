#!/usr/bin/env python3
"""Fail-fast validation for the HelpSteer2 experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    checks: dict[str, bool | int | float] = {}
    manifest = json.loads((ROOT / "data/helpsteer2_manifest.json").read_text(encoding="utf-8"))
    checks["zero_cross_role_prompt_overlap"] = manifest["cross_role_prompt_overlap"] == 0
    checks["zero_missing_required_values"] = manifest["missing_required_values"] == 0
    checks["zero_out_of_range_labels"] = manifest["out_of_range_label_values"] == 0

    for alias in ["flash", "pro"]:
        rows = [json.loads(line) for line in (ROOT / f"results/{alias}_labels.jsonl").read_text().splitlines() if line]
        successful = [row for row in rows if row.get("status") == "ok"]
        checks[f"{alias}_500_unique_successes"] = (
            len(successful) == 500 and len({row["row_id"] for row in successful}) == 500
        )
        checks[f"{alias}_zero_retries"] = sum(max(int(row["attempts"]) - 1, 0) for row in successful) == 0
        checks[f"{alias}_ratings_valid"] = all(
            set(row["ratings"]) == {"helpfulness", "correctness", "coherence", "complexity", "verbosity"}
            and all(0 <= float(value) <= 4 for value in row["ratings"].values())
            for row in successful
        )

    raw = pd.read_csv(ROOT / "results/downstream_cost_matched/raw_metrics.csv")
    summary = pd.read_csv(ROOT / "results/downstream_cost_matched/summary.csv")
    checks["downstream_expected_rows"] = len(raw) == 20 * 6
    checks["downstream_unique_seed_method"] = not raw.duplicated(["seed", "method"]).any()
    checks["downstream_no_missing"] = not raw.isna().any().any()
    checks["coverage_in_unit_interval"] = raw[
        ["pre_coverage", "coverage", "min_group_coverage", "pre_min_group_coverage"]
    ].apply(lambda column: column.between(0, 1).all()).all()
    checks["nonnegative_msce"] = bool((raw[["pre_msce", "msce"]] >= 0).all().all())

    recomputed = raw.groupby("method")["msce"].mean()
    reported = summary.set_index("method")["msce_mean"]
    checks["summary_matches_raw"] = bool(np.allclose(recomputed.sort_index(), reported.sort_index(), atol=1e-12))
    flash_cost = float(summary.loc[summary.method.eq("Flash-PPI"), "query_cost_usd_mean"].iloc[0])
    matched_cost = float(
        summary.loc[summary.method.eq("Pro-PPI-CostMatched"), "query_cost_usd_mean"].iloc[0]
    )
    checks["cost_match_within_0.0005_usd"] = abs(flash_cost - matched_cost) < 0.0005
    checks["flash_ppi_improves_msce"] = recomputed["Flash-PPI"] < recomputed["RCP"]
    checks["pro_ppi_improves_msce"] = recomputed["Pro-PPI"] < recomputed["RCP"]
    checks["pro_ppi_improves_min_group_coverage"] = (
        raw.loc[raw.method.eq("Pro-PPI"), "min_group_coverage"].mean()
        > raw.loc[raw.method.eq("RCP"), "min_group_coverage"].mean()
    )

    notebook = nbformat.read(ROOT / "helpsteer2_llm_pilot_report.ipynb", as_version=4)
    errors = [
        output
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.output_type == "error"
    ]
    checks["notebook_has_no_execution_errors"] = len(errors) == 0
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    result = {"all_checks_passed": not failed, "failed_checks": failed, "checks": checks}
    (ROOT / "results/validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
