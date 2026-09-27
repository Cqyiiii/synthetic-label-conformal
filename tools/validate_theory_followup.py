#!/usr/bin/env python3
"""Independent integrity and numerical validation for the V16 follow-up run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_NUMERIC_COLUMNS = [
    "finite_calibration_error",
    "population_calibrated_error",
    "predicted_learning_error",
    "predicted_total_error",
    "hessian_condition",
    "update_norm",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, *, atol: float = 1e-12) -> None:
    check(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol),
          f"{actual} differs from {expected}")


def validate(output_dir: Path) -> dict[str, object]:
    aggregate = output_dir / "aggregate"
    task_files = sorted((output_dir / "tasks").glob("*.json"))
    rows = pd.read_csv(aggregate / "three_resource_task_method_rows.csv")
    means = pd.read_csv(aggregate / "three_resource_scaling_summary.csv")
    saved_fits = pd.read_csv(aggregate / "three_resource_scaling_fits.csv")
    derivative = pd.read_csv(aggregate / "post_conformal_derivative_rows.csv")
    derivative_summary = pd.read_csv(aggregate / "post_conformal_derivative_summary.csv")
    manifest = json.loads((output_dir / "RUN_MANIFEST.json").read_text())

    check(len(task_files) == 270, "expected 270 task JSON files")
    check(len(rows) == 540, "expected 540 method rows")
    check(len(derivative) == 24, "expected three perturbations times eight epsilons")
    check(set(rows["method"]) == {"supervised", "oracle_power"}, "unexpected method")
    check(set(rows["sweep"]) == {"n_scaling", "N_scaling"}, "unexpected sweep")

    task_ids: set[str] = set()
    task_keys: set[tuple[object, ...]] = set()
    reconstructed: list[dict[str, object]] = []
    for path in task_files:
        payload = json.loads(path.read_text())
        task_id = payload["task_id"]
        descriptor = payload["descriptor"]
        result = payload["result"]
        check(path.stem == task_id, f"task filename/id mismatch: {path.name}")
        check(task_id not in task_ids, f"duplicate task id: {task_id}")
        task_ids.add(task_id)
        key = (descriptor["sweep"], descriptor["seed"], descriptor["n"],
               descriptor["N"], descriptor["m"])
        check(key not in task_keys, f"duplicate task descriptor: {key}")
        task_keys.add(key)
        check(len(result["rows"]) == 2, f"task {task_id} must contain two methods")
        for method_row in result["rows"]:
            reconstructed.append({**descriptor, **method_row})

    reconstructed_df = pd.DataFrame(reconstructed)
    identity = ["sweep", "seed", "n", "N", "m", "method"]
    check(not rows.duplicated(identity).any(), "aggregate contains duplicate method rows")
    check(not reconstructed_df.duplicated(identity).any(), "tasks contain duplicate method rows")
    check(set(map(tuple, rows[identity].to_numpy())) ==
          set(map(tuple, reconstructed_df[identity].to_numpy())),
          "task descriptors disagree with aggregate rows")

    sweep_counts = rows.groupby("sweep")["seed"].count().to_dict()
    check(sweep_counts == {"N_scaling": 300, "n_scaling": 240},
          f"unexpected sweep row counts: {sweep_counts}")
    for column in REQUIRED_NUMERIC_COLUMNS:
        check(np.isfinite(rows[column].to_numpy(dtype=float)).all(),
              f"non-finite values in {column}")
    check(not rows["update_clipped"].astype(bool).any(), "one or more updates were clipped")

    recomputed = (
        rows.groupby(["sweep", "method", "n", "N", "m"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            finite_calibration_error_mean=("finite_calibration_error", "mean"),
            population_calibrated_error_mean=("population_calibrated_error", "mean"),
        )
        .sort_values(["sweep", "method", "n", "N", "m"])
        .reset_index(drop=True)
    )
    saved_means = means[
        [
            "sweep", "method", "n", "N", "m", "seeds",
            "finite_calibration_error_mean", "population_calibrated_error_mean",
        ]
    ].sort_values(["sweep", "method", "n", "N", "m"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(recomputed, saved_means, check_exact=False, rtol=1e-12, atol=1e-15)
    check((saved_means["seeds"] == 30).all(), "each cell must contain 30 unique seeds")

    independently_fitted: dict[str, float] = {}
    for method in ["supervised", "oracle_power"]:
        n_rows = means[(means["sweep"] == "n_scaling") & (means["method"] == method)]
        n_slope = float(np.polyfit(np.log(n_rows["n"]),
                                   np.log(n_rows["finite_calibration_error_mean"]), 1)[0])
        independently_fitted[f"n_scaling:{method}"] = n_slope
        saved = saved_fits[(saved_fits["sweep"] == "n_scaling") &
                           (saved_fits["method"] == method)].iloc[0]
        close(n_slope, float(saved["estimate"]))

        pool_rows = means[(means["sweep"] == "N_scaling") & (means["method"] == method)]
        pool_slope = float(np.polyfit(1.0 / pool_rows["N"],
                                      pool_rows["population_calibrated_error_mean"], 1)[0])
        independently_fitted[f"N_scaling:{method}"] = pool_slope
        saved = saved_fits[(saved_fits["sweep"] == "N_scaling") &
                           (saved_fits["method"] == method)].iloc[0]
        close(pool_slope, float(saved["estimate"]))

    check(-1.4 < independently_fitted["n_scaling:oracle_power"] < -0.9,
          "oracle n-scaling is inconsistent with the 1/n prediction")
    check(independently_fitted["N_scaling:oracle_power"] > 0,
          "oracle N-scaling coefficient must be positive")
    check(abs(independently_fitted["N_scaling:supervised"]) < 1e-12,
          "supervised baseline should be invariant to N")
    oracle_n_means = means[(means["sweep"] == "n_scaling") &
                           (means["method"] == "oracle_power")].sort_values("n")
    check(np.all(np.diff(oracle_n_means["finite_calibration_error_mean"]) < 0),
          "oracle total error must decrease across the n sweep")

    check(set(derivative["perturbation"]) ==
          {"constant", "standardized_linear", "standardized_quadratic"},
          "unexpected perturbation family")
    check((derivative.groupby("perturbation").size() == 8).all(),
          "each perturbation must contain eight epsilon values")
    constant = derivative[derivative["perturbation"] == "constant"]
    check(constant["max_abs_threshold_difference"].max() < 1e-10,
          "constant perturbation is not an exact threshold null")
    check(constant["max_abs_coverage_difference"].max() < 1e-10,
          "constant perturbation is not an exact coverage null")
    derivative_slopes: dict[str, float] = {}
    for perturbation in ["standardized_linear", "standardized_quadratic"]:
        subset = derivative[derivative["perturbation"] == perturbation].nsmallest(5, "epsilon")
        slope = float(np.polyfit(np.log(subset["epsilon"]),
                                 np.log(subset["remainder_l2"]), 1)[0])
        derivative_slopes[perturbation] = slope
        check(1.8 <= slope <= 2.2, f"{perturbation} remainder slope is not second order")
        saved = derivative_summary[
            derivative_summary["perturbation"] == perturbation
        ].iloc[0]
        close(slope, float(saved["small_epsilon_loglog_slope"]))
        ordered = derivative[derivative["perturbation"] == perturbation].sort_values(
            "epsilon", ascending=False
        )
        check(np.all(np.diff(ordered["remainder_over_epsilon"]) < 0),
              f"{perturbation} remainder/epsilon does not decrease")

    figures = [
        output_dir / "figures" / "post_conformal_derivative.pdf",
        output_dir / "figures" / "three_resource_scaling.pdf",
        output_dir / "figures" / "post_conformal_derivative.png",
        output_dir / "figures" / "three_resource_scaling.png",
    ]
    for figure in figures:
        check(figure.exists() and figure.stat().st_size > 10_000,
              f"missing or suspiciously small figure: {figure}")
    check(manifest["status"] == "COMPLETE", "run manifest is incomplete")
    check(manifest["acceptance"]["passed"] is True, "run manifest acceptance failed")

    result: dict[str, object] = {
        "status": "PASS",
        "independent_of_production_aggregator": True,
        "task_files": len(task_files),
        "method_rows": len(rows),
        "unique_task_ids": len(task_ids),
        "cells": len(means),
        "seeds_per_cell": 30,
        "derivative_rows": len(derivative),
        "derivative_small_epsilon_slopes": derivative_slopes,
        "independent_scaling_fits": independently_fitted,
        "validated_figures": [str(path.relative_to(output_dir)) for path in figures],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = validate(args.output_dir)
    destination = args.output_dir / "VALIDATION.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
