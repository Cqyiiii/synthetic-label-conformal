from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import paired_bootstrap_difference


def _load_task_files(output_root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted((output_root / "tasks").glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            payload["_path"] = str(path)
            payloads.append(payload)
    return payloads


def _prefixed(mapping: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if not mapping:
        return {}
    return {f"{prefix}{key}": value for key, value in mapping.items()}


def _flatten_simulation(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        task = payload["task"]
        result = payload["result"]
        base = {
            "task_id": task["task_id"],
            "suite": task["suite"],
            "experiment": task["experiment"],
            "mechanism": result.get("mechanism"),
            "seed": result.get("seed"),
            "n": result.get("n"),
            "N": result.get("N"),
            "m": result.get("m"),
            "tau": result.get("tau"),
            "well_specified": result.get("well_specified"),
            "certified_synthetic_law": result.get("certified_synthetic_law"),
            "pseudo_truth_msce": result.get("pseudo_truth_msce"),
            "runtime_seconds": payload.get("runtime_seconds"),
            **_prefixed(result.get("population_constants"), "pop_"),
            **_prefixed(result.get("sample_constants"), "sample_"),
            **_prefixed(result.get("power_estimate"), "power_"),
        }
        for row in result.get("rows", []):
            rows.append({**base, **row})
    return pd.DataFrame(rows)


def _flatten_conformal_null(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        task = payload["task"]
        result = payload["result"]
        rows.append(
            {
                "task_id": task["task_id"],
                "suite": task["suite"],
                "seed": result.get("seed"),
                "strength": result.get("strength"),
                "n_mc": result.get("n_mc"),
                "raw_gain": result.get("raw_gain"),
                "postconformal_gain": result.get("postconformal_gain"),
                "raw_minimizer": result.get("raw_minimizer"),
                "postconformal_minimizer": result.get("postconformal_minimizer"),
                "acceptance_raw_positive": result.get("acceptance_raw_positive"),
                "acceptance_projected_zero": result.get("acceptance_projected_zero"),
                **_prefixed(result.get("raw_constants"), "raw_"),
                **_prefixed(result.get("projected_constants"), "projected_"),
            }
        )
    return pd.DataFrame(rows)


def _flatten_real(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        task = payload["task"]
        result = payload["result"]
        nuisance_mode = task.get("params", {}).get("nuisance_mode")
        for method in result.get("methods", []):
            base = {
                "task_id": task["task_id"],
                "suite": task["suite"],
                "experiment": task["experiment"],
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "role_seed": result.get("role_seed"),
                "teacher_strength": result.get("teacher_strength"),
                "label_budget": result.get("label_budget"),
                "synthetic_prop": task.get("params", {}).get("synthetic_prop"),
                "oracle_synthetic_labels": result.get("oracle_synthetic_labels"),
                "y_dim": result.get("y_dim"),
                "nuisance_mode": nuisance_mode,
                "teacher_validation_mse": result.get("teacher_validation_mse"),
                "role_hash_learning": result.get("role_hashes", {}).get("learning"),
                "role_hash_calibration": result.get("role_hashes", {}).get("calibration"),
                "role_hash_evaluation": result.get("role_hashes", {}).get("evaluation"),
                **{
                    key: value
                    for key, value in method.items()
                    if key not in {"grouped", "difficulty_profile"}
                },
            }
            base["wsc_error"] = (
                abs(float(base["wsc"]) - float(result.get("tau", 0.9)))
                if base.get("wsc") is not None
                else None
            )
            grouped = method.get("grouped", [])
            if grouped:
                for group in grouped:
                    rows.append({**base, **{f"group_{key}": value for key, value in group.items()}})
            else:
                rows.append(base)
    return pd.DataFrame(rows)


def _flatten_difficulty_profiles(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        task = payload["task"]
        result = payload["result"]
        for method in result.get("methods", []):
            for profile_row in method.get("difficulty_profile", []) or []:
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "suite": task["suite"],
                        "experiment": task["experiment"],
                        "dataset": result.get("dataset"),
                        "seed": result.get("seed"),
                        "role_seed": result.get("role_seed"),
                        "teacher_strength": result.get("teacher_strength"),
                        "label_budget": result.get("label_budget"),
                        "synthetic_prop": task.get("params", {}).get("synthetic_prop"),
                        "method": method.get("method"),
                        "method_family": method.get("method_family"),
                        "supervised_counterpart": method.get("supervised_counterpart"),
                        "profile_reference": method.get("difficulty_profile_reference"),
                        **profile_row,
                    }
                )
    return pd.DataFrame(rows)


def _resource_ledger(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        result = payload["result"]
        if "resource_ledger" not in result:
            continue
        rows.append(
            {
                "task_id": payload["task"]["task_id"],
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                **result["resource_ledger"],
            }
        )
    return pd.DataFrame(rows)


def _default_method_descriptor(method: str) -> dict[str, str]:
    """Backwards-compatible metadata for V5 and toy output rows."""

    method = str(method)
    if method == "Split":
        return {"method_family": "Split", "supervised_counterpart": "Split", "synthetic_use": "none", "legacy_alias": "Split"}
    if method in {"SPI-score-transporter", "SemiCP-NNM"}:
        use = "synthetic calibration scores" if method.startswith("SPI") else "proxy-score imputation"
        alias = "SPI" if method.startswith("SPI") else "SemiCP"
        return {"method_family": "Split", "supervised_counterpart": "Split", "synthetic_use": use, "legacy_alias": alias}
    if method == "CQR":
        return {"method_family": "CQR", "supervised_counterpart": "CQR", "synthetic_use": "none", "legacy_alias": "CQR"}
    if method.startswith("CQR-PPI"):
        return {"method_family": "CQR", "supervised_counterpart": "CQR", "synthetic_use": "endpoint learning", "legacy_alias": "CQR-Syn (audited one-step)"}
    if method == "RCP-supervised":
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "none", "legacy_alias": "RCP"}
    if method == "Budget-matched-supervised-RCP":
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "additional trusted labels", "legacy_alias": "budget-matched supervised"}
    if method == "RCP-PPI-unit-one-step":
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "prediction-powered threshold learning", "legacy_alias": "RCP-Syn (audited one-step)"}
    if method.startswith("RCP-PPI"):
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "prediction-powered threshold learning", "legacy_alias": "new power-selection variant"}
    if method == "RCP-synthetic-weighted-ERM":
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "weighted threshold learning", "legacy_alias": "new weighted synthetic variant"}
    if method == "RCP-synthetic-weighted-permuted-placebo":
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "permuted synthetic-label placebo", "legacy_alias": "marginal-preserving shuffled-label placebo"}
    if method == "RCP-synthetic-pretrain-one-step":
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "pretraining plus trusted correction", "legacy_alias": "new synthetic-pretrain variant"}
    if method == "Naive-synthetic-pooling-diagnostic":
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "direct pooling diagnostic", "legacy_alias": "naive pooling diagnostic"}
    return {"method_family": "other", "supervised_counterpart": "", "synthetic_use": "unspecified", "legacy_alias": ""}



def _ensure_method_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    defaults = enriched.get("method", pd.Series(index=enriched.index, dtype=object)).map(
        _default_method_descriptor
    )
    for column in ["method_family", "supervised_counterpart", "synthetic_use", "legacy_alias"]:
        fallback = defaults.map(lambda item: item[column])
        if column not in enriched:
            enriched[column] = fallback
        else:
            missing = enriched[column].isna() | (enriched[column].astype(str).str.len() == 0)
            enriched.loc[missing, column] = fallback.loc[missing]
    return enriched


def _paired_real_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare every synthetic method with its own supervised counterpart."""

    if frame.empty or "dataset" not in frame:
        return pd.DataFrame()
    frame = _ensure_method_metadata(frame)
    metric_directions = {
        "coverage_error": True,
        "group_cross_product": True,
        "group_raw": True,
        "group_debiased": True,
        "wsc_error": True,
        "mean_log_volume": True,
        "median_log_volume": True,
        "empty_frequency": True,
        "full_frequency": True,
    }
    records: list[dict[str, Any]] = []
    grouping = [
        "dataset",
        "teacher_strength",
        "label_budget",
        "synthetic_prop",
        "nuisance_mode",
        "group_k",
    ]
    existing_grouping = [column for column in grouping if column in frame]
    for group_values, group in frame.groupby(existing_grouping, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_descriptor = dict(zip(existing_grouping, group_values, strict=True))
        method_meta = (
            group[["method", "supervised_counterpart", "method_family", "synthetic_use", "legacy_alias"]]
            .drop_duplicates("method")
            .set_index("method")
        )
        for metric, lower_is_better in metric_directions.items():
            if metric not in group:
                continue
            valid = group.copy()
            if "skipped" in valid:
                valid = valid[~valid["skipped"].fillna(False).astype(bool)]
            pivot = valid.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
            for method in pivot.columns:
                if method not in method_meta.index:
                    continue
                counterpart = str(method_meta.loc[method, "supervised_counterpart"])
                if not counterpart or method == counterpart or counterpart not in pivot:
                    continue
                paired = pivot[[counterpart, method]].replace([np.inf, -np.inf], np.nan).dropna()
                if paired.empty:
                    continue
                summary = paired_bootstrap_difference(
                    paired[counterpart].to_numpy(),
                    paired[method].to_numpy(),
                    seed=2026,
                    lower_is_better=lower_is_better,
                )
                records.append(
                    {
                        **group_descriptor,
                        "metric": metric,
                        "method": method,
                        "counterpart": counterpart,
                        "method_family": method_meta.loc[method, "method_family"],
                        "synthetic_use": method_meta.loc[method, "synthetic_use"],
                        "legacy_alias": method_meta.loc[method, "legacy_alias"],
                        **summary,
                        "paired_n": int(len(paired)),
                        "lower_is_better": lower_is_better,
                    }
                )
    return pd.DataFrame(records)


def _dataset_equal_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    grouping = [
        column
        for column in [
            "method",
            "counterpart",
            "method_family",
            "synthetic_use",
            "legacy_alias",
            "metric",
            "teacher_strength",
            "label_budget",
            "synthetic_prop",
            "group_k",
        ]
        if column in comparisons
    ]
    records: list[dict[str, Any]] = []
    for values, group in comparisons.groupby(grouping, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        descriptor = dict(zip(grouping, values, strict=True))
        records.append(
            {
                **descriptor,
                "datasets": int(group["dataset"].nunique()),
                "dataset_equal_mean_difference": float(group["mean"].mean()),
                "dataset_win_fraction": float(np.mean(group["mean"] < 0)),
                "minimum_paired_n": int(group["paired_n"].min()),
            }
        )
    return pd.DataFrame(records)


def _curve_path(root: Path, payload: dict[str, Any]) -> Path:
    return root / "curves" / f"{payload['task']['task_id']}.npz"


def _simulation_curve_decomposition(
    payloads: list[dict[str, Any]], root: Path
) -> pd.DataFrame:
    grouped: dict[tuple[Any, ...], list[tuple[np.ndarray, np.ndarray, float]]] = defaultdict(list)
    for payload in payloads:
        result = payload["result"]
        path = _curve_path(root, payload)
        if not path.exists() or "mechanism" not in result:
            continue
        with np.load(path) as curves:
            if "pseudo_truth" not in curves:
                continue
            pseudo_truth = np.asarray(curves["pseudo_truth"], dtype=float)
            tau = float(result["tau"])
            for name in curves.files:
                if not (name.startswith("pi__") or name.startswith("pi_popcal__")):
                    continue
                calibration_mode, label = (
                    ("finite", name[len("pi__") :])
                    if name.startswith("pi__")
                    else ("population", name[len("pi_popcal__") :])
                )
                key = (
                    result.get("mechanism"),
                    result.get("n"),
                    result.get("N"),
                    result.get("m"),
                    tau,
                    calibration_mode,
                    label,
                )
                grouped[key].append((np.asarray(curves[name], dtype=float), pseudo_truth, tau))
    records = []
    for key, entries in grouped.items():
        if not entries:
            continue
        curves = np.stack([entry[0] for entry in entries], axis=0)
        pseudo = entries[0][1]
        tau = entries[0][2]
        ensemble = np.mean(curves, axis=0)
        procedure_variance = float(np.mean((curves - ensemble[None, :]) ** 2))
        ensemble_bias = float(np.mean((ensemble - pseudo) ** 2))
        approximation = float(np.mean((pseudo - tau) ** 2))
        interaction = float(2.0 * np.mean((pseudo - tau) * (ensemble - pseudo)))
        total = float(np.mean((curves - tau) ** 2))
        identity_error = total - (procedure_variance + ensemble_bias + approximation + interaction)
        mechanism, n, n_pool, m, tau_value, calibration_mode, label = key
        records.append(
            {
                "mechanism": mechanism,
                "n": n,
                "N": n_pool,
                "m": m,
                "tau": tau_value,
                "calibration_mode": calibration_mode,
                "label": label,
                "seeds": len(entries),
                "procedure_variance": procedure_variance,
                "ensemble_bias_squared": ensemble_bias,
                "approximation_error": approximation,
                "approximation_procedure_interaction": interaction,
                "total_msce": total,
                "identity_error": float(identity_error),
            }
        )
    return pd.DataFrame(records)


def _mechanism_decomposition(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for payload in payloads:
        result = payload["result"]
        constants = result.get("population_constants")
        if not constants:
            continue
        records.append(
            {
                "task_id": payload["task"]["task_id"],
                "mechanism": result.get("mechanism"),
                "seed": result.get("seed"),
                "n": result.get("n"),
                "N": result.get("N"),
                "m": result.get("m"),
                "b_total": constants.get("b"),
                "b_residual": constants.get("b_residual"),
                "b_alignment": constants.get("b_alignment"),
                "decomposition_error": constants.get("b_decomposition_error"),
                "a": constants.get("a"),
                "c": constants.get("c"),
                "c_pool": constants.get("c_pool"),
                "lambda_oracle": result.get("lambda_oracle"),
            }
        )
    return pd.DataFrame(records)


def _lambda_curve_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "label" not in frame:
        return pd.DataFrame()
    grid = frame[frame["label"].astype(str).str.startswith("grid_")].copy()
    if grid.empty:
        return pd.DataFrame()
    group_columns = ["mechanism", "n", "N", "m", "lambda"]
    metrics = [
        column
        for column in [
            "msce",
            "stochastic_error",
            "population_calibrated_msce",
            "stochastic_error_population_calibrated",
            "theory_learning",
        ]
        if column in grid
    ]
    return grid.groupby(group_columns, dropna=False, as_index=False)[metrics].agg(["mean", "std", "count"]).reset_index()


def _plot_lambda_curves(frame: pd.DataFrame, figure_dir: Path) -> None:
    if frame.empty or "lambda" not in frame or "msce" not in frame:
        return
    grid = frame[frame["label"].astype(str).str.startswith("grid_")]
    if grid.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    for keys, group in grid.groupby(["mechanism", "n", "N", "m"], dropna=False):
        mechanism, n, n_pool, m = keys
        summary = group.groupby("lambda", as_index=False)[
            ["msce", "stochastic_error", "theory_learning"]
        ].mean()
        plt.figure(figsize=(6.4, 4.2))
        plt.plot(summary["lambda"], summary["msce"], marker="o", label="finite-calibration MSCE")
        plt.plot(
            summary["lambda"],
            summary["stochastic_error"],
            marker="s",
            label="stochastic error",
        )
        plt.plot(
            summary["lambda"],
            summary["theory_learning"],
            marker="^",
            label="leading learning term",
        )
        plt.xlabel("power")
        plt.ylabel("error")
        plt.title(f"{mechanism}; n={n}, N={n_pool}, m={m}")
        plt.legend()
        plt.tight_layout()
        safe = str(mechanism).replace("/", "_")
        plt.savefig(figure_dir / f"lambda_{safe}_n{n}_N{n_pool}_m{m}.pdf")
        plt.close()


def _failure_summary(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for payload in payloads:
        result = payload["result"]
        rows = result.get("rows", result.get("methods", []))
        records.append(
            {
                "task_id": payload["task"]["task_id"],
                "experiment": payload["task"]["experiment"],
                "runtime_seconds": payload.get("runtime_seconds"),
                "rows": len(rows),
                "update_clipped_count": int(
                    sum(bool(row.get("update_clipped", False)) for row in rows)
                ),
                "infinite_calibration_count": int(
                    sum(bool(row.get("calibration_infinite", False)) for row in rows)
                ),
                "nonfinite_hessian_count": int(
                    sum(
                        not np.isfinite(float(row.get("hessian_condition", np.nan)))
                        for row in rows
                        if (
                            row.get("hessian_condition") is not None
                            and (
                                # Simulation rows with ``lambda=None`` are
                                # direct pooling fits, where a PPI Hessian is
                                # not applicable.  Do not report their
                                # intentional NaN sentinel as a failure.
                                ("label" in row and row.get("lambda") is not None)
                                or (
                                    not bool(row.get("skipped", False))
                                    and str(row.get("method", "")).startswith("RCP-PPI")
                                )
                                or (
                                    not bool(row.get("skipped", False))
                                    and str(row.get("method", "")).startswith("CQR-PPI")
                                )
                            )
                        )
                    )
                ),
                "density_floor_frequency": result.get("diagnostics", {})
                .get("plugin", {})
                .get("density_floor_frequency"),
                "density_crossing_frequency": result.get("diagnostics", {})
                .get("plugin", {})
                .get("density_crossing_frequency"),
                "skipped_method_count": int(sum(bool(row.get("skipped", False)) for row in rows)),
            }
        )
    return pd.DataFrame(records)


def aggregate_output(output_root: str | Path) -> dict[str, str]:
    root = Path(output_root)
    payloads = _load_task_files(root)
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    if not payloads:
        raise FileNotFoundError(f"No task JSON files under {root / 'tasks'}")
    experiment = payloads[0]["task"]["experiment"]
    outputs: dict[str, str] = {}

    failure = _failure_summary(payloads)
    failure_path = aggregate_dir / "failure_summary.csv"
    failure.to_csv(failure_path, index=False)
    outputs["failure_summary"] = str(failure_path)

    if experiment == "conformal_null":
        frame = _flatten_conformal_null(payloads)
        path = aggregate_dir / "conformal_null_rows.csv"
        frame.to_csv(path, index=False)
        outputs["rows"] = str(path)
        return outputs

    if (
        experiment.startswith("real_data")
        or experiment.startswith("synthetic_")
        or experiment.startswith("v9_")
        or experiment == "smoke_real"
    ):
        frame = _flatten_real(payloads)
        path = aggregate_dir / "real_data_rows.csv"
        frame.to_csv(path, index=False)
        outputs["rows"] = str(path)
        comparisons = _paired_real_comparisons(frame)
        comparison_path = aggregate_dir / "paired_real_comparisons.csv"
        comparisons.to_csv(comparison_path, index=False)
        outputs["comparisons"] = str(comparison_path)
        dataset_summary = _dataset_equal_summary(comparisons)
        dataset_summary_path = aggregate_dir / "dataset_equal_summary.csv"
        dataset_summary.to_csv(dataset_summary_path, index=False)
        outputs["dataset_equal_summary"] = str(dataset_summary_path)
        ledger = _resource_ledger(payloads)
        ledger_path = aggregate_dir / "resource_ledger.csv"
        ledger.to_csv(ledger_path, index=False)
        outputs["resource_ledger"] = str(ledger_path)
        profiles = _flatten_difficulty_profiles(payloads)
        if not profiles.empty:
            profile_path = aggregate_dir / "conditional_difficulty_profiles.csv"
            profiles.to_csv(profile_path, index=False)
            outputs["conditional_difficulty_profiles"] = str(profile_path)
        return outputs

    frame = _flatten_simulation(payloads)
    path = aggregate_dir / "mechanism_rows.csv"
    frame.to_csv(path, index=False)
    outputs["rows"] = str(path)

    mechanism = _mechanism_decomposition(payloads)
    mechanism_path = aggregate_dir / "mechanism_decomposition.csv"
    mechanism.to_csv(mechanism_path, index=False)
    outputs["mechanism_decomposition"] = str(mechanism_path)

    curve_decomposition = _simulation_curve_decomposition(payloads, root)
    decomposition_path = aggregate_dir / "procedure_decomposition.csv"
    curve_decomposition.to_csv(decomposition_path, index=False)
    outputs["procedure_decomposition"] = str(decomposition_path)

    lambda_summary = _lambda_curve_summary(frame)
    lambda_path = aggregate_dir / "lambda_curve_summary.csv"
    lambda_summary.to_csv(lambda_path, index=False)
    outputs["lambda_summary"] = str(lambda_path)
    _plot_lambda_curves(frame, aggregate_dir / "figures")
    return outputs
