#!/usr/bin/env python3
"""Analyze the leakage-free DeepSeek prompt-calibration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODELS = ("flash", "pro")
VARIANTS = ("zero_shot", "icl", "icl_distributional")
PRICING = {
    "flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}
GATE = {
    "mae": 0.13,
    "absolute_bias": 0.03,
    "spearman": 0.45,
    "boundary_disagreement": 0.16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/civil_comments_20k.parquet"))
    parser.add_argument(
        "--assignments", type=Path, default=Path("prompt_gate/data/prompt_gate_assignments.parquet")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("prompt_gate/results"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/tfidf_svd64_features.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("prompt_gate/analysis"))
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--run-tag", default="interleaved_v2")
    return parser.parse_args()


def safe_corr(function: Callable, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(function(x, y)[0])


def read_records(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    raw = pd.DataFrame(records)
    if raw.empty:
        raise ValueError(f"No records found in {path}")
    raw["record_order"] = np.arange(len(raw))
    successful = raw.loc[raw["status"].eq("ok")].copy()
    successful = successful.sort_values("record_order").drop_duplicates("row_id", keep="last")
    return successful, raw


def calculate_cost(records: pd.DataFrame, model: str) -> float:
    price = PRICING[model]
    hit = float(records["prompt_cache_hit_tokens"].fillna(0).sum())
    miss = float(records["prompt_cache_miss_tokens"].fillna(0).sum())
    prompt = float(records["prompt_tokens"].fillna(0).sum())
    if hit == 0 and miss == 0:
        miss = prompt
    completion = float(records["completion_tokens"].fillna(0).sum())
    return (hit * price["cache_hit"] + miss * price["cache_miss"] + completion * price["output"]) / 1_000_000


def pinball_loss(score: np.ndarray, quantile: np.ndarray, tau: float) -> np.ndarray:
    residual = score - quantile
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def bootstrap_metric_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    truth: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    observed = statistic(truth, candidate) - statistic(truth, reference)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(truth), len(truth))
        draws[index] = statistic(truth[sample], candidate[sample]) - statistic(
            truth[sample], reference[sample]
        )
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(observed), float(lo), float(hi)


def mean_abs_error(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - truth)))


def mean_bias(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(prediction - truth))


def spearman_stat(truth: np.ndarray, prediction: np.ndarray) -> float:
    return safe_corr(spearmanr, truth, prediction)


def main() -> None:
    args = parse_args()
    source = pd.read_parquet(args.source.resolve()).reset_index(drop=True)
    assignments = pd.read_parquet(args.assignments.resolve()).copy()
    validation = assignments.loc[assignments["prompt_role"].eq("prompt_validation")].copy()
    if len(validation) != 500 or not validation["row_id"].is_unique:
        raise ValueError("Expected 500 unique prompt-validation rows")

    feature_cache = np.load(args.feature_cache.resolve())
    if feature_cache.shape[0] != len(source):
        raise ValueError(
            f"Feature cache has {feature_cache.shape[0]} rows but source has {len(source)}"
        )
    row_to_position = pd.Series(np.arange(len(source)), index=source["row_id"]).to_dict()
    base_positions = np.flatnonzero(source["role"].eq("base_train").to_numpy())
    validation_positions = np.array([row_to_position[row_id] for row_id in validation["row_id"]])
    x_base = feature_cache[base_positions]
    x_validation = feature_cache[validation_positions]

    mean_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    mean_model.fit(x_base, source.iloc[base_positions]["toxicity"].to_numpy())
    validation["mu_hat"] = mean_model.predict(x_validation)
    validation["score_true"] = np.abs(validation["toxicity"] - validation["mu_hat"])

    calibration_mask = validation["calibration_fold"].eq("calibration_train").to_numpy()
    evaluation_mask = validation["calibration_fold"].eq("evaluation").to_numpy()
    if calibration_mask.sum() != 250 or evaluation_mask.sum() != 250:
        raise ValueError("Expected a 250/250 calibration/evaluation split")

    tau = 0.9
    quantile_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=tau,
        n_estimators=120,
        max_depth=2,
        learning_rate=0.04,
        min_samples_leaf=12,
        random_state=args.seed,
    )
    quantile_model.fit(
        x_validation[calibration_mask], validation.loc[calibration_mask, "score_true"].to_numpy()
    )
    q_evaluation = np.maximum(quantile_model.predict(x_validation[evaluation_mask]), 0.0)
    true_score_evaluation = validation.loc[evaluation_mask, "score_true"].to_numpy()
    true_boundary = true_score_evaluation <= q_evaluation

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = validation.copy()
    raw_records: dict[tuple[str, str], pd.DataFrame] = {}
    quality_rows: list[dict] = []

    for model in MODELS:
        for variant in VARIANTS:
            labels, records = read_records(
                args.results_dir.resolve() / f"{model}_{variant}_{args.run_tag}.jsonl"
            )
            raw_records[(model, variant)] = records
            column = f"pred_{model}_{variant}"
            labels = labels[["row_id", "toxicity_pred", "served_model", "attempts", "latency_seconds"]].rename(
                columns={
                    "toxicity_pred": column,
                    "served_model": f"served_{model}_{variant}",
                    "attempts": f"attempts_{model}_{variant}",
                    "latency_seconds": f"latency_{model}_{variant}",
                }
            )
            merged = merged.merge(labels, on="row_id", how="left", validate="one_to_one")
            quality_rows.append(
                {
                    "model": model,
                    "prompt_variant": variant,
                    "raw_records": int(len(records)),
                    "unique_requested_rows": int(records["row_id"].nunique()),
                    "successful_unique_rows": int(labels["row_id"].nunique()),
                    "duplicate_success_records": int(
                        records.loc[records["status"].eq("ok"), "row_id"].duplicated(keep=False).sum()
                    ),
                    "error_records": int(records["status"].ne("ok").sum()),
                    "out_of_range_successes": int(
                        (~labels[column].between(0, 1, inclusive="both")).sum()
                    ),
                }
            )

    prediction_columns = [f"pred_{model}_{variant}" for model in MODELS for variant in VARIANTS]
    missing_predictions = merged[prediction_columns].isna().sum()
    if int(missing_predictions.sum()) != 0:
        raise ValueError(f"Missing predictions: {missing_predictions[missing_predictions.gt(0)].to_dict()}")

    summary_rows: list[dict] = []
    bin_rows: list[dict] = []
    boundary_arrays: dict[tuple[str, str], np.ndarray] = {}
    evaluation_predictions: dict[tuple[str, str], np.ndarray] = {}
    evaluation_truth = merged.loc[evaluation_mask, "toxicity"].to_numpy()

    for model in MODELS:
        for variant in VARIANTS:
            column = f"pred_{model}_{variant}"
            y_true = merged["toxicity"].to_numpy()
            y_pred = merged[column].to_numpy()
            calibration_true = y_true[calibration_mask]
            calibration_pred = y_pred[calibration_mask]
            eval_pred = y_pred[evaluation_mask]
            evaluation_predictions[(model, variant)] = eval_pred

            calibrator = LinearRegression().fit(calibration_pred.reshape(-1, 1), calibration_true)
            eval_calibrated = np.clip(calibrator.predict(eval_pred.reshape(-1, 1)), 0.0, 1.0)

            synthetic_score_eval = np.abs(eval_pred - merged.loc[evaluation_mask, "mu_hat"].to_numpy())
            calibrated_score_eval = np.abs(
                eval_calibrated - merged.loc[evaluation_mask, "mu_hat"].to_numpy()
            )
            raw_disagreement = (synthetic_score_eval <= q_evaluation) != true_boundary
            calibrated_disagreement = (calibrated_score_eval <= q_evaluation) != true_boundary
            boundary_arrays[(model, variant)] = raw_disagreement.astype(float)

            records = raw_records[(model, variant)]
            served_column = f"served_{model}_{variant}"
            attempts_column = f"attempts_{model}_{variant}"
            latency_column = f"latency_{model}_{variant}"
            row = {
                "model": model,
                "prompt_variant": variant,
                "served_model": ", ".join(sorted(merged[served_column].dropna().astype(str).unique())),
                "n": int(len(y_true)),
                "raw_mae": float(mean_absolute_error(y_true, y_pred)),
                "raw_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "raw_r2": float(r2_score(y_true, y_pred)),
                "raw_pearson": safe_corr(pearsonr, y_true, y_pred),
                "raw_spearman": safe_corr(spearmanr, y_true, y_pred),
                "raw_bias": float(np.mean(y_pred - y_true)),
                "n_unique_predictions": int(np.unique(y_pred).size),
                "toxicity_auc_at_0_5": float(roc_auc_score(y_true >= 0.5, y_pred)),
                "linear_calibration_intercept": float(calibrator.intercept_),
                "linear_calibration_slope": float(calibrator.coef_[0]),
                "heldout_calibrated_mae": float(mean_absolute_error(evaluation_truth, eval_calibrated)),
                "heldout_calibrated_r2": float(r2_score(evaluation_truth, eval_calibrated)),
                "heldout_calibrated_bias": float(np.mean(eval_calibrated - evaluation_truth)),
                "boundary_disagreement": float(np.mean(raw_disagreement)),
                "heldout_calibrated_boundary_disagreement": float(np.mean(calibrated_disagreement)),
                "mean_abs_pinball_proxy_gap": float(
                    np.mean(
                        np.abs(
                            pinball_loss(synthetic_score_eval, q_evaluation, tau)
                            - pinball_loss(true_score_evaluation, q_evaluation, tau)
                        )
                    )
                ),
                "latency_p50_seconds": float(np.nanmedian(merged[latency_column])),
                "latency_p90_seconds": float(np.nanquantile(merged[latency_column], 0.9)),
                "mean_attempts": float(np.nanmean(merged[attempts_column])),
                "prompt_tokens": int(records["prompt_tokens"].fillna(0).sum()),
                "completion_tokens": int(records["completion_tokens"].fillna(0).sum()),
                "prompt_cache_hit_tokens": int(records["prompt_cache_hit_tokens"].fillna(0).sum()),
                "prompt_cache_miss_tokens": int(records["prompt_cache_miss_tokens"].fillna(0).sum()),
                "estimated_cost_usd": calculate_cost(records, model),
            }
            row["gate_mae"] = bool(row["raw_mae"] <= GATE["mae"])
            row["gate_bias"] = bool(abs(row["raw_bias"]) <= GATE["absolute_bias"])
            row["gate_spearman"] = bool(row["raw_spearman"] >= GATE["spearman"])
            row["gate_boundary"] = bool(
                row["boundary_disagreement"] <= GATE["boundary_disagreement"]
            )
            gate_flags = [row["gate_mae"], row["gate_bias"], row["gate_spearman"], row["gate_boundary"]]
            row["gate_pass_count"] = int(sum(gate_flags))
            row["gate_pass_all"] = bool(all(gate_flags))
            summary_rows.append(row)

            for bin_name, group in merged.groupby("true_bin", observed=True):
                bin_rows.append(
                    {
                        "model": model,
                        "prompt_variant": variant,
                        "true_bin": str(bin_name),
                        "n": int(len(group)),
                        "true_mean": float(group["toxicity"].mean()),
                        "pred_mean": float(group[column].mean()),
                        "mae": float(np.mean(np.abs(group[column] - group["toxicity"]))),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(["model", "prompt_variant"]).reset_index(drop=True)
    quality = pd.DataFrame(quality_rows)
    by_bin = pd.DataFrame(bin_rows)
    summary.to_csv(output_dir / "prompt_gate_summary.csv", index=False)
    quality.to_csv(output_dir / "prompt_gate_data_quality.csv", index=False)
    by_bin.to_csv(output_dir / "prompt_gate_by_true_bin.csv", index=False)
    merged.to_parquet(output_dir / "prompt_gate_predictions.parquet", index=False)

    comparison_rows: list[dict] = []
    for model_index, model in enumerate(MODELS):
        reference = evaluation_predictions[(model, "zero_shot")]
        reference_boundary = boundary_arrays[(model, "zero_shot")]
        for variant_index, variant in enumerate(("icl", "icl_distributional")):
            candidate = evaluation_predictions[(model, variant)]
            for metric_index, (metric_name, statistic) in enumerate(
                [("mae", mean_abs_error), ("bias", mean_bias), ("spearman", spearman_stat)]
            ):
                estimate, lo, hi = bootstrap_metric_difference(
                    candidate,
                    reference,
                    evaluation_truth,
                    statistic,
                    args.bootstrap,
                    args.seed + 100 * model_index + 10 * variant_index + metric_index,
                )
                comparison_rows.append(
                    {
                        "model": model,
                        "candidate_variant": variant,
                        "reference_variant": "zero_shot",
                        "metric": metric_name,
                        "candidate_minus_reference": estimate,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
            candidate_boundary = boundary_arrays[(model, variant)]
            rng = np.random.default_rng(args.seed + 500 + 10 * model_index + variant_index)
            boundary_draws = np.empty(args.bootstrap)
            for index in range(args.bootstrap):
                sample = rng.integers(0, len(reference_boundary), len(reference_boundary))
                boundary_draws[index] = np.mean(candidate_boundary[sample]) - np.mean(
                    reference_boundary[sample]
                )
            comparison_rows.append(
                {
                    "model": model,
                    "candidate_variant": variant,
                    "reference_variant": "zero_shot",
                    "metric": "boundary_disagreement",
                    "candidate_minus_reference": float(
                        np.mean(candidate_boundary) - np.mean(reference_boundary)
                    ),
                    "ci95_low": float(np.quantile(boundary_draws, 0.025)),
                    "ci95_high": float(np.quantile(boundary_draws, 0.975)),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output_dir / "prompt_variant_paired_comparisons.csv", index=False)

    ranked = summary.sort_values(
        ["gate_pass_all", "gate_pass_count", "raw_mae", "boundary_disagreement", "raw_spearman"],
        ascending=[False, False, True, True, False],
    )
    selected = ranked.iloc[0]
    decision = {
        "gate_thresholds": GATE,
        "gate_passers": summary.loc[summary["gate_pass_all"], ["model", "prompt_variant"]].to_dict(
            orient="records"
        ),
        "recommended_model": str(selected["model"]),
        "recommended_prompt_variant": str(selected["prompt_variant"]),
        "recommended_gate_pass_count": int(selected["gate_pass_count"]),
        "recommend_continue_civil_comments": bool(selected["gate_pass_all"]),
        "total_estimated_api_cost_usd": float(summary["estimated_cost_usd"].sum()),
        "validation_rows": int(len(validation)),
        "calibration_rows": int(calibration_mask.sum()),
        "evaluation_rows": int(evaluation_mask.sum()),
        "selection_rule": (
            "Prefer all-gate passers; then maximize gates passed, minimize raw MAE and boundary "
            "disagreement, and maximize raw Spearman. No downstream metric is used."
        ),
        "run_tag": args.run_tag,
    }
    (output_dir / "prompt_gate_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    display_names = {
        "zero_shot": "Zero-shot",
        "icl": "16-shot ICL",
        "icl_distributional": "ICL + scale instruction",
    }
    colors = {"flash": "#3478BF", "pro": "#D95F59"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [
        ("raw_mae", "Raw MAE", GATE["mae"], "lower"),
        ("raw_bias", "Raw mean bias", GATE["absolute_bias"], "absolute"),
        ("raw_spearman", "Spearman correlation", GATE["spearman"], "higher"),
        ("boundary_disagreement", "Boundary disagreement", GATE["boundary_disagreement"], "lower"),
    ]
    x = np.arange(len(VARIANTS))
    width = 0.36
    for axis, (metric, title, threshold, direction) in zip(axes.flat, metrics):
        for model_index, model in enumerate(MODELS):
            model_view = summary.set_index(["model", "prompt_variant"]).loc[model]
            values = model_view.loc[list(VARIANTS), metric].to_numpy()
            axis.bar(x + (model_index - 0.5) * width, values, width, label=model, color=colors[model])
        if direction == "absolute":
            axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
            axis.axhline(-threshold, color="black", linestyle="--", linewidth=1)
        else:
            axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(x, [display_names[value] for value in VARIANTS], rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("DeepSeek prompt-calibration gate on 500 held-out Civil Comments", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_gate_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))
    print(f"Saved prompt-gate analysis to {output_dir}")


if __name__ == "__main__":
    main()
