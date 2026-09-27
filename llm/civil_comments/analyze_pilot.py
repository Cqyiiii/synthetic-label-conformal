#!/usr/bin/env python3
"""Analyze DeepSeek pilot labels, including score-boundary diagnostics."""

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


PRICING = {
    "flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/civil_comments_20k.parquet"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    return parser.parse_args()


def read_latest_labels(path: Path, model_alias: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    all_records = pd.DataFrame(records)
    if all_records.empty:
        raise ValueError(f"No label records found in {path}")
    all_records["record_order"] = np.arange(len(all_records))
    successful = all_records.loc[all_records["status"].eq("ok")].copy()
    successful = successful.sort_values("record_order").drop_duplicates("row_id", keep="last")
    successful = successful.rename(
        columns={
            "toxicity_pred": f"toxicity_{model_alias}",
            "latency_seconds": f"latency_{model_alias}",
        }
    )
    return successful, all_records


def safe_corr(function: Callable, x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(function(x, y)[0])


def pinball_loss(score: np.ndarray, quantile: np.ndarray, tau: float) -> np.ndarray:
    residual = score - quantile
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(left) != len(right):
        raise ValueError("Paired bootstrap arrays must have equal length")
    rng = np.random.default_rng(seed)
    observed = statistic(left) - statistic(right)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(left), size=len(left))
        draws[index] = statistic(left[sample]) - statistic(right[sample])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(observed), float(lo), float(hi)


def calculate_cost(records: pd.DataFrame, model_alias: str) -> float:
    price = PRICING[model_alias]
    hit = float(records["prompt_cache_hit_tokens"].fillna(0).sum())
    miss = float(records["prompt_cache_miss_tokens"].fillna(0).sum())
    prompt = float(records["prompt_tokens"].fillna(0).sum())
    if hit == 0 and miss == 0:
        miss = prompt
    completion = float(records["completion_tokens"].fillna(0).sum())
    return (hit * price["cache_hit"] + miss * price["cache_miss"] + completion * price["output"]) / 1_000_000


def main() -> None:
    args = parse_args()
    data = pd.read_parquet(args.data.resolve())
    pilot = data.loc[data["role"].eq("pilot"), ["row_id", "text", "toxicity"]].copy()
    base_train = data.loc[data["role"].eq("base_train"), ["row_id", "text", "toxicity"]].copy()

    model_records: dict[str, pd.DataFrame] = {}
    merged = pilot.copy()
    for model_alias in ["flash", "pro"]:
        labels, records = read_latest_labels(
            args.results_dir.resolve() / f"{model_alias}_labels.jsonl", model_alias
        )
        model_records[model_alias] = records
        merged = merged.merge(
            labels[["row_id", f"toxicity_{model_alias}", f"latency_{model_alias}", "served_model", "attempts"]].rename(
                columns={
                    "served_model": f"served_model_{model_alias}",
                    "attempts": f"attempts_{model_alias}",
                }
            ),
            on="row_id",
            how="left",
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The pilot uses a lightweight, deterministic text representation only to define score residuals.
    combined_text = pd.concat([base_train["text"], pilot["text"]], ignore_index=True)
    vectorizer = TfidfVectorizer(
        max_features=12_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    sparse_features = vectorizer.fit_transform(combined_text)
    svd = TruncatedSVD(n_components=64, random_state=args.seed)
    dense_features = svd.fit_transform(sparse_features)
    base_count = len(base_train)
    x_base = dense_features[:base_count]
    x_pilot = dense_features[base_count:]

    mean_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    mean_model.fit(x_base, base_train["toxicity"].to_numpy())
    merged["mu_hat"] = mean_model.predict(x_pilot)
    merged["score_true"] = np.abs(merged["toxicity"] - merged["mu_hat"])
    for model_alias in ["flash", "pro"]:
        merged[f"score_{model_alias}"] = np.abs(
            merged[f"toxicity_{model_alias}"] - merged["mu_hat"]
        )

    complete_mask = merged[["toxicity_flash", "toxicity_pro"]].notna().all(axis=1).to_numpy()
    complete_indices = np.flatnonzero(complete_mask)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(complete_indices)
    split = len(complete_indices) // 2
    boundary_cal = complete_indices[:split]
    boundary_eval = complete_indices[split:]

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
    quantile_model.fit(x_pilot[boundary_cal], merged.iloc[boundary_cal]["score_true"].to_numpy())
    q_eval = quantile_model.predict(x_pilot[boundary_eval])
    q_eval = np.maximum(q_eval, 0.0)
    true_score_eval = merged.iloc[boundary_eval]["score_true"].to_numpy()

    summary_rows: list[dict] = []
    calibration_rows: list[dict] = []
    true_bins = pd.cut(
        merged["toxicity"],
        bins=[-1e-9, 0.001, 0.1, 0.3, 0.5, 0.7, 1.0001],
        include_lowest=True,
    )
    for model_alias in ["flash", "pro"]:
        prediction = merged[f"toxicity_{model_alias}"].to_numpy()
        valid = np.isfinite(prediction)
        y_true = merged.loc[valid, "toxicity"].to_numpy()
        y_pred = prediction[valid]

        score_pred_eval = merged.iloc[boundary_eval][f"score_{model_alias}"].to_numpy()
        boundary_true = true_score_eval <= q_eval
        boundary_pred = score_pred_eval <= q_eval
        disagreement = boundary_true != boundary_pred
        true_pinball = pinball_loss(true_score_eval, q_eval, tau)
        synthetic_pinball = pinball_loss(score_pred_eval, q_eval, tau)

        linear_calibrator = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
        calibrated_prediction = np.clip(
            linear_calibrator.predict(y_pred.reshape(-1, 1)), 0.0, 1.0
        )

        calibration_frame = pd.DataFrame(
            {"true": merged["toxicity"], "pred": prediction, "true_bin": true_bins}
        )
        for bin_name, group in calibration_frame.groupby("true_bin", observed=True):
            calibration_rows.append(
                {
                    "model": model_alias,
                    "true_bin": str(bin_name),
                    "n": int(len(group)),
                    "true_mean": float(group["true"].mean()),
                    "pred_mean": float(group["pred"].mean()),
                    "mae": float(np.mean(np.abs(group["pred"] - group["true"]))),
                }
            )

        records = model_records[model_alias]
        ok_rows = records["status"].eq("ok").sum()
        total_unique_requested = records["row_id"].nunique()
        summary_rows.append(
            {
                "model": model_alias,
                "served_model": ", ".join(sorted(merged[f"served_model_{model_alias}"].dropna().astype(str).unique())),
                "requested_rows": int(total_unique_requested),
                "completed_rows": int(valid.sum()),
                "completion_rate": float(valid.mean()),
                "raw_ok_records": int(ok_rows),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
                "pearson": safe_corr(pearsonr, y_true, y_pred),
                "spearman": safe_corr(spearmanr, y_true, y_pred),
                "mean_bias_pred_minus_true": float(np.mean(y_pred - y_true)),
                "n_unique_predictions": int(np.unique(y_pred).size),
                "linear_calibration_intercept": float(linear_calibrator.intercept_),
                "linear_calibration_slope": float(linear_calibrator.coef_[0]),
                "in_sample_calibrated_mae": float(mean_absolute_error(y_true, calibrated_prediction)),
                "in_sample_calibrated_r2": float(r2_score(y_true, calibrated_prediction)),
                "toxicity_auc_at_0.5": float(roc_auc_score(y_true >= 0.5, y_pred)) if np.unique(y_true >= 0.5).size == 2 else float("nan"),
                "boundary_disagreement": float(np.mean(disagreement)),
                "boundary_side_agreement": float(1.0 - np.mean(disagreement)),
                "mean_abs_pinball_proxy_gap": float(np.mean(np.abs(synthetic_pinball - true_pinball))),
                "latency_p50_seconds": float(np.nanmedian(merged[f"latency_{model_alias}"])),
                "latency_p90_seconds": float(np.nanquantile(merged[f"latency_{model_alias}"], 0.9)),
                "mean_attempts": float(np.nanmean(merged[f"attempts_{model_alias}"])),
                "prompt_tokens": int(records["prompt_tokens"].fillna(0).sum()),
                "completion_tokens": int(records["completion_tokens"].fillna(0).sum()),
                "cache_hit_tokens": int(records["prompt_cache_hit_tokens"].fillna(0).sum()),
                "cache_miss_tokens": int(records["prompt_cache_miss_tokens"].fillna(0).sum()),
                "estimated_api_cost_usd": calculate_cost(records, model_alias),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "pilot_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_by_true_bin.csv", index=False)
    merged.to_parquet(output_dir / "pilot_predictions.parquet", index=False)

    paired = merged.dropna(subset=["toxicity_flash", "toxicity_pro"]).copy()
    flash_abs_error = np.abs(paired["toxicity_flash"].to_numpy() - paired["toxicity"].to_numpy())
    pro_abs_error = np.abs(paired["toxicity_pro"].to_numpy() - paired["toxicity"].to_numpy())
    mae_diff = bootstrap_difference(
        flash_abs_error,
        pro_abs_error,
        np.mean,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    boundary_eval_frame = merged.iloc[boundary_eval].copy()
    boundary_true = true_score_eval <= q_eval
    flash_disagreement = (boundary_eval_frame["score_flash"].to_numpy() <= q_eval) != boundary_true
    pro_disagreement = (boundary_eval_frame["score_pro"].to_numpy() <= q_eval) != boundary_true
    boundary_diff = bootstrap_difference(
        flash_disagreement.astype(float),
        pro_disagreement.astype(float),
        np.mean,
        n_bootstrap=args.bootstrap,
        seed=args.seed + 1,
    )

    comparison = {
        "pilot_rows": int(len(pilot)),
        "paired_complete_rows": int(len(paired)),
        "true_toxicity_mean": float(pilot["toxicity"].mean()),
        "true_toxicity_std": float(pilot["toxicity"].std()),
        "true_toxicity_ge_0_5_rate": float(pilot["toxicity"].ge(0.5).mean()),
        "zero_predictor_mae": float(mean_absolute_error(pilot["toxicity"], np.zeros(len(pilot)))),
        "mean_predictor_mae": float(
            mean_absolute_error(
                pilot["toxicity"], np.full(len(pilot), float(pilot["toxicity"].mean()))
            )
        ),
        "flash_pro_exact_agreement_rate": float(
            np.mean(paired["toxicity_flash"].to_numpy() == paired["toxicity_pro"].to_numpy())
        ),
        "flash_pro_mean_absolute_difference": float(
            np.mean(
                np.abs(
                    paired["toxicity_flash"].to_numpy()
                    - paired["toxicity_pro"].to_numpy()
                )
            )
        ),
        "boundary_cal_rows": int(len(boundary_cal)),
        "boundary_eval_rows": int(len(boundary_eval)),
        "flash_minus_pro_mae": {"estimate": mae_diff[0], "ci95": [mae_diff[1], mae_diff[2]]},
        "flash_minus_pro_boundary_disagreement": {
            "estimate": boundary_diff[0],
            "ci95": [boundary_diff[1], boundary_diff[2]],
        },
    }
    with (output_dir / "paired_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for model_alias, color in [("flash", "#3478BF"), ("pro", "#D95F59")]:
        axes[0].scatter(
            merged["toxicity"],
            merged[f"toxicity_{model_alias}"],
            alpha=0.35,
            s=18,
            label=model_alias,
            color=color,
        )
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axes[0].set(xlabel="Human toxicity fraction", ylabel="Synthetic toxicity", title="Label agreement")
    axes[0].legend()

    metric_view = summary.set_index("model")
    axes[1].bar(metric_view.index, metric_view["mae"], color=["#3478BF", "#D95F59"])
    axes[1].set(title="Label MAE", ylabel="MAE")
    axes[2].bar(
        metric_view.index,
        metric_view["boundary_disagreement"],
        color=["#3478BF", "#D95F59"],
    )
    axes[2].set(title="Local score-boundary disagreement", ylabel="Disagreement rate")
    fig.tight_layout()
    fig.savefig(output_dir / "pilot_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(json.dumps(comparison, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
