#!/usr/bin/env python3
"""Audit five-dimensional DeepSeek labels for the HelpSteer2 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold


LABELS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
PRICING = {
    "flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/helpsteer2.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/helpsteer2_embeddings.npz"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok" and isinstance(record.get("ratings"), dict):
            records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError(f"No successful records in {path}")
    return result.drop_duplicates("row_id", keep="last")


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, model: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, label in enumerate(LABELS):
        truth = y_true[:, index]
        pred = y_pred[:, index]
        rho = spearmanr(truth, pred).statistic
        rows.append(
            {
                "model": model,
                "dimension": label,
                "mae": mean_absolute_error(truth, pred),
                "rmse": mean_squared_error(truth, pred) ** 0.5,
                "r2": r2_score(truth, pred),
                "spearman": float(rho) if np.isfinite(rho) else np.nan,
                "bias": float(np.mean(pred - truth)),
                "exact_match": float(np.mean(pred == truth)),
                "within_one": float(np.mean(np.abs(pred - truth) <= 1)),
            }
        )
    rows.append(
        {
            "model": model,
            "dimension": "macro",
            "mae": mean_absolute_error(y_true.ravel(), y_pred.ravel()),
            "rmse": mean_squared_error(y_true.ravel(), y_pred.ravel()) ** 0.5,
            "r2": float(np.mean([r2_score(y_true[:, k], y_pred[:, k]) for k in range(len(LABELS))])),
            "spearman": float(
                np.nanmean([spearmanr(y_true[:, k], y_pred[:, k]).statistic for k in range(len(LABELS))])
            ),
            "bias": float(np.mean(y_pred - y_true)),
            "exact_match": float(np.mean(y_pred == y_true)),
            "within_one": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        }
    )
    return rows


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    n = len(values)
    estimates = np.empty(draws)
    for draw in range(draws):
        estimates[draw] = np.mean(values[rng.integers(0, n, n)])
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data)
    pilot = data.loc[data["role"].eq("pilot")].copy()
    base = data.loc[data["role"].eq("base_train")].copy()

    prediction_frames = []
    usage_rows = []
    for alias in ["flash", "pro"]:
        raw = load_jsonl(args.results_dir / f"{alias}_labels.jsonl")
        if len(raw) != len(pilot):
            raise ValueError(f"{alias}: expected {len(pilot)} predictions, found {len(raw)}")
        ratings = pd.DataFrame(raw["ratings"].tolist(), index=raw.index)[LABELS].astype(float)
        pred = pd.concat([raw[["row_id"]].reset_index(drop=True), ratings.reset_index(drop=True)], axis=1)
        pred = pred.rename(columns={label: f"syn_{alias}_{label}" for label in LABELS})
        prediction_frames.append(pred)

        hit = float(raw["prompt_cache_hit_tokens"].fillna(0).sum())
        miss = float(raw["prompt_cache_miss_tokens"].fillna(raw["prompt_tokens"]).sum())
        completion = float(raw["completion_tokens"].fillna(0).sum())
        price = PRICING[alias]
        cost = (hit * price["cache_hit"] + miss * price["cache_miss"] + completion * price["output"]) / 1e6
        usage_rows.append(
            {
                "model": alias,
                "rows": int(len(raw)),
                "prompt_tokens": int(raw["prompt_tokens"].fillna(0).sum()),
                "cache_hit_tokens": int(hit),
                "cache_miss_tokens": int(miss),
                "completion_tokens": int(completion),
                "estimated_cost_usd": cost,
                "mean_latency_seconds": float(raw["latency_seconds"].mean()),
                "retries": int((raw["attempts"] - 1).clip(lower=0).sum()),
                "served_models": ",".join(sorted(raw["served_model"].dropna().astype(str).unique())),
            }
        )

    merged = pilot.merge(prediction_frames[0], on="row_id", how="inner", validate="one_to_one")
    merged = merged.merge(prediction_frames[1], on="row_id", how="inner", validate="one_to_one")
    y_true = merged[LABELS].to_numpy(float)
    constant = np.tile(base[LABELS].median().to_numpy(float), (len(merged), 1))
    metric_data = metric_rows(y_true, constant, "base_median")
    for alias in ["flash", "pro"]:
        y_pred = merged[[f"syn_{alias}_{label}" for label in LABELS]].to_numpy(float)
        metric_data.extend(metric_rows(y_true, y_pred, alias))

    embeddings = np.load(args.embeddings)
    row_to_embedding = {str(row): embedding for row, embedding in zip(embeddings["row_ids"], embeddings["features"])}
    x_base = np.stack([row_to_embedding[row] for row in base["row_id"]])
    x_pilot = np.stack([row_to_embedding[row] for row in merged["row_id"]])
    y_base = base[LABELS].to_numpy(float) / 4.0
    y_pilot = y_true / 4.0
    mean_model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_base, y_base)
    mu_pilot = np.clip(mean_model.predict(x_pilot), 0, 1)
    true_scores = np.max(np.abs(y_pilot - mu_pilot), axis=1)

    groups = merged["prompt_key"].to_numpy()
    folds = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    q_crossfit = np.empty(len(merged))
    for train_index, valid_index in splitter.split(x_pilot, true_scores, groups=groups):
        quantile_model = GradientBoostingRegressor(
            loss="quantile", alpha=0.9, n_estimators=160, max_depth=2,
            min_samples_leaf=12, learning_rate=0.04, random_state=args.seed,
        )
        quantile_model.fit(x_pilot[train_index], true_scores[train_index])
        q_crossfit[valid_index] = np.clip(quantile_model.predict(x_pilot[valid_index]), 0, 1)

    boundary_rows = []
    true_above = true_scores > q_crossfit
    boundary_rng = np.random.default_rng(args.seed + 1)
    boundary_predictions = {
        "base_median": np.tile(base[LABELS].median().to_numpy(float) / 4.0, (len(merged), 1)),
        **{
            alias: merged[[f"syn_{alias}_{label}" for label in LABELS]].to_numpy(float) / 4.0
            for alias in ["flash", "pro"]
        },
    }
    for alias, syn_y in boundary_predictions.items():
        syn_scores = np.max(np.abs(syn_y - mu_pilot), axis=1)
        syn_above = syn_scores > q_crossfit
        disagreement = true_above != syn_above
        shuffled_disagreement = np.empty(2000)
        for draw in range(len(shuffled_disagreement)):
            shuffled_syn_above = syn_scores[boundary_rng.permutation(len(syn_scores))] > q_crossfit
            shuffled_disagreement[draw] = np.mean(true_above != shuffled_syn_above)
        shuffled_mean = float(shuffled_disagreement.mean())
        boundary_rows.append(
            {
                "model": alias,
                "boundary_disagreement": float(disagreement.mean()),
                "boundary_agreement": float(1 - disagreement.mean()),
                "true_above_rate": float(true_above.mean()),
                "synthetic_above_rate": float(syn_above.mean()),
                "always_below_disagreement": float(true_above.mean()),
                "shuffled_pair_disagreement": shuffled_mean,
                "relative_reduction_vs_shuffled": float(
                    (shuffled_mean - disagreement.mean()) / shuffled_mean
                ),
                "balanced_accuracy": float(balanced_accuracy_score(true_above, syn_above)),
                "above_boundary_precision": float(precision_score(true_above, syn_above, zero_division=0)),
                "above_boundary_recall": float(recall_score(true_above, syn_above, zero_division=0)),
                "above_boundary_f1": float(f1_score(true_above, syn_above, zero_division=0)),
                "matthews_correlation": float(matthews_corrcoef(true_above, syn_above)),
                "cohen_kappa": float(cohen_kappa_score(true_above, syn_above)),
                "score_mae": float(np.mean(np.abs(syn_scores - true_scores))),
                "true_score_mean": float(true_scores.mean()),
                "synthetic_score_mean": float(syn_scores.mean()),
            }
        )

    rng = np.random.default_rng(args.seed)
    flash_abs = np.mean(
        np.abs(merged[[f"syn_flash_{label}" for label in LABELS]].to_numpy(float) - y_true), axis=1
    )
    pro_abs = np.mean(
        np.abs(merged[[f"syn_pro_{label}" for label in LABELS]].to_numpy(float) - y_true), axis=1
    )
    mae_difference = pro_abs - flash_abs
    mae_ci = bootstrap_interval(mae_difference, rng, args.bootstrap)
    comparison = {
        "pro_minus_flash_macro_mae": float(mae_difference.mean()),
        "pro_minus_flash_macro_mae_ci95": list(mae_ci),
        "interpretation": "positive values favor Flash",
    }

    merged.to_parquet(args.results_dir / "pilot_predictions.parquet", index=False)
    pd.DataFrame(metric_data).to_csv(args.results_dir / "label_metrics.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(args.results_dir / "boundary_metrics.csv", index=False)
    pd.DataFrame(usage_rows).to_csv(args.results_dir / "api_usage.csv", index=False)
    summary = {
        "rows": int(len(merged)),
        "metrics": pd.DataFrame(metric_data).round(6).to_dict(orient="records"),
        "boundary": pd.DataFrame(boundary_rows).round(6).to_dict(orient="records"),
        "usage": pd.DataFrame(usage_rows).round(8).to_dict(orient="records"),
        "paired_comparison": comparison,
    }
    (args.results_dir / "label_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
