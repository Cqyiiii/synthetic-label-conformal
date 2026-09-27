#!/usr/bin/env python3
"""Run a small downstream RCP/PPI experiment using the 500 paired LLM labels."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


TAU = 0.9


class Net(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/civil_comments_20k.parquet"))
    parser.add_argument("--predictions", type=Path, default=Path("analysis/pilot_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("downstream"))
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--paired-size", type=int, default=100)
    parser.add_argument("--conformal-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--feature-cache", type=Path, default=Path("data/tfidf_svd64_features.npy"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pinball(prediction: torch.Tensor, target: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(tau * residual, (tau - 1.0) * residual)


def fit_mean(model: Net, x: torch.Tensor, y: torch.Tensor, epochs: int) -> Net:
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.HuberLoss(delta=1.0)(model(x), y)
        loss.backward()
        optimizer.step()
    return model


def fit_quantile(
    model: Net,
    mode: str,
    x_pair: torch.Tensor,
    score_pair_true: torch.Tensor,
    score_pair_syn: torch.Tensor,
    x_aux: torch.Tensor,
    score_aux_syn: torch.Tensor,
    epochs: int,
) -> Net:
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        if mode == "supervised":
            loss = pinball(model(x_pair), score_pair_true).mean()
        elif mode == "synthetic_only":
            loss = pinball(model(x_aux), score_aux_syn).mean()
        elif mode == "ppi":
            loss = pinball(model(x_pair), score_pair_true).mean() + (
                pinball(model(x_aux), score_aux_syn).mean()
                - pinball(model(x_pair), score_pair_syn).mean()
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        loss.backward()
        optimizer.step()
    return model


def finite_sample_quantile(values: np.ndarray, tau: float = TAU) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(ordered) + 1) * tau))
    return float(ordered[min(max(rank - 1, 0), len(ordered) - 1)])


def msce_from_labels(covered: np.ndarray, cluster_labels: np.ndarray, n_clusters: int) -> float:
    total = len(covered)
    error = 0.0
    for cluster in range(n_clusters):
        mask = cluster_labels == cluster
        if mask.any():
            error += float(mask.mean()) * float((covered[mask].mean() - TAU) ** 2)
    return error


def get_features(data: pd.DataFrame, cache_path: Path, feature_dim: int) -> np.ndarray:
    resolved = cache_path.resolve()
    if resolved.exists():
        features = np.load(resolved)
        if features.shape == (len(data), feature_dim):
            return features

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    sparse = vectorizer.fit_transform(data["text"])
    svd = TruncatedSVD(n_components=feature_dim, random_state=20260803)
    features = svd.fit_transform(sparse)
    features = StandardScaler().fit_transform(features).astype(np.float32)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.save(resolved, features)
    return features


def main() -> None:
    args = parse_args()
    data = pd.read_parquet(args.data.resolve()).reset_index(drop=True)
    predictions = pd.read_parquet(args.predictions.resolve())[
        ["row_id", "toxicity_flash", "toxicity_pro"]
    ]
    data = data.merge(predictions, on="row_id", how="left")

    features = get_features(data, args.feature_cache, args.feature_dim)
    base_indices = np.flatnonzero(data["role"].eq("base_train").to_numpy())
    pilot_indices = np.flatnonzero(data["role"].eq("pilot").to_numpy())
    test_indices = np.flatnonzero(data["role"].eq("pool").to_numpy())
    if args.paired_size + args.conformal_size >= len(pilot_indices):
        raise ValueError("paired-size + conformal-size must leave a nonempty auxiliary pool")

    x_base = torch.tensor(features[base_indices], dtype=torch.float32)
    y_base = torch.tensor(data.iloc[base_indices]["toxicity"].to_numpy()[:, None], dtype=torch.float32)
    x_pilot_all = features[pilot_indices]
    y_pilot_all = data.iloc[pilot_indices]["toxicity"].to_numpy()
    x_test = torch.tensor(features[test_indices], dtype=torch.float32)
    y_test = data.iloc[test_indices]["toxicity"].to_numpy()

    n_clusters = 10
    cluster_labels = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=20260803,
    ).fit_predict(features[test_indices])

    output_rows: list[dict] = []
    methods = [
        ("RCP", "supervised", None),
        ("Flash-SynOnly", "synthetic_only", "flash"),
        ("Flash-PPI", "ppi", "flash"),
        ("Pro-SynOnly", "synthetic_only", "pro"),
        ("Pro-PPI", "ppi", "pro"),
    ]

    for seed in range(args.n_seeds):
        set_seed(42 + seed)
        permutation = np.random.default_rng(42 + seed).permutation(len(pilot_indices))
        pair_local = permutation[: args.paired_size]
        conf_local = permutation[args.paired_size : args.paired_size + args.conformal_size]
        aux_local = permutation[args.paired_size + args.conformal_size :]

        mean_model = Net(args.feature_dim)
        fit_mean(mean_model, x_base, y_base, args.epochs)
        mean_model.eval()
        with torch.no_grad():
            mu_pilot = mean_model(torch.tensor(x_pilot_all, dtype=torch.float32)).numpy().reshape(-1)
            mu_test = mean_model(x_test).numpy().reshape(-1)

        score_true_pilot = np.abs(y_pilot_all - mu_pilot)
        score_true_test = np.abs(y_test - mu_test)
        x_pair = torch.tensor(x_pilot_all[pair_local], dtype=torch.float32)
        x_conf = torch.tensor(x_pilot_all[conf_local], dtype=torch.float32)
        x_aux = torch.tensor(x_pilot_all[aux_local], dtype=torch.float32)
        score_pair_true = torch.tensor(score_true_pilot[pair_local, None], dtype=torch.float32)
        score_conf_true = score_true_pilot[conf_local]

        set_seed(10_000 + seed)
        initial_quantile_model = Net(args.feature_dim)
        initial_state = copy.deepcopy(initial_quantile_model.state_dict())

        for method_name, mode, labeler in methods:
            quantile_model = Net(args.feature_dim)
            quantile_model.load_state_dict(initial_state)

            if labeler is None:
                score_pair_syn = score_pair_true
                score_aux_syn = torch.tensor(score_true_pilot[aux_local, None], dtype=torch.float32)
            else:
                synthetic_y = data.iloc[pilot_indices][f"toxicity_{labeler}"].to_numpy()
                synthetic_score = np.abs(synthetic_y - mu_pilot)
                score_pair_syn = torch.tensor(synthetic_score[pair_local, None], dtype=torch.float32)
                score_aux_syn = torch.tensor(synthetic_score[aux_local, None], dtype=torch.float32)

            fit_quantile(
                quantile_model,
                mode=mode,
                x_pair=x_pair,
                score_pair_true=score_pair_true,
                score_pair_syn=score_pair_syn,
                x_aux=x_aux,
                score_aux_syn=score_aux_syn,
                epochs=args.epochs,
            )
            quantile_model.eval()
            with torch.no_grad():
                q_conf = quantile_model(x_conf).numpy().reshape(-1)
                q_test = quantile_model(x_test).numpy().reshape(-1)

            true_pinball = np.maximum(
                TAU * (score_true_test - q_test),
                (TAU - 1.0) * (score_true_test - q_test),
            ).mean()
            pre_covered = score_true_test <= q_test
            gamma = finite_sample_quantile(score_conf_true - q_conf)
            final_width = np.maximum(q_test + gamma, 0.0)
            covered = score_true_test <= final_width

            output_rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "true_pinball": float(true_pinball),
                    "pre_msce_10": msce_from_labels(pre_covered, cluster_labels, n_clusters),
                    "coverage": float(covered.mean()),
                    "msce_10": msce_from_labels(covered, cluster_labels, n_clusters),
                    "mean_interval_width": float(2.0 * final_width.mean()),
                    "gamma": gamma,
                    "paired_size": args.paired_size,
                    "conformal_size": args.conformal_size,
                    "auxiliary_size": int(len(aux_local)),
                }
            )
        print(f"seed {seed + 1}/{args.n_seeds} complete", flush=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(output_rows)
    raw.to_csv(output_dir / "downstream_raw.csv", index=False)
    summary = raw.groupby("method", sort=False).agg(
        true_pinball_mean=("true_pinball", "mean"),
        true_pinball_std=("true_pinball", "std"),
        pre_msce_10_mean=("pre_msce_10", "mean"),
        coverage_mean=("coverage", "mean"),
        msce_10_mean=("msce_10", "mean"),
        msce_10_std=("msce_10", "std"),
        mean_interval_width=("mean_interval_width", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "downstream_summary.csv", index=False)

    paired_differences: dict[str, dict[str, float]] = {}
    baseline = raw.loc[raw["method"].eq("RCP")].set_index("seed")
    for method_name in [name for name, _, _ in methods if name != "RCP"]:
        candidate = raw.loc[raw["method"].eq(method_name)].set_index("seed")
        paired_differences[method_name] = {
            "pinball_improvement_vs_rcp": float(
                (baseline["true_pinball"] - candidate["true_pinball"]).mean()
            ),
            "msce_improvement_vs_rcp": float(
                (baseline["msce_10"] - candidate["msce_10"]).mean()
            ),
            "coverage_change_vs_rcp": float(
                (candidate["coverage"] - baseline["coverage"]).mean()
            ),
            "width_change_vs_rcp": float(
                (candidate["mean_interval_width"] - baseline["mean_interval_width"]).mean()
            ),
        }
    with (output_dir / "paired_differences.json").open("w", encoding="utf-8") as handle:
        json.dump(paired_differences, handle, indent=2)

    print(summary.to_string(index=False))
    print(json.dumps(paired_differences, indent=2))
    print(f"Saved downstream pilot to {output_dir}")


if __name__ == "__main__":
    main()
