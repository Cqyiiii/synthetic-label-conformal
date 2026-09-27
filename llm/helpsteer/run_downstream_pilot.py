#!/usr/bin/env python3
"""Run a multi-output HelpSteer2 RCP/PPI screening experiment."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn


LABELS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
PRICING = {
    "flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/helpsteer2.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/helpsteer2_embeddings.npz"))
    parser.add_argument("--predictions", type=Path, default=Path("results/pilot_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/downstream"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--paired-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--mean-epochs", type=int, default=350)
    parser.add_argument("--quantile-epochs", type=int, default=260)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--include-cost-match", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MeanNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantileNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pinball(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    residual = target - pred
    return torch.maximum(tau * residual, (tau - 1.0) * residual)


def fit_mean(x: np.ndarray, y: np.ndarray, epochs: int, device: str) -> MeanNet:
    seed_everything(81723)
    model = MeanNet(x.shape[1]).to(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=0.15)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
    return model.cpu().eval()


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy()


def fit_quantile(
    initial_state: dict[str, torch.Tensor],
    x_pair: np.ndarray,
    true_pair: np.ndarray,
    syn_pair: np.ndarray,
    x_aux: np.ndarray,
    syn_aux: np.ndarray,
    method: str,
    tau: float,
    epochs: int,
    x_validation: np.ndarray,
    true_validation: np.ndarray,
) -> tuple[QuantileNet, int]:
    model = QuantileNet(x_pair.shape[1])
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=2e-4)
    xp = torch.as_tensor(x_pair, dtype=torch.float32)
    tp = torch.as_tensor(true_pair, dtype=torch.float32)
    sp = torch.as_tensor(syn_pair, dtype=torch.float32)
    xa = torch.as_tensor(x_aux, dtype=torch.float32)
    sa = torch.as_tensor(syn_aux, dtype=torch.float32)
    xv = torch.as_tensor(x_validation, dtype=torch.float32)
    tv = torch.as_tensor(true_validation, dtype=torch.float32)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        if method == "rcp":
            loss = pinball(model(xp), tp, tau).mean()
        elif method == "synonly":
            loss = pinball(model(xa), sa, tau).mean()
        elif method == "ppi":
            loss = pinball(model(xa), sa, tau).mean()
            loss = loss + (pinball(model(xp), tp, tau) - pinball(model(xp), sp, tau)).mean()
        else:
            raise ValueError(method)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(pinball(model(xv), tv, tau).mean())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        model.train()
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


def conformal_quantile(values: np.ndarray, tau: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(len(ordered), int(np.ceil((len(ordered) + 1) * tau)))
    return float(ordered[rank - 1])


def pinball_numpy(pred: np.ndarray, target: np.ndarray, tau: float) -> np.ndarray:
    residual = target - pred
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def grouped_coverage(covered: np.ndarray, groups: np.ndarray, tau: float) -> tuple[float, float]:
    coverages = np.array([covered[groups == group].mean() for group in np.unique(groups)])
    return float(np.mean((coverages - tau) ** 2)), float(np.min(coverages))


def evaluate(
    q_test: np.ndarray,
    q_conformal: np.ndarray,
    score_test: np.ndarray,
    score_conformal: np.ndarray,
    mu_test: np.ndarray,
    groups: np.ndarray,
    tau: float,
) -> dict[str, float]:
    pre_covered = score_test <= q_test
    pre_msce, pre_min = grouped_coverage(pre_covered, groups, tau)
    gamma = conformal_quantile(score_conformal - q_conformal, tau)
    widths = np.clip(q_test + gamma, 0, 1)
    covered = score_test <= widths
    msce, minimum = grouped_coverage(covered, groups, tau)
    lower = np.clip(mu_test - widths[:, None], 0, 1)
    upper = np.clip(mu_test + widths[:, None], 0, 1)
    dimension_widths = np.maximum(upper - lower, 1e-8)
    return {
        "true_pinball": float(pinball_numpy(q_test, score_test, tau).mean()),
        "pre_coverage": float(pre_covered.mean()),
        "pre_msce": pre_msce,
        "pre_min_group_coverage": pre_min,
        "gamma": gamma,
        "coverage": float(covered.mean()),
        "msce": msce,
        "min_group_coverage": minimum,
        "mean_radius": float(widths.mean()),
        "mean_log_volume": float(np.log(dimension_widths).sum(axis=1).mean()),
    }


def paired_bootstrap(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    statistics = np.empty(draws)
    for index in range(draws):
        statistics[index] = values[rng.integers(0, n, n)].mean()
    return tuple(np.quantile(statistics, [0.025, 0.975]).tolist())


def load_row_costs(path: Path, alias: str) -> dict[str, float]:
    price = PRICING[alias]
    costs: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        hit = float(record.get("prompt_cache_hit_tokens") or 0)
        miss = float(record.get("prompt_cache_miss_tokens") or record.get("prompt_tokens") or 0)
        completion = float(record.get("completion_tokens") or 0)
        costs[str(record["row_id"])] = (
            hit * price["cache_hit"] + miss * price["cache_miss"] + completion * price["output"]
        ) / 1e6
    return costs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data)
    predictions = pd.read_parquet(args.predictions)
    data = data.merge(
        predictions[["row_id", *[f"syn_{alias}_{label}" for alias in ["flash", "pro"] for label in LABELS]]],
        on="row_id", how="left", validate="one_to_one",
    )
    embedding_file = np.load(args.embeddings)
    row_to_embedding = {
        str(row_id): vector.astype(np.float32)
        for row_id, vector in zip(embedding_file["row_ids"], embedding_file["features"])
    }

    def role_arrays(role: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        frame = data.loc[data["role"].eq(role)].sort_values("row_id").copy()
        x = np.stack([row_to_embedding[row_id] for row_id in frame["row_id"]])
        y = frame[LABELS].to_numpy(np.float32) / 4.0
        return frame, x, y

    base, x_base, y_base = role_arrays("base_train")
    pilot, x_pilot, y_pilot = role_arrays("pilot")
    conformal_all, x_conformal_all, y_conformal_all = role_arrays("conformal")
    test, x_test, y_test = role_arrays("test")
    if len(pilot) - args.paired_size < 1:
        raise ValueError("Pilot requires at least one auxiliary-only point")

    if args.device != "auto":
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    mean_model = fit_mean(x_base, y_base, args.mean_epochs, device)
    mu_base = predict(mean_model, x_base)
    mu_pilot = predict(mean_model, x_pilot)
    mu_conformal_all = predict(mean_model, x_conformal_all)
    mu_test = predict(mean_model, x_test)
    base_mae = float(np.abs(mu_base - y_base).mean())
    test_mae = float(np.abs(mu_test - y_test).mean())

    score_pilot = np.max(np.abs(y_pilot - mu_pilot), axis=1)
    score_conformal_all = np.max(np.abs(y_conformal_all - mu_conformal_all), axis=1)
    score_test = np.max(np.abs(y_test - mu_test), axis=1)
    syn_scores: dict[str, np.ndarray] = {}
    for alias in ["flash", "pro"]:
        synthetic_y = pilot[[f"syn_{alias}_{label}" for label in LABELS]].to_numpy(np.float32) / 4.0
        if not np.isfinite(synthetic_y).all():
            raise ValueError(f"Missing {alias} synthetic labels")
        syn_scores[alias] = np.max(np.abs(synthetic_y - mu_pilot), axis=1)
    row_costs = {
        alias: np.array(
            [load_row_costs(Path("results") / f"{alias}_labels.jsonl", alias)[row_id] for row_id in pilot["row_id"]]
        )
        for alias in ["flash", "pro"]
    }
    flash_full_budget = float(row_costs["flash"].sum())

    groups = KMeans(n_clusters=10, n_init=20, random_state=20260804).fit_predict(x_test)
    development_keys = conformal_all["prompt_key"].drop_duplicates().to_numpy().copy()
    development_rng = np.random.default_rng(20260804)
    development_rng.shuffle(development_keys)
    selected_keys: list[str] = []
    selected_rows = 0
    prompt_sizes = conformal_all.groupby("prompt_key").size()
    for key in development_keys:
        if selected_rows >= 100:
            break
        selected_keys.append(str(key))
        selected_rows += int(prompt_sizes.loc[key])
    development_mask = conformal_all["prompt_key"].isin(selected_keys).to_numpy()
    x_development = x_conformal_all[development_mask]
    score_development = score_conformal_all[development_mask]
    x_conformal = x_conformal_all[~development_mask]
    score_conformal = score_conformal_all[~development_mask]
    records: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        indices = rng.permutation(len(pilot))
        paired = indices[: args.paired_size]
        auxiliary = indices[args.paired_size :]
        seed_everything(91000 + seed)
        initial_model = QuantileNet(x_pilot.shape[1])
        initial_state = copy.deepcopy(initial_model.state_dict())

        method_specs: list[tuple[str, str, str | None, np.ndarray]] = [
            ("RCP", "rcp", None, np.array([], dtype=int))
        ]
        for alias in ["flash", "pro"]:
            method_specs.extend(
                [
                    (f"{alias.title()}-SynOnly", "synonly", alias, auxiliary),
                    (f"{alias.title()}-PPI", "ppi", alias, auxiliary),
                ]
            )
        if args.include_cost_match:
            remaining_budget = flash_full_budget - float(row_costs["pro"][paired].sum())
            cumulative = np.cumsum(row_costs["pro"][auxiliary])
            matched_auxiliary = auxiliary[cumulative <= max(remaining_budget, 0.0)]
            if len(matched_auxiliary) == 0:
                raise ValueError("The cost-matched Pro auxiliary pool is empty")
            method_specs.append(("Pro-PPI-CostMatched", "ppi", "pro", matched_auxiliary))
        for display_name, method, alias, method_auxiliary in method_specs:
            if alias is None:
                synthetic = np.zeros_like(score_pilot)
            else:
                synthetic = syn_scores[alias]
            quantile_model, selected_epoch = fit_quantile(
                initial_state,
                x_pilot[paired], score_pilot[paired], synthetic[paired],
                x_pilot[method_auxiliary], synthetic[method_auxiliary], method,
                args.tau, args.quantile_epochs,
                x_development, score_development,
            )
            q_test = predict(quantile_model, x_test)
            q_conformal = predict(quantile_model, x_conformal)
            metrics = evaluate(
                q_test, q_conformal, score_test, score_conformal, mu_test, groups, args.tau
            )
            if alias is None:
                query_cost = 0.0
            elif method == "synonly":
                query_cost = float(row_costs[alias][method_auxiliary].sum())
            else:
                query_cost = float(row_costs[alias][np.concatenate([paired, method_auxiliary])].sum())
            records.append(
                {
                    "seed": seed, "method": display_name, "selected_epoch": selected_epoch,
                    "auxiliary_size": len(method_auxiliary), "query_cost_usd": query_cost, **metrics,
                }
            )

    raw = pd.DataFrame(records)
    raw.to_csv(args.output_dir / "raw_metrics.csv", index=False)
    metrics = [
        "true_pinball", "pre_coverage", "pre_msce", "pre_min_group_coverage",
        "coverage", "msce", "min_group_coverage", "mean_radius", "mean_log_volume",
    ]
    summary = raw.groupby("method")[["auxiliary_size", "query_cost_usd", *metrics]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    comparisons = []
    rng = np.random.default_rng(20260804)
    baseline = raw.loc[raw["method"].eq("RCP")].set_index("seed")
    candidate_methods = [method for method in raw["method"].unique() if method != "RCP"]
    for method in candidate_methods:
        candidate = raw.loc[raw["method"].eq(method)].set_index("seed")
        for metric in ["true_pinball", "pre_msce", "msce", "min_group_coverage", "mean_log_volume"]:
            difference = candidate[metric] - baseline[metric]
            low, high = paired_bootstrap(difference.to_numpy(), args.bootstrap, rng)
            comparisons.append(
                {
                    "method": method,
                    "metric": metric,
                    "difference_vs_rcp": float(difference.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "direction": "negative is better" if metric not in ["min_group_coverage"] else "positive is better",
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    result = {
        "settings": {
            "base_train": len(base), "paired": args.paired_size,
            "auxiliary": len(pilot) - args.paired_size,
            "quantile_development": len(x_development), "conformal": len(x_conformal),
            "test": len(test), "seeds": args.seeds, "tau": args.tau,
            "feature_dimension": x_test.shape[1], "device_for_mean_model": device,
        },
        "mean_model": {"base_mae": base_mae, "test_mae": test_mae},
        "summary": summary.round(7).to_dict(orient="records"),
        "paired_comparisons": comparison_frame.round(7).to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
