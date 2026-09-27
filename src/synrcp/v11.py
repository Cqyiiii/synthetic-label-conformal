from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .conformal import (
    box_volume_summary,
    calibrate_scores,
    covered,
    covered_box,
    exact_order_statistic,
    interval_box_volume_summary,
)
from .io import hash_indices
from .real_data import (
    _column_medians,
    _coverage_metric_row,
    _evaluate_global_radius_method,
    _evaluate_threshold_predictor,
    _impute,
    _read_dataset,
)
from .synthetic_methods import (
    calibrate_cqr,
    semicp_nnm_threshold,
    spi_score_threshold,
)
from .v11_neural import (
    V11NeuralFit,
    fit_cqr_network,
    fit_mean_network,
    fit_threshold_network,
)


V11_DATASETS = [
    "bike",
    "diamond",
    "gas_turbine",
    "naval",
    "SGEMM",
    "superconduct",
    "Transcoding",
    "WEC",
]

V11_MAIN_METHODS = [
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
]

V11_ABLATION_METHODS = [
    "RCP",
    "RCP-Aug",
    "RCP-Aug-Permuted",
    "RCP-Extra",
    "RCP-Pool",
]


def _score(y: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.max(
        np.abs(np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)), axis=1
    )


def _pinball(score: np.ndarray, prediction: np.ndarray, tau: float) -> float:
    residual = np.asarray(score, dtype=float).reshape(-1) - np.asarray(
        prediction, dtype=float
    ).reshape(-1)
    return float(
        np.mean(np.maximum(float(tau) * residual, (float(tau) - 1.0) * residual))
    )


def _role_size(n: int, proportion: float, minimum: int) -> int:
    return max(int(minimum), int(math.floor(float(proportion) * int(n))))


def _allocate_v11_roles(
    n: int,
    seed: int,
    *,
    pool_max: float = 0.40,
    base_max: float = 0.04,
    conformal_max: float = 0.02,
    minimum: int = 30,
) -> dict[str, np.ndarray]:
    """Allocate fixed reservoirs so every sensitivity uses nested prefixes."""

    specs = [
        ("preprocessing", 0.01),
        ("teacher_train", 0.16),
        ("teacher_val", 0.04),
        ("synthetic_pool", float(pool_max)),
        ("base_train", float(base_max)),
        ("conformal", float(conformal_max)),
        ("partition", 0.03),
        ("slice_select", 0.03),
    ]
    sizes = {name: _role_size(n, prop, minimum) for name, prop in specs}
    if sum(sizes.values()) + minimum > n:
        raise ValueError("V11 role reservoirs leave insufficient evaluation data.")
    permutation = np.random.default_rng(int(seed)).permutation(n)
    roles: dict[str, np.ndarray] = {}
    cursor = 0
    for name, _ in specs:
        size = sizes[name]
        roles[name] = permutation[cursor : cursor + size]
        cursor += size
    roles["evaluation"] = permutation[cursor:]
    return roles


def _prefix(
    reservoir: np.ndarray, n_total: int, proportion: float, minimum: int = 30
) -> np.ndarray:
    size = min(len(reservoir), _role_size(n_total, proportion, minimum))
    return np.asarray(reservoir[:size], dtype=int)


def _load_v11_data(
    dataset: str,
    data_dir: Path,
    *,
    dataset_sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, y = _read_dataset(dataset, data_dir)
    original_rows = int(len(x))
    if dataset == "SGEMM" and len(x) > 50_000:
        rng = np.random.default_rng(int(dataset_sample_seed))
        selected = np.sort(rng.choice(len(x), size=50_000, replace=False))
        x, y = x[selected], y[selected]
    return x, y, {
        "original_rows": original_rows,
        "used_rows": int(len(x)),
        "dataset_sample_seed": int(dataset_sample_seed),
        "sgemm_cap": 50_000,
    }


def _fit_teacher(
    x: np.ndarray,
    y: np.ndarray,
    *,
    strength: str,
    seed: int,
) -> RandomForestRegressor | None:
    if strength == "oracle":
        return None
    params: dict[str, Any] = {
        "n_estimators": 50,
        "random_state": int(seed),
        "n_jobs": 1,
        "max_features": 1.0,
        "bootstrap": True,
    }
    if strength == "weak":
        params.update({"max_depth": 5, "max_leaf_nodes": 50, "min_samples_leaf": 5})
    elif strength == "moderate":
        params.update({"max_depth": None, "min_samples_leaf": 1})
    else:
        raise ValueError(f"Unknown V11 teacher strength: {strength}")
    model = RandomForestRegressor(**params)
    target = y.reshape(-1) if y.shape[1] == 1 else y
    model.fit(x, target)
    return model


def _teacher_predict(
    model: RandomForestRegressor | None,
    x: np.ndarray,
    *,
    oracle_y: np.ndarray | None = None,
) -> np.ndarray:
    if model is None:
        if oracle_y is None:
            raise ValueError("Oracle teacher requires the corresponding true outcomes.")
        return np.asarray(oracle_y, dtype=float).copy()
    prediction = np.asarray(model.predict(x), dtype=float)
    return prediction.reshape(-1, 1) if prediction.ndim == 1 else prediction


def _threshold_bound(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    return float(
        max(
            1.0,
            2.0 * float(np.max(values)),
            4.0 * float(np.quantile(values, 0.9)),
        )
    )


@dataclass(frozen=True)
class _CQRAdapter:
    fit: V11NeuralFit
    outcome_dim: int

    def endpoints(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.fit.predict(x)
        return (
            prediction[:, : self.outcome_dim],
            prediction[:, self.outcome_dim :],
        )


def _evaluate_cqr(
    *,
    method: str,
    model: _CQRAdapter,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    x_slice: np.ndarray,
    y_slice: np.ndarray,
    x_partition: np.ndarray,
    tau: float,
    seed: int,
    outcome_scale: np.ndarray,
) -> dict[str, Any]:
    lower_cal, upper_cal = model.endpoints(x_cal)
    calibration = calibrate_cqr(y_cal, lower_cal, upper_cal, tau)
    lower_eval, upper_eval = model.endpoints(x_eval)
    lower_slice, upper_slice = model.endpoints(x_slice)
    covered_eval = covered_box(
        y_eval, lower_eval, upper_eval, calibration.correction
    )
    covered_slice = covered_box(
        y_slice, lower_slice, upper_slice, calibration.correction
    )
    size = interval_box_volume_summary(
        lower_eval, upper_eval, calibration.correction, outcome_scale
    )
    return _coverage_metric_row(
        method=method,
        covered_eval=covered_eval,
        covered_slice=covered_slice,
        x_eval=x_eval,
        x_partition=x_partition,
        x_slice_select=x_slice,
        tau=tau,
        k_values=[10, 30],
        seed=seed,
        size_summary=size,
        calibration_correction=calibration.correction,
        calibration_k=calibration.order_index,
        calibration_m=calibration.calibration_size,
        calibration_infinite=calibration.infinite,
        include_ert=True,
        extra={
            "lambda": 1.0 if method == "CQR-PPI" else 0.0,
            "method_family": "CQR",
            "supervised_counterpart": "CQR",
            "synthetic_use": (
                "signed endpoint PPI" if method == "CQR-PPI" else "none"
            ),
            "method_diagnostics": model.fit.diagnostics,
        },
    )


def _evaluate_threshold(
    *,
    method: str,
    fit: V11NeuralFit,
    x_cal: np.ndarray,
    s_cal: np.ndarray,
    x_eval: np.ndarray,
    s_eval: np.ndarray,
    x_slice: np.ndarray,
    s_slice: np.ndarray,
    x_partition: np.ndarray,
    tau: float,
    seed: int,
    outcome_scale: np.ndarray,
    power: float | None,
    cv_losses: dict[str, float] | None = None,
) -> dict[str, Any]:
    row = _evaluate_threshold_predictor(
        method=method,
        threshold_predict=fit.predict,
        s_cal=s_cal,
        x_cal=x_cal,
        s_eval=s_eval,
        x_eval=x_eval,
        x_partition=x_partition,
        x_slice_select=x_slice,
        s_slice_select=s_slice,
        tau=tau,
        k_values=[10, 30],
        seed=seed,
        outcome_scale=outcome_scale,
        include_ert=True,
    )
    for key in [key for key in row if key.startswith("_profile_")]:
        row.pop(key, None)
    row.update(
        {
            "method_family": "RCP",
            "supervised_counterpart": "RCP",
            "synthetic_use": {
                "RCP": "none",
                "RCP-PTFT": "synthetic pretraining then trusted fine-tuning",
                "RCP-PPI": "signed prediction-powered threshold learning",
                "RCP-PPI-CV": "CV-selected signed prediction-powered learning",
                "RCP-Aug": "positive synthetic augmentation",
                "RCP-Aug-Permuted": "permuted synthetic-label placebo",
                "RCP-Extra": "additional true pool outcomes (resource ceiling)",
                "RCP-Pool": "unit pooling of trusted and synthetic scores",
            }.get(method, "synthetic threshold learning"),
            "lambda": power,
            "cv_losses": cv_losses or {},
            "method_diagnostics": fit.diagnostics,
        }
    )
    return row


def _skipped_row(
    method: str,
    family: str,
    counterpart: str,
    reason: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": method,
        "method_family": family,
        "supervised_counterpart": counterpart,
        "synthetic_use": "attempted",
        "skipped": True,
        "skip_reason": reason,
        "grouped": [],
        "method_diagnostics": diagnostics,
    }


def _fit_cv_threshold(
    *,
    x_learning: np.ndarray,
    s_learning: np.ndarray,
    s_paired: np.ndarray,
    x_pool: np.ndarray,
    s_pool: np.ndarray,
    tau: float,
    seed: int,
    hidden: int,
    epochs: int,
    cv_epochs: int,
    learning_rate: float,
    output_bound: float,
    grid: list[float],
) -> tuple[V11NeuralFit, float, dict[str, float]]:
    permutation = np.random.default_rng(int(seed) + 991).permutation(len(x_learning))
    split = max(2, int(math.floor(0.75 * len(permutation))))
    split = min(split, len(permutation) - 2)
    train, valid = permutation[:split], permutation[split:]
    losses: dict[str, float] = {}
    for power in grid:
        candidate = fit_threshold_network(
            x_learning[train],
            s_learning[train],
            tau=tau,
            seed=seed,
            objective="signed",
            power=float(power),
            x_pool=x_pool,
            score_pool=s_pool,
            score_paired=s_paired[train],
            hidden=hidden,
            epochs=cv_epochs,
            learning_rate=learning_rate,
            output_bound=output_bound,
        )
        losses[str(power)] = (
            _pinball(s_learning[valid], candidate.predict(x_learning[valid]), tau)
            if candidate.diagnostics["finite"]
            else float("inf")
        )
    selected = min(grid, key=lambda value: losses[str(value)])
    final = fit_threshold_network(
        x_learning,
        s_learning,
        tau=tau,
        seed=seed,
        objective="signed",
        power=float(selected),
        x_pool=x_pool,
        score_pool=s_pool,
        score_paired=s_paired,
        hidden=hidden,
        epochs=epochs,
        learning_rate=learning_rate,
        output_bound=output_bound,
    )
    return final, float(selected), losses


def run_v11_task(
    task: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    params = dict(task["params"])
    label_budget = params.get("label_budget")
    if label_budget == "ultra_low":
        params["base_prop"] = 0.01
        params["conformal_prop"] = 0.005
    elif label_budget == "moderate":
        params["base_prop"] = 0.04
        params["conformal_prop"] = 0.02
    elif label_budget is not None:
        raise ValueError(f"Unknown V11 label budget: {label_budget}")
    dataset = str(params["dataset"])
    if dataset not in V11_DATASETS:
        raise ValueError(f"Unknown V11 dataset: {dataset}")
    seed = int(params.get("seed", 0))
    alpha = float(params.get("alpha", 0.1))
    tau = 1.0 - alpha
    data_dir = Path(params.get("data_dir", "legacy/Datasets"))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    raw_x, raw_y, sample_meta = _load_v11_data(
        dataset,
        data_dir,
        dataset_sample_seed=int(params.get("dataset_sample_seed", 2026)),
    )
    n_total = len(raw_x)
    roles = _allocate_v11_roles(
        n_total,
        seed=int(params.get("role_seed", seed + 1000)),
        pool_max=float(params.get("max_synthetic_prop", 0.40)),
        base_max=float(params.get("max_base_prop", 0.04)),
        conformal_max=float(params.get("max_conformal_prop", 0.02)),
        minimum=int(params.get("minimum_role_size", 30)),
    )
    base_idx = _prefix(
        roles["base_train"],
        n_total,
        float(params.get("base_prop", 0.02)),
    )
    pool_idx = _prefix(
        roles["synthetic_pool"],
        n_total,
        float(params.get("synthetic_prop", 0.30)),
    )
    conformal_idx = _prefix(
        roles["conformal"],
        n_total,
        float(params.get("conformal_prop", 0.01)),
    )

    # Preprocessing is fitted exclusively on its dedicated role.
    prep_idx = roles["preprocessing"]
    x_medians = _column_medians(raw_x[prep_idx])
    y_medians = _column_medians(raw_y[prep_idx])
    x_imputed = _impute(raw_x, x_medians)
    y_imputed = _impute(raw_y, y_medians)
    x_scaler = StandardScaler().fit(x_imputed[prep_idx])
    y_scaler = StandardScaler().fit(y_imputed[prep_idx])
    x = x_scaler.transform(x_imputed)
    y = y_scaler.transform(y_imputed)
    outcome_scale = np.asarray(y_scaler.scale_, dtype=float).reshape(-1)
    outcome_scale = np.where(outcome_scale > 1e-12, outcome_scale, 1.0)

    # The conformal reservoir is nested. RCP uses half for threshold learning
    # and half for untouched final calibration; other families use it in full.
    midpoint = max(1, len(conformal_idx) // 2)
    learning_idx = conformal_idx[:midpoint]
    final_cal_idx = conformal_idx[midpoint:]
    if len(final_cal_idx) < 2:
        raise ValueError("V11 final calibration role is too small.")

    teacher_strength = str(params.get("teacher_strength", "moderate"))
    teacher = _fit_teacher(
        x[roles["teacher_train"]],
        y[roles["teacher_train"]],
        strength=teacher_strength,
        seed=int(params.get("teacher_seed", seed + 2000)),
    )
    oracle_access = int(teacher is None)

    def synthetic(indices: np.ndarray) -> np.ndarray:
        return _teacher_predict(
            teacher,
            x[indices],
            oracle_y=y[indices] if teacher is None else None,
        )

    teacher_val_prediction = synthetic(roles["teacher_val"])
    teacher_diagnostics = {
        "strength": teacher_strength,
        "rmse_standardized": float(
            np.sqrt(mean_squared_error(y[roles["teacher_val"]], teacher_val_prediction))
        ),
        "r2": float(
            r2_score(
                y[roles["teacher_val"]],
                teacher_val_prediction,
                multioutput="variance_weighted",
            )
        ),
        "oracle_true_outcome_access": oracle_access,
    }

    hidden = int(params.get("hidden", 128))
    mean_epochs = int(params.get("mean_epochs", 100))
    threshold_epochs = int(params.get("threshold_epochs", 100))
    cqr_epochs = int(params.get("cqr_epochs", 200))
    cv_epochs = int(params.get("cv_epochs", 40))
    learning_rate = float(params.get("learning_rate", 2e-3))
    family_seed = int(params.get("model_seed", seed + 3000))

    mean_fit = fit_mean_network(
        x[base_idx],
        y[base_idx],
        seed=family_seed,
        hidden=hidden,
        epochs=mean_epochs,
        learning_rate=learning_rate,
    )
    mean_all: dict[str, np.ndarray] = {}

    def mean_for(name: str, indices: np.ndarray) -> np.ndarray:
        if name not in mean_all:
            mean_all[name] = mean_fit.predict(x[indices])
        return mean_all[name]

    s_conformal = _score(y[conformal_idx], mean_for("conformal", conformal_idx))
    s_learning = s_conformal[:midpoint]
    s_final = s_conformal[midpoint:]
    s_pool = _score(synthetic(pool_idx), mean_for("pool", pool_idx))
    s_paired = _score(synthetic(learning_idx), mean_for("learning", learning_idx))
    s_eval = _score(y[roles["evaluation"]], mean_for("evaluation", roles["evaluation"]))
    s_slice = _score(
        y[roles["slice_select"]], mean_for("slice_select", roles["slice_select"])
    )
    methods = [
        str(value)
        for value in params.get(
            "methods",
            V11_ABLATION_METHODS
            if task["experiment"] == "v11_ablations"
            else V11_MAIN_METHODS,
        )
    ]
    rows: list[dict[str, Any]] = []
    metric_seed = int(params.get("metric_seed", seed + 4000))

    # Split-family methods.
    if "Split" in methods:
        calibration = exact_order_statistic(s_conformal, tau)
        rows.append(
            _evaluate_global_radius_method(
                method="Split",
                threshold=calibration.correction,
                calibration_k=calibration.order_index,
                calibration_m=calibration.calibration_size,
                s_eval=s_eval,
                x_eval=x[roles["evaluation"]],
                s_slice_select=s_slice,
                x_slice_select=x[roles["slice_select"]],
                x_partition=x[roles["partition"]],
                tau=tau,
                k_values=[10, 30],
                seed=metric_seed,
                outcome_scale=outcome_scale,
                include_ert=True,
                diagnostics={"base_mean": mean_fit.diagnostics},
            )
        )
    if "SPPI" in methods:
        threshold, diagnostics = spi_score_threshold(
            s_conformal,
            s_pool,
            tau=tau,
            beta=float(params.get("sppi_beta", 0.4)),
        )
        rows.append(
            _evaluate_global_radius_method(
                method="SPPI",
                threshold=threshold,
                calibration_k=int(diagnostics["trusted_upper_index"]),
                calibration_m=len(s_conformal),
                s_eval=s_eval,
                x_eval=x[roles["evaluation"]],
                s_slice_select=s_slice,
                x_slice_select=x[roles["slice_select"]],
                x_partition=x[roles["partition"]],
                tau=tau,
                k_values=[10, 30],
                seed=metric_seed,
                outcome_scale=outcome_scale,
                include_ert=True,
                diagnostics=diagnostics,
            )
        )
    if "NNM" in methods:
        s_conformal_synthetic = _score(
            synthetic(conformal_idx), mean_for("conformal", conformal_idx)
        )
        threshold, diagnostics = semicp_nnm_threshold(
            s_conformal,
            s_conformal_synthetic,
            s_conformal,
            s_pool,
            tau=tau,
        )
        rows.append(
            _evaluate_global_radius_method(
                method="NNM",
                threshold=threshold,
                calibration_k=int(diagnostics["calibration_k"]),
                calibration_m=int(
                    diagnostics["trusted_calibration_scores"]
                    + diagnostics["synthetic_scores"]
                ),
                s_eval=s_eval,
                x_eval=x[roles["evaluation"]],
                s_slice_select=s_slice,
                x_slice_select=x[roles["slice_select"]],
                x_partition=x[roles["partition"]],
                tau=tau,
                k_values=[10, 30],
                seed=metric_seed,
                outcome_scale=outcome_scale,
                include_ert=True,
                diagnostics=diagnostics,
            )
        )

    # CQR family. Both methods use the same architecture, initialization,
    # optimizer, epoch budget and trusted base-training role.
    if "CQR" in methods or "CQR-PPI" in methods:
        cqr_seed = family_seed + 101
        if "CQR" in methods:
            fit = fit_cqr_network(
                x[base_idx],
                y[base_idx],
                alpha=alpha,
                seed=cqr_seed,
                objective="supervised",
                hidden=hidden,
                epochs=cqr_epochs,
                learning_rate=learning_rate,
            )
            if fit.diagnostics["finite"]:
                rows.append(
                    _evaluate_cqr(
                        method="CQR",
                        model=_CQRAdapter(fit, y.shape[1]),
                        x_cal=x[conformal_idx],
                        y_cal=y[conformal_idx],
                        x_eval=x[roles["evaluation"]],
                        y_eval=y[roles["evaluation"]],
                        x_slice=x[roles["slice_select"]],
                        y_slice=y[roles["slice_select"]],
                        x_partition=x[roles["partition"]],
                        tau=tau,
                        seed=metric_seed,
                        outcome_scale=outcome_scale,
                    )
                )
            else:
                rows.append(
                    _skipped_row(
                        "CQR", "CQR", "CQR", "nonfinite neural fit", fit.diagnostics
                    )
                )
        if "CQR-PPI" in methods:
            fit = fit_cqr_network(
                x[base_idx],
                y[base_idx],
                alpha=alpha,
                seed=cqr_seed,
                objective="signed",
                power=1.0,
                x_pool=x[pool_idx],
                y_pool=synthetic(pool_idx),
                y_paired=synthetic(base_idx),
                hidden=hidden,
                epochs=cqr_epochs,
                learning_rate=learning_rate,
            )
            if fit.diagnostics["finite"]:
                rows.append(
                    _evaluate_cqr(
                        method="CQR-PPI",
                        model=_CQRAdapter(fit, y.shape[1]),
                        x_cal=x[conformal_idx],
                        y_cal=y[conformal_idx],
                        x_eval=x[roles["evaluation"]],
                        y_eval=y[roles["evaluation"]],
                        x_slice=x[roles["slice_select"]],
                        y_slice=y[roles["slice_select"]],
                        x_partition=x[roles["partition"]],
                        tau=tau,
                        seed=metric_seed,
                        outcome_scale=outcome_scale,
                    )
                )
            else:
                rows.append(
                    _skipped_row(
                        "CQR-PPI",
                        "CQR",
                        "CQR",
                        "nonfinite neural fit",
                        fit.diagnostics,
                    )
                )

    # RCP family.
    threshold_seed = family_seed + 202
    bound = _threshold_bound(s_learning)
    rcp_fits: list[
        tuple[str, V11NeuralFit, float | None, dict[str, float] | None]
    ] = []
    if "RCP" in methods:
        rcp_fits.append(
            (
                "RCP",
                fit_threshold_network(
                    x[learning_idx],
                    s_learning,
                    tau=tau,
                    seed=threshold_seed,
                    objective="supervised",
                    hidden=hidden,
                    epochs=threshold_epochs,
                    learning_rate=learning_rate,
                    output_bound=bound,
                ),
                0.0,
                None,
            )
        )
    if "RCP-PTFT" in methods:
        rcp_fits.append(
            (
                "RCP-PTFT",
                fit_threshold_network(
                    x[learning_idx],
                    s_learning,
                    tau=tau,
                    seed=threshold_seed,
                    objective="pretrain_finetune",
                    x_pool=x[pool_idx],
                    score_pool=s_pool,
                    hidden=hidden,
                    epochs=threshold_epochs,
                    learning_rate=learning_rate,
                    output_bound=bound,
                    pretrain_epochs=threshold_epochs // 2,
                ),
                None,
                None,
            )
        )
    if "RCP-PPI" in methods:
        rcp_fits.append(
            (
                "RCP-PPI",
                fit_threshold_network(
                    x[learning_idx],
                    s_learning,
                    tau=tau,
                    seed=threshold_seed,
                    objective="signed",
                    power=1.0,
                    x_pool=x[pool_idx],
                    score_pool=s_pool,
                    score_paired=s_paired,
                    hidden=hidden,
                    epochs=threshold_epochs,
                    learning_rate=learning_rate,
                    output_bound=bound,
                ),
                1.0,
                None,
            )
        )
    if "RCP-PPI-CV" in methods:
        cv_fit, cv_power, cv_losses = _fit_cv_threshold(
            x_learning=x[learning_idx],
            s_learning=s_learning,
            s_paired=s_paired,
            x_pool=x[pool_idx],
            s_pool=s_pool,
            tau=tau,
            seed=threshold_seed,
            hidden=hidden,
            epochs=threshold_epochs,
            cv_epochs=cv_epochs,
            learning_rate=learning_rate,
            output_bound=bound,
            grid=[
                float(value)
                for value in params.get(
                    "lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0]
                )
            ],
        )
        rcp_fits.append(("RCP-PPI-CV", cv_fit, cv_power, cv_losses))
    if "RCP-Aug" in methods:
        augment_weight = float(params.get("augment_weight", 0.5))
        rcp_fits.append(
            (
                "RCP-Aug",
                fit_threshold_network(
                    x[learning_idx],
                    s_learning,
                    tau=tau,
                    seed=threshold_seed,
                    objective="positive",
                    power=augment_weight,
                    x_pool=x[pool_idx],
                    score_pool=s_pool,
                    hidden=hidden,
                    epochs=threshold_epochs,
                    learning_rate=learning_rate,
                    output_bound=bound,
                ),
                augment_weight,
                None,
            )
        )
    if "RCP-Aug-Permuted" in methods:
        placebo_scores = np.asarray(s_pool)[
            np.random.default_rng(seed + 271828).permutation(len(s_pool))
        ]
        augment_weight = float(params.get("augment_weight", 0.5))
        rcp_fits.append(
            (
                "RCP-Aug-Permuted",
                fit_threshold_network(
                    x[learning_idx],
                    s_learning,
                    tau=tau,
                    seed=threshold_seed,
                    objective="positive",
                    power=augment_weight,
                    x_pool=x[pool_idx],
                    score_pool=placebo_scores,
                    hidden=hidden,
                    epochs=threshold_epochs,
                    learning_rate=learning_rate,
                    output_bound=bound,
                ),
                augment_weight,
                None,
            )
        )
    if "RCP-Pool" in methods:
        rcp_fits.append(
            (
                "RCP-Pool",
                fit_threshold_network(
                    x[learning_idx],
                    s_learning,
                    tau=tau,
                    seed=threshold_seed,
                    objective="pool",
                    x_pool=x[pool_idx],
                    score_pool=s_pool,
                    hidden=hidden,
                    epochs=threshold_epochs,
                    learning_rate=learning_rate,
                    output_bound=bound,
                ),
                None,
                None,
            )
        )
    if "RCP-Extra" in methods:
        # Explicit resource ceiling: the pool's true outcomes are accessed only
        # for this ablation and are fully recorded in the outcome-access ledger.
        s_pool_true = _score(y[pool_idx], mean_for("pool", pool_idx))
        rcp_fits.append(
            (
                "RCP-Extra",
                fit_threshold_network(
                    x[learning_idx],
                    s_learning,
                    tau=tau,
                    seed=threshold_seed,
                    objective="positive",
                    power=float(params.get("augment_weight", 0.5)),
                    x_pool=x[pool_idx],
                    score_pool=s_pool_true,
                    hidden=hidden,
                    epochs=threshold_epochs,
                    learning_rate=learning_rate,
                    output_bound=bound,
                ),
                float(params.get("augment_weight", 0.5)),
                None,
            )
        )
        oracle_access += int(len(pool_idx))

    for method, fit, power, cv_losses in rcp_fits:
        if fit.diagnostics["finite"]:
            rows.append(
                _evaluate_threshold(
                    method=method,
                    fit=fit,
                    x_cal=x[final_cal_idx],
                    s_cal=s_final,
                    x_eval=x[roles["evaluation"]],
                    s_eval=s_eval,
                    x_slice=x[roles["slice_select"]],
                    s_slice=s_slice,
                    x_partition=x[roles["partition"]],
                    tau=tau,
                    seed=metric_seed,
                    outcome_scale=outcome_scale,
                    power=power,
                    cv_losses=cv_losses,
                )
            )
        else:
            rows.append(
                _skipped_row(
                    method, "RCP", "RCP", "nonfinite neural fit", fit.diagnostics
                )
            )

    row_names = [row["method"] for row in rows]
    missing = [method for method in methods if method not in row_names]
    if missing:
        raise RuntimeError(f"V11 task did not produce requested methods: {missing}")

    role_sizes = {
        "preprocessing": int(len(roles["preprocessing"])),
        "teacher_train": int(len(roles["teacher_train"])),
        "teacher_val": int(len(roles["teacher_val"])),
        "synthetic_pool": int(len(pool_idx)),
        "base_train": int(len(base_idx)),
        "conformal_full": int(len(conformal_idx)),
        "rcp_learning": int(len(learning_idx)),
        "rcp_final_calibration": int(len(final_cal_idx)),
        "partition": int(len(roles["partition"])),
        "slice_select": int(len(roles["slice_select"])),
        "evaluation": int(len(roles["evaluation"])),
    }
    result = {
        "dataset": dataset,
        "seed": seed,
        "alpha": alpha,
        "tau": tau,
        "experiment": task["experiment"],
        "condition": {
            "teacher_strength": teacher_strength,
            "synthetic_prop": float(params.get("synthetic_prop", 0.30)),
            "base_prop": float(params.get("base_prop", 0.02)),
            "conformal_prop": float(params.get("conformal_prop", 0.01)),
            "augment_weight": float(params.get("augment_weight", 0.5)),
            "label_budget": label_budget,
        },
        "sample": sample_meta,
        "role_sizes": role_sizes,
        "role_hashes": {
            name: hash_indices(np.asarray(indices, dtype=int))
            for name, indices in roles.items()
        },
        "teacher": teacher_diagnostics,
        "mean_model": mean_fit.diagnostics,
        "methods": rows,
        "outcome_access": {
            "ordinary_synthetic_pool_true_outcomes": 0,
            "oracle_or_resource_control_true_outcomes": int(oracle_access),
            "representation_labels": 0,
        },
        "protocol": {
            "data_cap": "full except SGEMM=50000",
            "preprocessing_fit_role": "preprocessing only",
            "partition_fit_role": "partition only",
            "wsc_selection_role": "slice_select only",
            "calibration": "exact ceil((m+1)tau) order statistic",
            "architecture": f"two-hidden-layer ReLU width {hidden}",
            "metric_k": [10, 30],
        },
    }
    return result, {}
