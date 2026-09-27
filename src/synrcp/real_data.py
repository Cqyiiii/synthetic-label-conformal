from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
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
from .metrics import (
    clear_partition_cache,
    cross_fitted_ert,
    evaluate_slab,
    grouped_coverage_metrics,
    select_worst_slab,
)
from .models import (
    boundary_score,
    cross_fitted_power,
    estimate_conditional_boundary_density,
    estimate_geometry,
    fit_linear_quantile,
    fit_weighted_linear_quantile,
    one_step_update,
    pinball_loss,
    projected_constants,
)
from .neural import (
    fit_neural_threshold,
    train_covariate_representation,
    train_quantile_representation,
)
from .synthetic_methods import (
    calibrate_cqr,
    fit_cqr_ppi_one_step,
    fit_cqr_supervised,
    semicp_nnm_threshold,
    spi_score_threshold,
)

DATASET_NAMES = [
    "bike",
    "diamond",
    "gas_turbine",
    "naval",
    "SGEMM",
    "superconduct",
    "Transcoding",
    "WEC",
]

MAIN_DATASETS = ["bike", "diamond", "gas_turbine", "SGEMM", "superconduct"]


def _read_dataset(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read raw arrays without fitting imputation or scaling on the full data."""

    if name == "diamond":
        frame = pd.read_csv(data_dir / "diamonds.csv")
        frame = pd.get_dummies(
            frame,
            columns=[column for column in ["cut", "color", "clarity"] if column in frame],
            drop_first=True,
        )
        y = frame[["price"]].to_numpy(dtype=float)
        x = frame.drop(columns=["price"]).to_numpy(dtype=float)
    elif name == "bike":
        frame = pd.read_csv(data_dir / "bike_hour.csv")
        y = frame[["cnt"]].to_numpy(dtype=float)
        drop = [column for column in ["dteday", "casual", "registered", "cnt"] if column in frame]
        x = frame.drop(columns=drop).to_numpy(dtype=float)
    elif name == "superconduct":
        frame = pd.read_csv(data_dir / "super.csv")
        y = frame[["critical_temp"]].to_numpy(dtype=float)
        x = frame.drop(columns=["critical_temp"]).to_numpy(dtype=float)
    elif name == "naval":
        frame = pd.read_csv(data_dir / "naval.csv")
        y = frame.iloc[:, -2:].to_numpy(dtype=float)
        x = frame.iloc[:, :-2].to_numpy(dtype=float)
    elif name == "gas_turbine":
        frame = pd.read_csv(data_dir / "gt_full.csv", index_col=0)
        y = frame.iloc[:, -2:].to_numpy(dtype=float)
        x = frame.iloc[:, :-2].to_numpy(dtype=float)
    elif name == "SGEMM":
        frame = pd.read_csv(data_dir / "sgemm_product.csv")
        targets = ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]
        if all(target in frame for target in targets):
            y = frame[targets].to_numpy(dtype=float)
            x = frame.drop(columns=targets).to_numpy(dtype=float)
        else:
            y = frame.iloc[:, -4:].to_numpy(dtype=float)
            x = frame.iloc[:, :-4].to_numpy(dtype=float)
    elif name == "Transcoding":
        frame = pd.read_csv(data_dir / "transcoding_measurement.tsv", sep="\t")
        frame = frame.drop(columns=[column for column in ["id", "url"] if column in frame])
        frame = pd.get_dummies(
            frame,
            columns=[column for column in ["codec", "o_codec"] if column in frame],
            drop_first=True,
        )
        y = frame[["utime", "umem"]].to_numpy(dtype=float)
        x = frame.drop(columns=["utime", "umem"]).to_numpy(dtype=float)
    elif name == "WEC":
        frame = pd.read_csv(data_dir / "WEC_Perth_49.csv")
        position_columns = [
            column
            for column in frame
            if (column.startswith("X") or column.startswith("Y")) and column[1:].isdigit()
        ]
        power_columns = [
            column
            for column in frame
            if (column.startswith("P") or column.startswith("Power")) and column[-1:].isdigit()
        ]
        if position_columns and power_columns:
            x = frame[position_columns].to_numpy(dtype=float)
            y = frame[power_columns].to_numpy(dtype=float)
        else:
            x = frame.iloc[:, :98].to_numpy(dtype=float)
            y = frame.iloc[:, 98:147].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def load_dataset(name: str, data_dir: Path, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = _read_dataset(name, data_dir)
    if max_rows > 0 and x.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))
        x, y = x[indices], y[indices]
    return x, y


def _finite_with_nan(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array, dtype=float).copy()
    output[~np.isfinite(output)] = np.nan
    return output


def _column_medians(reference: np.ndarray) -> np.ndarray:
    reference = _finite_with_nan(reference)
    medians = np.nanmedian(reference, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def _impute(array: np.ndarray, medians: np.ndarray) -> np.ndarray:
    output = _finite_with_nan(array)
    missing = np.isnan(output)
    if missing.any():
        output[missing] = np.take(np.asarray(medians, dtype=float), np.where(missing)[1])
    return output


_ROLE_SPECS = [
    ("preprocessing", "preprocessing_prop", "max_preprocessing_prop", 0.02),
    ("teacher_train", "teacher_prop", "max_teacher_prop", 0.15),
    ("teacher_val", "teacher_val_prop", "max_teacher_val_prop", 0.03),
    ("mean_train", "mean_prop", "max_mean_prop", 0.03),
    ("representation_train", "representation_prop", "max_representation_prop", 0.03),
    ("synthetic_pool", "synthetic_prop", "max_synthetic_prop", 0.20),
    ("learning", "learning_prop", "max_learning_prop", 0.03),
    ("calibration", "calibration_prop", "max_calibration_prop", 0.03),
    ("partition", "partition_prop", "max_partition_prop", 0.03),
    ("slice_select", "slice_select_prop", "max_slice_select_prop", 0.03),
]


def _allocate_role_reservoirs(
    n_total: int,
    params: dict[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Allocate fixed maximum reservoirs and return current nested prefixes."""

    minimum = int(params.get("minimum_role_size", 30))
    current_props: dict[str, float] = {}
    maximum_props: dict[str, float] = {}
    for role, current_key, maximum_key, default in _ROLE_SPECS:
        current = float(params.get(current_key, default))
        maximum = float(params.get(maximum_key, max(current, default)))
        if current < 0 or maximum < current:
            raise ValueError(f"Invalid current/maximum proportion for role {role}.")
        current_props[role] = current
        maximum_props[role] = maximum
    if sum(maximum_props.values()) >= 0.90:
        raise ValueError("Maximum role reservoirs leave insufficient fixed evaluation data.")

    permutation = np.random.default_rng(int(seed)).permutation(int(n_total))
    cursor = 0
    roles: dict[str, np.ndarray] = {}
    reservoirs: dict[str, np.ndarray] = {}
    for role, _, _, _ in _ROLE_SPECS:
        max_size = max(minimum, int(math.floor(maximum_props[role] * n_total)))
        current_size = max(minimum, int(math.floor(current_props[role] * n_total)))
        max_size = min(max_size, n_total - cursor)
        if current_size > max_size:
            raise ValueError(f"Current role {role} exceeds its maximum reservoir.")
        reservoir = permutation[cursor : cursor + max_size]
        reservoirs[role] = reservoir
        roles[role] = reservoir[:current_size]
        cursor += max_size
    roles["evaluation"] = permutation[cursor:]
    reservoirs["evaluation"] = roles["evaluation"]
    if roles["evaluation"].size < minimum:
        raise ValueError("Evaluation role is too small; reduce maximum role proportions.")
    return roles, reservoirs


def _allocate_roles(n_total: int, params: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    roles, _ = _allocate_role_reservoirs(n_total, params, seed)
    return roles


def _teacher_params(strength: str, seed: int, threads: int) -> dict[str, Any]:
    common = {
        "random_state": int(seed),
        "n_jobs": int(threads),
        "bootstrap": True,
        "max_features": 1.0,
    }
    if strength == "weak":
        return {**common, "n_estimators": 50, "max_depth": 5, "min_samples_leaf": 20}
    if strength == "moderate":
        return {**common, "n_estimators": 80, "max_depth": 12, "min_samples_leaf": 5}
    if strength == "strong":
        return {**common, "n_estimators": 120, "max_depth": None, "min_samples_leaf": 1}
    raise ValueError(f"Unknown teacher strength: {strength}")


def _fit_rf(x: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> RandomForestRegressor:
    model = RandomForestRegressor(**params)
    target = y.reshape(-1) if y.shape[1] == 1 else y
    model.fit(x, target)
    return model


def _predict_2d(model: RandomForestRegressor, x: np.ndarray) -> np.ndarray:
    prediction = np.asarray(model.predict(x), dtype=float)
    return prediction.reshape(-1, 1) if prediction.ndim == 1 else prediction


def _score(y: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """L-infinity residual score matching the manuscript set geometry."""

    return np.max(np.abs(np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)), axis=1)


def _held_out_pinball_power(
    *,
    phi: np.ndarray,
    score: np.ndarray,
    score_dagger: np.ndarray,
    phi_pool: np.ndarray,
    score_pool: np.ndarray,
    tau: float,
    lambda_grid: list[float],
    ridge: float,
    quantile_alpha: float,
    density_delta_tau: float,
    seed: int,
) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(phi))
    split = max(phi.shape[1] + 2, int(0.75 * len(indices)))
    split = min(split, len(indices) - 2)
    train, valid = indices[:split], indices[split:]
    if valid.size < 2 or train.size < phi.shape[1] + 2:
        return 0.0, {}
    pilot = fit_linear_quantile(phi[train], score[train], tau, alpha=quantile_alpha)
    q_train = pilot.predict_features(phi[train])
    density = estimate_conditional_boundary_density(
        phi[train],
        score[train],
        tau=tau,
        phi_eval=phi[train],
        delta_tau=density_delta_tau,
        alpha=quantile_alpha,
    )
    hessian, _, _ = estimate_geometry(phi[train], density.density, ridge=ridge)
    psi = boundary_score(phi[train], score[train], q_train, tau)
    psi_dagger = boundary_score(phi[train], score_dagger[train], q_train, tau)
    # Crucially, every candidate uses the held-out pilot to recompute pool scores.
    q_pool = pilot.predict_features(phi_pool)
    psi_pool = boundary_score(phi_pool, score_pool, q_pool, tau)
    losses: dict[str, float] = {}
    for power in lambda_grid:
        beta, _ = one_step_update(
            pilot.coef_, hessian, psi, psi_dagger, psi_pool, float(power), max_update_norm=5.0
        )
        losses[str(power)] = float(np.mean(pinball_loss(score[valid], phi[valid] @ beta, tau)))
    selected = min(lambda_grid, key=lambda value: losses[str(value)])
    return float(selected), losses


def _fit_method_betas(
    *,
    x_representation: np.ndarray,
    s_representation: np.ndarray | None,
    x_learning: np.ndarray,
    s_learning: np.ndarray,
    s_dagger: np.ndarray,
    x_pool: np.ndarray,
    s_pool: np.ndarray,
    tau: float,
    params: dict[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Callable[[np.ndarray], np.ndarray]]:
    representation_mode = str(params.get("representation_mode", "supervised_score_screening"))
    if representation_mode == "covariate_only":
        representation = train_covariate_representation(
            x_representation,
            tau=tau,
            hidden=int(params.get("hidden", 32)),
            feature_dim=int(params.get("feature_dim", 12)),
            seed=seed,
        )
    elif representation_mode == "supervised_score_screening":
        if s_representation is None:
            raise ValueError("Supervised representation screening requires external-role scores.")
        representation = train_quantile_representation(
            x_representation,
            s_representation,
            tau=tau,
            hidden=int(params.get("hidden", 32)),
            feature_dim=int(params.get("feature_dim", 12)),
            epochs=int(params.get("quantile_epochs", 80)),
            learning_rate=float(params.get("quantile_lr", 1e-3)),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown representation_mode: {representation_mode}")

    feature_fn = representation.features
    phi = feature_fn(x_learning)
    phi_pool = feature_fn(x_pool)
    alpha = float(params.get("quantile_alpha", 1e-5))
    ridge = float(params.get("ridge", 1e-3))
    density_delta = float(params.get("density_delta_tau", 0.04))
    pilot = fit_linear_quantile(phi, s_learning, tau, alpha=alpha)
    q = pilot.predict_features(phi)
    density = estimate_conditional_boundary_density(
        phi,
        s_learning,
        tau=tau,
        phi_eval=phi,
        delta_tau=density_delta,
        alpha=alpha,
    )
    hessian, g_c, _ = estimate_geometry(phi, density.density, ridge=ridge)
    psi = boundary_score(phi, s_learning, q, tau)
    psi_dagger = boundary_score(phi, s_dagger, q, tau)
    q_pool = pilot.predict_features(phi_pool)
    psi_pool = boundary_score(phi_pool, s_pool, q_pool, tau)
    constants = projected_constants(psi, psi_dagger, g_c, psi_pool)
    kappa = len(x_learning) / max(len(x_pool), 1)
    in_sample_power = constants["b"] / max(
        constants["c"] + kappa * constants["c_pool"], 1e-12
    )
    in_sample_power = float(
        np.clip(in_sample_power, 0.0, float(params.get("lambda_max", 1.5)))
    )
    plugin = cross_fitted_power(
        x_learning,
        s_learning,
        s_dagger,
        x_pool,
        s_pool,
        feature_fn=feature_fn,
        tau=tau,
        n_splits=int(params.get("power_folds", 2)),
        ridge=ridge,
        gate_rate=float(params.get("gate_rate", 1.0)),
        lambda_max=float(params.get("lambda_max", 1.5)),
        allow_signed=False,
        seed=seed + 13,
        density_delta_tau=density_delta,
        quantile_alpha=alpha,
    )

    lambda_grid = [
        float(value)
        for value in params.get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ]
    cv_power, cv_losses = _held_out_pinball_power(
        phi=phi,
        score=s_learning,
        score_dagger=s_dagger,
        phi_pool=phi_pool,
        score_pool=s_pool,
        tau=tau,
        lambda_grid=lambda_grid,
        ridge=ridge,
        quantile_alpha=alpha,
        density_delta_tau=density_delta,
        seed=seed + 99,
    )

    powers = {
        "RCP-supervised": 0.0,
        "RCP-PPI-unit-one-step": 1.0,
        "RCP-PPI-deterministic-gate": plugin.selected_power,
        "RCP-PPI-pinball-CV": cv_power,
    }
    if bool(params.get("include_in_sample_projected", False)):
        powers["RCP-PPI-in-sample-projected"] = in_sample_power
    if bool(params.get("include_lambda_grid", False)):
        powers.update({f"RCP-PPI-lambda-{value:g}": value for value in lambda_grid})

    betas: dict[str, np.ndarray] = {}
    update_meta: dict[str, Any] = {}
    if bool(params.get("include_rcp_split_baseline", True)):
        betas["Split"] = np.zeros(phi.shape[1], dtype=float)
        update_meta["Split"] = {
            "lambda": None,
            "update_norm": 0.0,
            "update_clipped": 0.0,
            "hessian_condition": float("nan"),
        }
    for method, power in powers.items():
        beta, metadata = one_step_update(
            pilot.coef_,
            hessian,
            psi,
            psi_dagger,
            psi_pool,
            power,
            max_update_norm=float(params.get("max_update_norm", 5.0)),
        )
        betas[method] = beta
        update_meta[method] = {"lambda": float(power), **metadata}

    if bool(params.get("include_weighted_synthetic", False)):
        synthetic_total_weight = float(params.get("synthetic_erm_weight", 0.5))
        weighted = fit_weighted_linear_quantile(
            phi,
            s_learning,
            phi_pool,
            s_pool,
            tau,
            synthetic_total_weight=synthetic_total_weight,
            alpha=alpha,
        )
        betas["RCP-synthetic-weighted-ERM"] = weighted.coef_
        update_meta["RCP-synthetic-weighted-ERM"] = {
            "lambda": None,
            "update_norm": 0.0,
            "update_clipped": 0.0,
            "hessian_condition": float("nan"),
            "synthetic_total_weight": synthetic_total_weight,
        }

        # Marginal-preserving empirical placebo.  Permuting synthetic scores
        # across pool covariates preserves the score distribution, pool size,
        # solver, and total synthetic weight while destroying input-conditional
        # information.  It therefore tests whether any gain comes from the
        # synthetic labels' alignment with covariate-dependent uncertainty rather
        # than from generic regularization or an altered marginal score law.
        if bool(params.get("include_weighted_synthetic_placebo", False)):
            placebo_rng = np.random.default_rng(seed + 271828)
            placebo_scores = np.asarray(s_pool)[placebo_rng.permutation(len(s_pool))]
            placebo = fit_weighted_linear_quantile(
                phi,
                s_learning,
                phi_pool,
                placebo_scores,
                tau,
                synthetic_total_weight=synthetic_total_weight,
                alpha=alpha,
            )
            placebo_name = "RCP-synthetic-weighted-permuted-placebo"
            betas[placebo_name] = placebo.coef_
            update_meta[placebo_name] = {
                "lambda": None,
                "update_norm": 0.0,
                "update_clipped": 0.0,
                "hessian_condition": float("nan"),
                "synthetic_total_weight": synthetic_total_weight,
                "placebo_type": "synthetic scores permuted across pool covariates",
                "placebo_seed": int(seed + 271828),
            }

    bridge_weights = [
        float(value) for value in params.get("bridge_linear_weights", [])
    ]
    if bridge_weights:
        placebo_rng = np.random.default_rng(seed + 271828)
        placebo_scores = np.asarray(s_pool)[placebo_rng.permutation(len(s_pool))]
        for weight in bridge_weights:
            genuine = fit_weighted_linear_quantile(
                phi,
                s_learning,
                phi_pool,
                s_pool,
                tau,
                synthetic_total_weight=weight,
                alpha=alpha,
            )
            genuine_name = f"Linear-RCP-weighted-genuine-w{weight:g}"
            betas[genuine_name] = genuine.coef_
            update_meta[genuine_name] = {
                "lambda": None,
                "synthetic_total_weight": weight,
                "bridge_estimator": "linear positive-weighted",
            }
            placebo = fit_weighted_linear_quantile(
                phi,
                s_learning,
                phi_pool,
                placebo_scores,
                tau,
                synthetic_total_weight=weight,
                alpha=alpha,
            )
            placebo_name = f"Linear-RCP-weighted-placebo-w{weight:g}"
            betas[placebo_name] = placebo.coef_
            update_meta[placebo_name] = {
                "lambda": None,
                "synthetic_total_weight": weight,
                "bridge_estimator": "linear positive-weighted placebo",
                "placebo_seed": int(seed + 271828),
            }

    shrinkage_grid = [
        float(value) for value in params.get("shape_shrinkage_grid", [])
    ]
    if shrinkage_grid:
        for eta in shrinkage_grid:
            beta_eta = np.asarray(pilot.coef_, dtype=float).copy()
            if beta_eta.size > 1:
                beta_eta[1:] *= eta
            name = f"RCP-supervised-shape-eta-{eta:g}"
            betas[name] = beta_eta
            update_meta[name] = {
                "lambda": None,
                "shape_eta": eta,
                "bridge_estimator": "supervised-only threshold-shape shrinkage",
            }

        folds = KFold(n_splits=2, shuffle=True, random_state=seed + 404)
        eta_losses = {eta: [] for eta in shrinkage_grid}
        for train, valid in folds.split(phi):
            fold_model = fit_linear_quantile(
                phi[train], s_learning[train], tau, alpha=alpha
            )
            for eta in shrinkage_grid:
                fold_beta = np.asarray(fold_model.coef_, dtype=float).copy()
                if fold_beta.size > 1:
                    fold_beta[1:] *= eta
                eta_losses[eta].append(
                    float(
                        np.mean(
                            pinball_loss(
                                s_learning[valid], phi[valid] @ fold_beta, tau
                            )
                        )
                    )
                )
        selected_eta = min(
            shrinkage_grid, key=lambda eta: float(np.mean(eta_losses[eta]))
        )
        selected_beta = np.asarray(pilot.coef_, dtype=float).copy()
        if selected_beta.size > 1:
            selected_beta[1:] *= selected_eta
        selected_name = "RCP-supervised-shape-CV"
        betas[selected_name] = selected_beta
        update_meta[selected_name] = {
            "lambda": None,
            "shape_eta": selected_eta,
            "shape_cv_losses": {
                str(eta): float(np.mean(losses))
                for eta, losses in eta_losses.items()
            },
            "shape_cv_folds": 2,
            "bridge_estimator": "supervised-only CV threshold-shape shrinkage",
        }

    if bool(params.get("include_synthetic_pretrain", False)):
        pretrain = fit_linear_quantile(phi_pool, s_pool, tau, alpha=alpha)
        q_pretrain = pretrain.predict_features(phi)
        pretrain_psi = boundary_score(phi, s_learning, q_pretrain, tau)
        pretrain_beta, pretrain_meta = one_step_update(
            pretrain.coef_,
            hessian,
            pretrain_psi,
            np.zeros_like(pretrain_psi),
            np.zeros_like(psi_pool),
            power=0.0,
            max_update_norm=float(params.get("max_update_norm", 5.0)),
        )
        betas["RCP-synthetic-pretrain-one-step"] = pretrain_beta
        update_meta["RCP-synthetic-pretrain-one-step"] = {
            "lambda": None,
            **pretrain_meta,
        }

    if bool(params.get("include_naive_pooling", True)):
        pooled_model = fit_linear_quantile(
            np.vstack([phi, phi_pool]),
            np.concatenate([s_learning, s_pool]),
            tau,
            alpha=alpha,
        )
        betas["Naive-synthetic-pooling-diagnostic"] = pooled_model.coef_
        update_meta["Naive-synthetic-pooling-diagnostic"] = {
            "lambda": None,
            "update_norm": 0.0,
            "update_clipped": 0.0,
            "hessian_condition": float("nan"),
        }

    diagnostics = {
        "representation": representation.training_summary,
        "representation_mode": representation_mode,
        "sample_constants": constants,
        "plugin": asdict(plugin),
        "in_sample_projected_power": in_sample_power,
        "cv_power": cv_power,
        "cv_losses": cv_losses,
        "updates": update_meta,
        "feature_dimension": int(phi.shape[1]),
        "density": {
            "estimator_label": "nearby-quantile projected-density surrogate",
            "lower_level": density.lower_level,
            "upper_level": density.upper_level,
            "floor_frequency": density.floor_frequency,
            "crossing_frequency": density.crossing_frequency,
            "standard_deviation": float(np.std(density.density)),
        },
    }
    return betas, diagnostics, feature_fn



def _method_descriptor(method: str) -> dict[str, str]:
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
    if method.startswith("Linear-RCP-weighted-genuine"):
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "positive weighted threshold learning", "legacy_alias": "linear bridge"}
    if method.startswith("Linear-RCP-weighted-placebo"):
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "permuted synthetic-label placebo", "legacy_alias": "linear placebo bridge"}
    if method == "Neural-RCP-supervised":
        return {"method_family": "RCP", "supervised_counterpart": "Neural-RCP-supervised", "synthetic_use": "none", "legacy_alias": "audited legacy-capacity RCP"}
    if method.startswith("Neural-RCP-weighted-genuine"):
        return {"method_family": "RCP", "supervised_counterpart": "Neural-RCP-supervised", "synthetic_use": "positive weighted threshold learning", "legacy_alias": "audited neural positive augmentation"}
    if method.startswith("Neural-RCP-weighted-placebo"):
        return {"method_family": "RCP", "supervised_counterpart": "Neural-RCP-supervised", "synthetic_use": "permuted synthetic-label placebo", "legacy_alias": "audited neural placebo"}
    if method.startswith("Neural-RCP-signed"):
        return {"method_family": "RCP", "supervised_counterpart": "Neural-RCP-supervised", "synthetic_use": "legacy signed PPI objective diagnostic", "legacy_alias": "audited legacy signed objective"}
    if method.startswith("RCP-supervised-shape-"):
        return {"method_family": "RCP", "supervised_counterpart": "RCP-supervised", "synthetic_use": "none; supervised shape shrinkage", "legacy_alias": "supervised shrinkage control"}
    return {"method_family": "other", "supervised_counterpart": "", "synthetic_use": "unspecified", "legacy_alias": ""}



def _coverage_metric_row(
    *,
    method: str,
    covered_eval: np.ndarray,
    covered_slice: np.ndarray,
    x_eval: np.ndarray,
    x_partition: np.ndarray,
    x_slice_select: np.ndarray,
    tau: float,
    k_values: list[int],
    seed: int,
    size_summary: dict[str, Any],
    calibration_correction: float,
    calibration_k: int,
    calibration_m: int,
    calibration_infinite: bool,
    include_ert: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    covered_eval = np.asarray(covered_eval, dtype=bool)
    covered_slice = np.asarray(covered_slice, dtype=bool)
    grouped = [
        asdict(
            grouped_coverage_metrics(
                x_partition=x_partition,
                x_eval=x_eval,
                covered_eval=covered_eval,
                tau=tau,
                k=k,
                seed=seed + k,
            )
        )
        for k in k_values
    ]
    slab = select_worst_slab(
        x_slice_select,
        covered_slice,
        min_mass=0.1,
        directions=64,
        quantile_grid=7,
        seed=seed + 303,
    )
    wsc = evaluate_slab(x_eval, covered_eval, slab)
    optional_ert = (
        cross_fitted_ert(x_eval, covered_eval, tau=tau, folds=5, seed=seed + 505)
        if include_ert
        else {}
    )
    row = {
        "method": method,
        **_method_descriptor(method),
        "skipped": False,
        "skip_reason": "",
        "coverage": float(np.mean(covered_eval)),
        "coverage_error": float(abs(np.mean(covered_eval) - tau)),
        "coverage_se": float(np.std(covered_eval, ddof=1) / np.sqrt(max(len(covered_eval), 1))),
        "calibration_correction": float(calibration_correction),
        "calibration_k": int(calibration_k),
        "calibration_m": int(calibration_m),
        "calibration_infinite": bool(calibration_infinite),
        "grouped": grouped,
        **wsc,
        **size_summary,
        **optional_ert,
    }
    if extra:
        row.update(extra)
    return row


def _skipped_method_row(method: str, reason: str) -> dict[str, Any]:
    return {
        "method": method,
        **_method_descriptor(method),
        "skipped": True,
        "skip_reason": reason,
        "lambda": None,
        "grouped": [],
    }

def _evaluate_threshold_predictor(
    *,
    method: str,
    threshold_predict: Callable[[np.ndarray], np.ndarray],
    s_cal: np.ndarray,
    x_cal: np.ndarray,
    s_eval: np.ndarray,
    x_eval: np.ndarray,
    x_partition: np.ndarray,
    x_slice_select: np.ndarray,
    s_slice_select: np.ndarray,
    tau: float,
    k_values: list[int],
    seed: int,
    outcome_scale: np.ndarray,
    include_ert: bool,
    include_metric_bridge: bool = False,
) -> dict[str, Any]:
    q_cal = np.asarray(threshold_predict(x_cal), dtype=float).reshape(-1)
    calibration = calibrate_scores(s_cal, q_cal, tau)
    q_eval = np.asarray(threshold_predict(x_eval), dtype=float).reshape(-1)
    q_partition = np.asarray(threshold_predict(x_partition), dtype=float).reshape(-1)
    q_slice = np.asarray(threshold_predict(x_slice_select), dtype=float).reshape(-1)
    covered_eval = covered(s_eval, q_eval, calibration.correction)
    covered_slice = covered(s_slice_select, q_slice, calibration.correction)
    grouped = [
        asdict(
            grouped_coverage_metrics(
                x_partition=x_partition,
                x_eval=x_eval,
                covered_eval=covered_eval,
                tau=tau,
                k=k,
                seed=seed + k,
            )
        )
        for k in k_values
    ]
    slab = select_worst_slab(
        x_slice_select,
        covered_slice,
        min_mass=0.1,
        directions=64,
        quantile_grid=7,
        seed=seed + 303,
    )
    wsc = evaluate_slab(x_eval, covered_eval, slab)
    set_size = box_volume_summary(q_eval, calibration.correction, outcome_scale)
    optional_ert = (
        cross_fitted_ert(x_eval, covered_eval, tau=tau, folds=5, seed=seed + 505)
        if include_ert
        else {}
    )
    metric_bridge: dict[str, float | int] = {}
    if include_metric_bridge:
        independent = next((item for item in grouped if int(item["k"]) == 10), grouped[0])
        legacy = asdict(
            grouped_coverage_metrics(
                x_partition=x_eval,
                x_eval=x_eval,
                covered_eval=covered_eval,
                tau=tau,
                k=30,
                seed=seed + 30030,
            )
        )
        metric_bridge = {
            "legacy_group_raw_k30": float(legacy["raw"]),
            "legacy_group_k30": int(legacy["k"]),
            "independent_group_raw_k10": float(independent["raw"]),
            "independent_group_debiased_k10": float(independent["debiased"]),
            "independent_group_cross_k10": float(independent["cross_product"]),
        }
    return {
        "method": method,
        **_method_descriptor(method),
        "skipped": False,
        "skip_reason": "",
        "lambda": None,
        "coverage": float(np.mean(covered_eval)),
        "coverage_error": float(abs(np.mean(covered_eval) - tau)),
        "coverage_se": float(
            np.std(covered_eval, ddof=1) / np.sqrt(max(len(covered_eval), 1))
        ),
        "calibration_correction": float(calibration.correction),
        "calibration_k": int(calibration.order_index),
        "calibration_m": int(calibration.calibration_size),
        "calibration_infinite": bool(calibration.infinite),
        "grouped": grouped,
        **wsc,
        **set_size,
        **optional_ert,
        **metric_bridge,
        # These private arrays are removed before serialization.  They are used
        # to construct a common, outcome-free difficulty profile based on the
        # supervised RCP calibrated radius.
        "_profile_radius_partition": np.asarray(q_partition + calibration.correction, dtype=float),
        "_profile_radius_eval": np.asarray(q_eval + calibration.correction, dtype=float),
        "_profile_covered_eval": np.asarray(covered_eval, dtype=bool),
    }


def _evaluate_method(
    *,
    method: str,
    beta: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    s_cal: np.ndarray,
    x_cal: np.ndarray,
    s_eval: np.ndarray,
    x_eval: np.ndarray,
    x_partition: np.ndarray,
    x_slice_select: np.ndarray,
    s_slice_select: np.ndarray,
    tau: float,
    k_values: list[int],
    seed: int,
    outcome_scale: np.ndarray,
    include_ert: bool,
    include_metric_bridge: bool = False,
) -> dict[str, Any]:
    return _evaluate_threshold_predictor(
        method=method,
        threshold_predict=lambda array: feature_fn(array) @ beta,
        s_cal=s_cal,
        x_cal=x_cal,
        s_eval=s_eval,
        x_eval=x_eval,
        x_partition=x_partition,
        x_slice_select=x_slice_select,
        s_slice_select=s_slice_select,
        tau=tau,
        k_values=k_values,
        seed=seed,
        outcome_scale=outcome_scale,
        include_ert=include_ert,
        include_metric_bridge=include_metric_bridge,
    )


def _attach_reference_difficulty_profiles(
    method_rows: list[dict[str, Any]],
    tau: float,
    bins: int = 10,
) -> None:
    """Attach comparable coverage profiles and remove temporary prediction arrays.

    Bins are fitted using only covariates through the supervised RCP calibrated
    radius on the independent partition/evaluation roles.  No evaluation
    outcome is used to define the ordering.  Every RCP-family method is then
    evaluated on the same bins, which makes the profile a direct visual check
    of whether synthetic labels reduce under-coverage in difficult regions.
    """

    reference = next(
        (row for row in method_rows if row.get("method") == "RCP-supervised"),
        None,
    )
    if reference is None or "_profile_radius_partition" not in reference:
        for row in method_rows:
            for key in [
                "_profile_radius_partition",
                "_profile_radius_eval",
                "_profile_covered_eval",
            ]:
                row.pop(key, None)
        return

    reference_partition = np.asarray(reference["_profile_radius_partition"], dtype=float)
    reference_eval = np.asarray(reference["_profile_radius_eval"], dtype=float)
    quantile_grid = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.unique(np.quantile(reference_partition, quantile_grid))
    if edges.size < 2:
        edges = np.asarray([-np.inf, np.inf], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    group_ids = np.searchsorted(edges[1:-1], reference_eval, side="right")
    n_eval = max(int(reference_eval.size), 1)

    for row in method_rows:
        if "_profile_covered_eval" not in row:
            continue
        covered_arr = np.asarray(row["_profile_covered_eval"], dtype=float)
        method_radius = np.asarray(row["_profile_radius_eval"], dtype=float)
        profile: list[dict[str, float | int]] = []
        for group_id in range(max(1, edges.size - 1)):
            mask = group_ids == group_id
            if not np.any(mask):
                continue
            coverage = float(np.mean(covered_arr[mask]))
            profile.append(
                {
                    "difficulty_bin": int(group_id + 1),
                    "n": int(np.sum(mask)),
                    "mass": float(np.sum(mask) / n_eval),
                    "coverage": coverage,
                    "coverage_error": float(abs(coverage - tau)),
                    "signed_coverage_error": float(coverage - tau),
                    "mean_reference_radius": float(np.mean(reference_eval[mask])),
                    "mean_method_radius": float(np.mean(method_radius[mask])),
                }
            )
        row["difficulty_profile"] = profile
        row["difficulty_profile_reference"] = (
            "deciles of the supervised RCP calibrated radius fitted on the independent partition role"
        )

    for row in method_rows:
        for key in [
            "_profile_radius_partition",
            "_profile_radius_eval",
            "_profile_covered_eval",
        ]:
            row.pop(key, None)


def _evaluate_global_radius_method(
    *,
    method: str,
    threshold: float,
    calibration_k: int,
    calibration_m: int,
    s_eval: np.ndarray,
    x_eval: np.ndarray,
    s_slice_select: np.ndarray,
    x_slice_select: np.ndarray,
    x_partition: np.ndarray,
    tau: float,
    k_values: list[int],
    seed: int,
    outcome_scale: np.ndarray,
    include_ert: bool,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    q_eval = np.zeros(len(s_eval), dtype=float)
    q_slice = np.zeros(len(s_slice_select), dtype=float)
    is_infinite = bool(np.isposinf(threshold))
    covered_eval = covered(s_eval, q_eval, threshold)
    covered_slice = covered(s_slice_select, q_slice, threshold)
    size = box_volume_summary(q_eval, threshold, outcome_scale)
    return _coverage_metric_row(
        method=method,
        covered_eval=covered_eval,
        covered_slice=covered_slice,
        x_eval=x_eval,
        x_partition=x_partition,
        x_slice_select=x_slice_select,
        tau=tau,
        k_values=k_values,
        seed=seed,
        size_summary=size,
        calibration_correction=threshold,
        calibration_k=calibration_k,
        calibration_m=calibration_m,
        calibration_infinite=is_infinite,
        include_ert=include_ert,
        extra={"lambda": None, "method_diagnostics": diagnostics or {}},
    )


def _evaluate_cqr_method(
    *,
    method: str,
    cqr_model: Any,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    y_cal: np.ndarray,
    x_cal: np.ndarray,
    y_eval: np.ndarray,
    x_eval: np.ndarray,
    y_slice_select: np.ndarray,
    x_slice_select: np.ndarray,
    x_partition: np.ndarray,
    tau: float,
    k_values: list[int],
    seed: int,
    outcome_scale: np.ndarray,
    include_ert: bool,
) -> dict[str, Any]:
    lower_cal, upper_cal = cqr_model.predict_features(feature_fn(x_cal))
    calibration = calibrate_cqr(y_cal, lower_cal, upper_cal, tau)
    lower_eval, upper_eval = cqr_model.predict_features(feature_fn(x_eval))
    lower_slice, upper_slice = cqr_model.predict_features(feature_fn(x_slice_select))
    covered_eval = covered_box(y_eval, lower_eval, upper_eval, calibration.correction)
    covered_slice = covered_box(y_slice_select, lower_slice, upper_slice, calibration.correction)
    size = interval_box_volume_summary(lower_eval, upper_eval, calibration.correction, outcome_scale)
    metadata = dict(cqr_model.metadata)
    return _coverage_metric_row(
        method=method,
        covered_eval=covered_eval,
        covered_slice=covered_slice,
        x_eval=x_eval,
        x_partition=x_partition,
        x_slice_select=x_slice_select,
        tau=tau,
        k_values=k_values,
        seed=seed,
        size_summary=size,
        calibration_correction=calibration.correction,
        calibration_k=calibration.order_index,
        calibration_m=calibration.calibration_size,
        calibration_infinite=calibration.infinite,
        include_ert=include_ert,
        extra={
            "lambda": 1.0 if method.startswith("CQR-PPI") else 0.0,
            "update_norm": float(metadata.get("maximum_update_norm", 0.0)),
            "update_clipped": float(metadata.get("update_clipped_count", 0) > 0),
            "hessian_condition": float(metadata.get("maximum_hessian_condition", float("nan"))),
            "method_diagnostics": metadata,
        },
    )


def _budget_matched_supervised(
    *,
    x: np.ndarray,
    y: np.ndarray,
    extra_indices: np.ndarray,
    x_cal: np.ndarray,
    x_eval: np.ndarray,
    x_slice: np.ndarray,
    tau: float,
    params: dict[str, Any],
    threads: int,
    seed: int,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build residual scores from genuinely out-of-fold mean predictions."""

    x_extra = x[extra_indices]
    y_extra = y[extra_indices]
    folds = KFold(n_splits=2, shuffle=True, random_state=seed)
    out_of_fold_mean = np.empty_like(y_extra)
    base_params = {
        "n_estimators": int(params.get("budget_mean_trees", 50)),
        "max_depth": int(params.get("mean_depth", 12)),
        "min_samples_leaf": int(params.get("mean_leaf", 3)),
        "max_features": 1.0,
        "bootstrap": True,
        "n_jobs": threads,
    }
    for fold, (train, valid) in enumerate(folds.split(x_extra)):
        model = _fit_rf(
            x_extra[train],
            y_extra[train],
            {**base_params, "random_state": seed + fold},
        )
        out_of_fold_mean[valid] = _predict_2d(model, x_extra[valid])
    score_extra = _score(y_extra, out_of_fold_mean)
    representation = train_quantile_representation(
        x_extra,
        score_extra,
        tau=tau,
        hidden=int(params.get("hidden", 32)),
        feature_dim=int(params.get("feature_dim", 12)),
        epochs=int(params.get("budget_quantile_epochs", max(30, int(params.get("quantile_epochs", 80)) // 2))),
        seed=seed + 20,
    )
    model = fit_linear_quantile(
        representation.features(x_extra),
        score_extra,
        tau,
        alpha=float(params.get("quantile_alpha", 1e-5)),
    )
    final_mean = _fit_rf(
        x_extra,
        y_extra,
        {**base_params, "random_state": seed + 40},
    )
    score_cal = _score(y[params["_calibration_indices"]], _predict_2d(final_mean, x_cal))
    score_eval = _score(y[params["_evaluation_indices"]], _predict_2d(final_mean, x_eval))
    score_slice = _score(y[params["_slice_indices"]], _predict_2d(final_mean, x_slice))
    return (
        model.coef_,
        representation.features,
        score_cal,
        score_eval,
        score_slice,
        {
            "out_of_fold_mean": True,
            "labeled_rows": int(len(extra_indices)),
            "representation": representation.training_summary,
        },
    )


def _apply_label_budget_regime(params: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(params)
    regime = str(resolved.get("label_budget", "custom"))
    if regime == "ultra_low":
        resolved.update(
            {
                "mean_prop": 0.01,
                "representation_prop": 0.01,
                "learning_prop": 0.0025,
                "calibration_prop": 0.0025,
            }
        )
    elif regime == "low":
        resolved.update(
            {
                "mean_prop": 0.02,
                "representation_prop": 0.01,
                "learning_prop": 0.005,
                "calibration_prop": 0.005,
            }
        )
    elif regime == "moderate":
        resolved.update(
            {
                "mean_prop": 0.03,
                "representation_prop": 0.01,
                "learning_prop": 0.02,
                "calibration_prop": 0.02,
            }
        )
    elif regime not in {"custom", "none", ""}:
        raise ValueError(f"Unknown label_budget regime: {regime}")
    return resolved


def run_real_data_task(
    task: dict[str, Any], work_dir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    clear_partition_cache()
    params = _apply_label_budget_regime(dict(task["params"]))
    dataset = str(params.get("dataset", "bike"))
    seed = int(params.get("seed", 0))
    threads = int(task.get("threads", 1))
    tau = float(params.get("tau", 0.9))
    data_dir = Path(params.get("data_dir", "legacy/Datasets"))
    if not data_dir.is_absolute():
        data_dir = (work_dir / data_dir).resolve()
    dataset_seed = int(params.get("dataset_sample_seed", 710)) + sum(dataset.encode("utf-8"))
    x_raw, y_raw = load_dataset(
        dataset,
        data_dir=data_dir,
        max_rows=int(params.get("max_rows", 12000)),
        seed=dataset_seed,
    )
    role_seed = int(params.get("role_seed", seed + 1))
    roles, reservoirs = _allocate_role_reservoirs(len(x_raw), params, seed=role_seed)

    preprocessing = roles["preprocessing"]
    x_median = _column_medians(x_raw[preprocessing])
    y_median = _column_medians(y_raw[preprocessing])
    x_imputed = _impute(x_raw, x_median)
    y_imputed = _impute(y_raw, y_median)
    x_scaler = StandardScaler().fit(x_imputed[preprocessing])
    y_scaler = StandardScaler().fit(y_imputed[preprocessing])
    x = x_scaler.transform(x_imputed)
    y = y_scaler.transform(y_imputed)

    oracle_synthetic = bool(params.get("oracle_synthetic_labels", False))
    teacher_strength = "oracle" if oracle_synthetic else str(params.get("teacher_strength", "moderate"))
    if oracle_synthetic:
        teacher = None
        teacher_parameters: dict[str, Any] = {"teacher_type": "oracle_target_outcomes"}
        teacher_val_mse = 0.0
    else:
        teacher_seed = int(params.get("teacher_fit_seed", seed + 2))
        teacher_parameters = _teacher_params(teacher_strength, teacher_seed, threads)
        teacher = _fit_rf(x[roles["teacher_train"]], y[roles["teacher_train"]], teacher_parameters)
        teacher_val_prediction = _predict_2d(teacher, x[roles["teacher_val"]])
        teacher_val_mse = float(np.mean((teacher_val_prediction - y[roles["teacher_val"]]) ** 2))

    mean_parameters = {
        "n_estimators": int(params.get("mean_trees", 80)),
        "max_depth": int(params.get("mean_depth", 12)),
        "min_samples_leaf": int(params.get("mean_leaf", 3)),
        "max_features": 1.0,
        "bootstrap": True,
        "random_state": int(params.get("mean_fit_seed", seed + 3)),
        "n_jobs": threads,
    }
    mean_model = _fit_rf(x[roles["mean_train"]], y[roles["mean_train"]], mean_parameters)

    def mean_predict(indices: np.ndarray) -> np.ndarray:
        return _predict_2d(mean_model, x[indices])

    representation_mode = str(params.get("representation_mode", "supervised_score_screening"))
    true_score_roles: dict[str, np.ndarray] = {}
    score_roles_to_read = ["learning", "calibration", "slice_select", "evaluation"]
    if representation_mode == "supervised_score_screening":
        score_roles_to_read.insert(0, "representation_train")
    for role in score_roles_to_read:
        true_score_roles[role] = _score(y[roles[role]], mean_predict(roles[role]))

    if oracle_synthetic:
        ydag_learning = np.asarray(y[roles["learning"]], dtype=float)
        ydag_pool = np.asarray(y[roles["synthetic_pool"]], dtype=float)
    else:
        assert teacher is not None
        ydag_learning = _predict_2d(teacher, x[roles["learning"]])
        ydag_pool = _predict_2d(teacher, x[roles["synthetic_pool"]])
    sdag_learning = _score(ydag_learning, mean_predict(roles["learning"]))
    sdag_pool = _score(ydag_pool, mean_predict(roles["synthetic_pool"]))

    betas, diagnostics, feature_fn = _fit_method_betas(
        x_representation=x[roles["representation_train"]],
        s_representation=true_score_roles.get("representation_train"),
        x_learning=x[roles["learning"]],
        s_learning=true_score_roles["learning"],
        s_dagger=sdag_learning,
        x_pool=x[roles["synthetic_pool"]],
        s_pool=sdag_pool,
        tau=tau,
        params=params,
        seed=int(params.get("representation_fit_seed", seed + 4)),
    )

    k_values = [int(value) for value in params.get("partition_k", [10])]
    include_ert = bool(params.get("include_ert", False))
    include_metric_bridge = bool(params.get("include_metric_bridge", False))
    outcome_scale = np.asarray(y_scaler.scale_, dtype=float)
    # All methods in a paired task use the same independently fitted groups,
    # split-half randomization, and candidate slab directions.
    metric_seed = int(params.get("metric_seed", role_seed + 9000))
    method_rows: list[dict[str, Any]] = []

    if bool(params.get("include_split_family", False)):
        split_scores = np.concatenate(
            [true_score_roles["learning"], true_score_roles["calibration"]]
        )
        split_calibration = exact_order_statistic(split_scores, tau)
        method_rows.append(
            _evaluate_global_radius_method(
                method="Split",
                threshold=split_calibration.correction,
                calibration_k=split_calibration.order_index,
                calibration_m=split_calibration.calibration_size,
                s_eval=true_score_roles["evaluation"],
                x_eval=x[roles["evaluation"]],
                s_slice_select=true_score_roles["slice_select"],
                x_slice_select=x[roles["slice_select"]],
                x_partition=x[roles["partition"]],
                tau=tau,
                k_values=k_values,
                seed=metric_seed,
                outcome_scale=outcome_scale,
                include_ert=include_ert,
                diagnostics={"trusted_calibration_roles": ["learning", "calibration"]},
            )
        )
        if bool(params.get("include_spi", True)):
            try:
                threshold, spi_diag = spi_score_threshold(
                    split_scores,
                    sdag_pool,
                    tau=tau,
                    beta=float(params.get("spi_beta", 0.4)),
                )
                method_rows.append(
                    _evaluate_global_radius_method(
                        method="SPI-score-transporter",
                        threshold=threshold,
                        calibration_k=int(spi_diag.get("trusted_upper_index", 0)),
                        calibration_m=int(len(split_scores)),
                        s_eval=true_score_roles["evaluation"],
                        x_eval=x[roles["evaluation"]],
                        s_slice_select=true_score_roles["slice_select"],
                        x_slice_select=x[roles["slice_select"]],
                        x_partition=x[roles["partition"]],
                        tau=tau,
                        k_values=k_values,
                        seed=metric_seed,
                        outcome_scale=outcome_scale,
                        include_ert=include_ert,
                        diagnostics=spi_diag,
                    )
                )
            except Exception as error:  # a comparator failure must remain visible
                method_rows.append(_skipped_method_row("SPI-score-transporter", str(error)))
        if bool(params.get("include_semicp", True)):
            try:
                threshold, semicp_diag = semicp_nnm_threshold(
                    paired_true_scores=true_score_roles["learning"],
                    paired_synthetic_scores=sdag_learning,
                    trusted_calibration_scores=true_score_roles["calibration"],
                    synthetic_pool_scores=sdag_pool,
                    tau=tau,
                )
                method_rows.append(
                    _evaluate_global_radius_method(
                        method="SemiCP-NNM",
                        threshold=threshold,
                        calibration_k=int(semicp_diag.get("calibration_k", 0)),
                        calibration_m=int(
                            len(true_score_roles["calibration"]) + len(sdag_pool)
                        ),
                        s_eval=true_score_roles["evaluation"],
                        x_eval=x[roles["evaluation"]],
                        s_slice_select=true_score_roles["slice_select"],
                        x_slice_select=x[roles["slice_select"]],
                        x_partition=x[roles["partition"]],
                        tau=tau,
                        k_values=k_values,
                        seed=metric_seed,
                        outcome_scale=outcome_scale,
                        include_ert=include_ert,
                        diagnostics=semicp_diag,
                    )
                )
            except Exception as error:
                method_rows.append(_skipped_method_row("SemiCP-NNM", str(error)))

    for method, beta in betas.items():
        row = _evaluate_method(
            method=method,
            beta=beta,
            feature_fn=feature_fn,
            s_cal=true_score_roles["calibration"],
            x_cal=x[roles["calibration"]],
            s_eval=true_score_roles["evaluation"],
            x_eval=x[roles["evaluation"]],
            x_partition=x[roles["partition"]],
            x_slice_select=x[roles["slice_select"]],
            s_slice_select=true_score_roles["slice_select"],
            tau=tau,
            k_values=k_values,
            seed=metric_seed,
            outcome_scale=outcome_scale,
            include_ert=include_ert,
            include_metric_bridge=include_metric_bridge,
        )
        metadata = diagnostics["updates"][method]
        row["lambda"] = metadata.get("lambda")
        row["update_norm"] = metadata.get("update_norm", 0.0)
        row["update_clipped"] = metadata.get("update_clipped", 0.0)
        row["hessian_condition"] = metadata.get("hessian_condition", float("nan"))
        row["method_diagnostics"] = metadata
        method_rows.append(row)

    if bool(params.get("include_neural_bridge", False)):
        neural_hidden = int(params.get("neural_bridge_hidden", 128))
        neural_epochs = int(params.get("neural_bridge_epochs", 100))
        neural_lr = float(params.get("neural_bridge_learning_rate", 2e-3))
        neural_weights = [
            float(value)
            for value in params.get("neural_bridge_positive_weights", [0.5, 1.0])
        ]
        neural_signed = [
            float(value)
            for value in params.get("neural_bridge_signed_powers", [0.5, 1.0])
        ]
        placebo_rng = np.random.default_rng(
            int(params.get("representation_fit_seed", seed + 4)) + 271828
        )
        placebo_pool = np.asarray(sdag_pool)[placebo_rng.permutation(len(sdag_pool))]
        specifications: list[tuple[str, str, float, np.ndarray | None]] = [
            ("Neural-RCP-supervised", "supervised", 0.0, None)
        ]
        for weight in neural_weights:
            specifications.extend(
                [
                    (
                        f"Neural-RCP-weighted-genuine-w{weight:g}",
                        "positive",
                        weight,
                        sdag_pool,
                    ),
                    (
                        f"Neural-RCP-weighted-placebo-w{weight:g}",
                        "positive",
                        weight,
                        placebo_pool,
                    ),
                ]
            )
        for power in neural_signed:
            specifications.append(
                (
                    f"Neural-RCP-signed-lambda-{power:g}",
                    "signed",
                    power,
                    sdag_pool,
                )
            )

        for neural_method, objective, power, pool_scores in specifications:
            try:
                neural_fit = fit_neural_threshold(
                    x[roles["learning"]],
                    true_score_roles["learning"],
                    tau=tau,
                    seed=int(params.get("neural_bridge_seed", seed + 8800)),
                    x_pool=x[roles["synthetic_pool"]],
                    score_pool=pool_scores,
                    score_paired_synthetic=sdag_learning,
                    objective=objective,
                    power=power,
                    hidden=neural_hidden,
                    epochs=neural_epochs,
                    learning_rate=neural_lr,
                )
                if not bool(neural_fit.diagnostics["finite"]):
                    method_rows.append(
                        _skipped_method_row(
                            neural_method,
                            "non-finite neural optimization diagnostic",
                        )
                    )
                    method_rows[-1]["method_diagnostics"] = neural_fit.diagnostics
                    continue
                neural_row = _evaluate_threshold_predictor(
                    method=neural_method,
                    threshold_predict=neural_fit.predict,
                    s_cal=true_score_roles["calibration"],
                    x_cal=x[roles["calibration"]],
                    s_eval=true_score_roles["evaluation"],
                    x_eval=x[roles["evaluation"]],
                    x_partition=x[roles["partition"]],
                    x_slice_select=x[roles["slice_select"]],
                    s_slice_select=true_score_roles["slice_select"],
                    tau=tau,
                    k_values=k_values,
                    seed=metric_seed,
                    outcome_scale=outcome_scale,
                    include_ert=include_ert,
                    include_metric_bridge=include_metric_bridge,
                )
                neural_row.update(
                    {
                        "lambda": power if objective == "signed" else None,
                        "update_norm": 0.0,
                        "update_clipped": 0.0,
                        "hessian_condition": float("nan"),
                        "method_diagnostics": neural_fit.diagnostics,
                    }
                )
                method_rows.append(neural_row)
            except Exception as error:
                failed = _skipped_method_row(neural_method, str(error))
                failed["method_diagnostics"] = {
                    "objective": objective,
                    "power": power,
                    "exception_type": type(error).__name__,
                }
                method_rows.append(failed)

    if bool(params.get("include_cqr_family", False)):
        max_outputs = int(params.get("cqr_max_outputs", 8))
        if y.shape[1] > max_outputs:
            reason = f"outcome dimension {y.shape[1]} exceeds cqr_max_outputs={max_outputs}"
            method_rows.extend(
                [_skipped_method_row("CQR", reason), _skipped_method_row("CQR-PPI-unit-one-step", reason)]
            )
        else:
            phi_learning = feature_fn(x[roles["learning"]])
            phi_pool = feature_fn(x[roles["synthetic_pool"]])
            cqr_alpha = 1.0 - tau
            cqr = fit_cqr_supervised(
                phi_learning,
                y[roles["learning"]],
                alpha=cqr_alpha,
                quantile_alpha=float(params.get("quantile_alpha", 1e-5)),
            )
            method_rows.append(
                _evaluate_cqr_method(
                    method="CQR",
                    cqr_model=cqr,
                    feature_fn=feature_fn,
                    y_cal=y[roles["calibration"]],
                    x_cal=x[roles["calibration"]],
                    y_eval=y[roles["evaluation"]],
                    x_eval=x[roles["evaluation"]],
                    y_slice_select=y[roles["slice_select"]],
                    x_slice_select=x[roles["slice_select"]],
                    x_partition=x[roles["partition"]],
                    tau=tau,
                    k_values=k_values,
                    seed=metric_seed,
                    outcome_scale=outcome_scale,
                    include_ert=include_ert,
                )
            )
            cqr_ppi = fit_cqr_ppi_one_step(
                phi_learning,
                y[roles["learning"]],
                ydag_learning,
                phi_pool,
                ydag_pool,
                alpha=cqr_alpha,
                quantile_alpha=float(params.get("quantile_alpha", 1e-5)),
                density_delta_tau=float(params.get("cqr_density_delta_tau", 0.02)),
                ridge=float(params.get("ridge", 1e-3)),
                max_update_norm=float(params.get("max_update_norm", 5.0)),
            )
            method_rows.append(
                _evaluate_cqr_method(
                    method="CQR-PPI-unit-one-step",
                    cqr_model=cqr_ppi,
                    feature_fn=feature_fn,
                    y_cal=y[roles["calibration"]],
                    x_cal=x[roles["calibration"]],
                    y_eval=y[roles["evaluation"]],
                    x_eval=x[roles["evaluation"]],
                    y_slice_select=y[roles["slice_select"]],
                    x_slice_select=x[roles["slice_select"]],
                    x_partition=x[roles["partition"]],
                    tau=tau,
                    k_values=k_values,
                    seed=metric_seed,
                    outcome_scale=outcome_scale,
                    include_ert=include_ert,
                )
            )

    budget_diagnostics: dict[str, Any] | None = None
    if bool(params.get("include_budget_matched", True)):
        extra = np.unique(
            np.concatenate(
                [
                    roles["preprocessing"],
                    roles["teacher_train"],
                    roles["teacher_val"],
                    roles["mean_train"],
                    roles["representation_train"],
                    roles["learning"],
                ]
            )
        )
        budget_params = dict(params)
        budget_params["_calibration_indices"] = roles["calibration"]
        budget_params["_evaluation_indices"] = roles["evaluation"]
        budget_params["_slice_indices"] = roles["slice_select"]
        (
            budget_beta,
            budget_feature_fn,
            budget_score_cal,
            budget_score_eval,
            budget_score_slice,
            budget_diagnostics,
        ) = _budget_matched_supervised(
            x=x,
            y=y,
            extra_indices=extra,
            x_cal=x[roles["calibration"]],
            x_eval=x[roles["evaluation"]],
            x_slice=x[roles["slice_select"]],
            tau=tau,
            params=budget_params,
            threads=threads,
            seed=seed + 30,
        )
        row = _evaluate_method(
            method="Budget-matched-supervised-RCP",
            beta=budget_beta,
            feature_fn=budget_feature_fn,
            s_cal=budget_score_cal,
            x_cal=x[roles["calibration"]],
            s_eval=budget_score_eval,
            x_eval=x[roles["evaluation"]],
            x_partition=x[roles["partition"]],
            x_slice_select=x[roles["slice_select"]],
            s_slice_select=budget_score_slice,
            tau=tau,
            k_values=k_values,
            seed=metric_seed,
            outcome_scale=outcome_scale,
            include_ert=include_ert,
            include_metric_bridge=include_metric_bridge,
        )
        row.update(
            {
                "lambda": 0.0,
                "update_norm": 0.0,
                "update_clipped": 0.0,
                "hessian_condition": float("nan"),
            }
        )
        method_rows.append(row)

    if bool(params.get("emit_conditional_profiles", False)):
        _attach_reference_difficulty_profiles(
            method_rows,
            tau=tau,
            bins=int(params.get("conditional_profile_bins", 10)),
        )
    else:
        # Temporary arrays are always stripped before JSON serialization.
        for method_row in method_rows:
            for private_key in [
                "_profile_radius_partition",
                "_profile_radius_eval",
                "_profile_covered_eval",
            ]:
                method_row.pop(private_key, None)

    role_hashes = {name: hash_indices(indices) for name, indices in roles.items()}
    reservoir_hashes = {name: hash_indices(indices) for name, indices in reservoirs.items()}
    representation_labels = (
        int(len(roles["representation_train"]))
        if representation_mode == "supervised_score_screening"
        else 0
    )
    oracle_pool_access = int(len(roles["synthetic_pool"])) if oracle_synthetic else 0
    result = {
        "experiment": task["experiment"],
        "dataset": dataset,
        "seed": seed,
        "role_seed": role_seed,
        "teacher_strength": teacher_strength,
        "label_budget": str(params.get("label_budget", "custom")),
        "n_rows": int(len(x)),
        "x_dim": int(x.shape[1]),
        "y_dim": int(y.shape[1]),
        "tau": tau,
        "teacher_validation_mse": teacher_val_mse,
        "teacher_parameters": teacher_parameters,
        "mean_parameters": mean_parameters,
        "diagnostics": diagnostics,
        "budget_diagnostics": budget_diagnostics,
        "role_hashes": role_hashes,
        "reservoir_hashes": reservoir_hashes,
        "preprocessing_fit_role": "preprocessing",
        "representation_fit_role": "representation_train",
        "representation_mode": representation_mode,
        "metric_seed": metric_seed,
        "coverage_partition": "audited NumPy k-means on an independent partition role",
        "score_geometry": "l_infinity",
        "oracle_synthetic_labels": oracle_synthetic,
        "resource_ledger": {
            "preprocessing_labels": int(len(roles["preprocessing"])),
            "external_teacher_labels": 0 if oracle_synthetic else int(len(roles["teacher_train"])),
            "teacher_validation_labels": 0 if oracle_synthetic else int(len(roles["teacher_val"])),
            "target_mean_labels": int(len(roles["mean_train"])),
            "representation_training_covariates": int(len(roles["representation_train"])),
            "representation_training_labels": representation_labels,
            "target_quantile_learning_labels": int(len(roles["learning"])),
            "final_calibration_labels": int(len(roles["calibration"])),
            "evaluation_only_labels": int(len(roles["slice_select"]) + len(roles["evaluation"])),
            "synthetic_pool_unlabeled_covariates": int(len(roles["synthetic_pool"])),
            "synthetic_pool_true_outcomes_accessed": oracle_pool_access,
            "oracle_outcome_access": oracle_pool_access,
            "target_trusted_method_labels": int(
                len(roles["mean_train"])
                + representation_labels
                + len(roles["learning"])
                + len(roles["calibration"])
            ),
            "end_to_end_outcome_access": int(
                len(roles["preprocessing"])
                + (0 if oracle_synthetic else len(roles["teacher_train"]))
                + (0 if oracle_synthetic else len(roles["teacher_val"]))
                + len(roles["mean_train"])
                + representation_labels
                + len(roles["learning"])
                + len(roles["calibration"])
                + oracle_pool_access
            ),
        },
        "methods": method_rows,
    }
    return result, {}
