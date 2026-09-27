from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from matplotlib.ticker import NullFormatter

from .conformal import calibrate_scores
from .dgp import ScoreDGP
from .io import atomic_write_json, stable_hash
from .metrics import population_coverage_metrics
from .models import boundary_score, estimate_geometry, fit_linear_quantile, one_step_update
from .simulation import population_constants, population_target


EPSILONS = tuple(2.0 ** (-index) for index in range(1, 9))
N_VALUES = (400, 800, 1600, 3200, 6400)
N_SAMPLE_VALUES = (200, 400, 800, 1600)
DEFAULT_SEEDS = tuple(range(30))
N_SWEEP_CALIBRATION_SIZE = 6400
REFERENCE_N = 400
REFERENCE_N_POOL = 1600
PALETTE = {
    "oracle": "#2F5D8C",
    "supervised": "#B36A2E",
    "linear": "#2F5D8C",
    "quadratic": "#B36A2E",
    "reference": "#383838",
    "grid": "#D9DEE5",
}


@dataclass(frozen=True)
class FollowupConfig:
    tau: float = 0.9
    heterogeneity: float = 0.35
    coupling: float = 0.75
    quadratic: float = 0.9
    synthetic_bias: float = 0.5
    population_mc: int = 200_000
    population_seed: int = 99_173
    bootstrap_draws: int = 2_000
    quadrature_nodes: int = 2_048
    n_sweep_calibration_ratio: int = 1
    n_sweep_pool_ratio: int = 4
    n_sweep_values: tuple[int, ...] = N_SAMPLE_VALUES
    n_pool_sweep_values: tuple[int, ...] = N_VALUES
    n_pool_sweep_n: int = REFERENCE_N
    n_pool_sweep_m: int = N_SWEEP_CALIBRATION_SIZE


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values))


def _weighted_l2(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(values))))


def _weighted_standardize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = values - _weighted_mean(values, weights)
    scale = math.sqrt(max(_weighted_mean(np.square(centered), weights), 1e-24))
    return centered / scale


def _population_correction(
    dgp: ScoreDGP,
    q_values: np.ndarray,
    x_values: np.ndarray,
    weights: np.ndarray,
    tau: float,
) -> float:
    def equation(gamma: float) -> float:
        return _weighted_mean(dgp.cdf(q_values + gamma, x_values), weights) - tau

    return float(brentq(equation, -20.0, 20.0, xtol=1e-14, rtol=1e-14))


def run_derivative_check(config: FollowupConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    nodes, raw_weights = np.polynomial.legendre.leggauss(config.quadrature_nodes)
    x = np.asarray(nodes, dtype=float)
    weights = np.asarray(raw_weights, dtype=float) / 2.0
    dgp = ScoreDGP(
        tau=config.tau,
        heterogeneity=config.heterogeneity,
        coupling=config.coupling,
        quadratic=config.quadratic,
        synthetic_bias=config.synthetic_bias,
    )
    q_star = dgp.q_star(x)
    f_star = dgp.density(q_star, x)
    bar_f = _weighted_mean(f_star, weights)
    perturbations = {
        "constant": np.ones_like(x),
        "standardized_linear": _weighted_standardize(x, weights),
        "standardized_quadratic": _weighted_standardize(x**2 - 1.0 / 3.0, weights),
    }
    rows: list[dict[str, Any]] = []
    for perturbation, e in perturbations.items():
        projection = f_star * (e - _weighted_mean(f_star * e, weights) / bar_f)
        for epsilon in EPSILONS:
            q_perturbed = q_star + epsilon * e
            gamma = _population_correction(dgp, q_perturbed, x, weights, config.tau)
            deployed_threshold = q_perturbed + gamma
            exact_curve = dgp.cdf(deployed_threshold, x) - config.tau
            first_order_curve = epsilon * projection
            remainder = exact_curve - first_order_curve
            rows.append(
                {
                    "perturbation": perturbation,
                    "epsilon": epsilon,
                    "population_correction": gamma,
                    "exact_deployed_l2": _weighted_l2(exact_curve, weights),
                    "first_order_l2": _weighted_l2(first_order_curve, weights),
                    "remainder_l2": _weighted_l2(remainder, weights),
                    "remainder_over_epsilon": _weighted_l2(remainder, weights) / epsilon,
                    "max_abs_threshold_difference": float(
                        np.max(np.abs(deployed_threshold - q_star))
                    ),
                    "max_abs_coverage_difference": float(np.max(np.abs(exact_curve))),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["perturbation", "epsilon"], ascending=[True, False])
    frame["adjacent_loglog_slope"] = np.nan
    summaries: list[dict[str, Any]] = []
    for perturbation, indices in frame.groupby("perturbation", sort=False).groups.items():
        ordered_indices = list(indices)
        ordered = frame.loc[ordered_indices]
        remainder = ordered["remainder_l2"].to_numpy(float)
        epsilon = ordered["epsilon"].to_numpy(float)
        adjacent = np.log(remainder[:-1] / remainder[1:]) / np.log(epsilon[:-1] / epsilon[1:])
        frame.loc[ordered_indices[:-1], "adjacent_loglog_slope"] = adjacent
        smallest = ordered.nsmallest(5, "epsilon")
        if perturbation == "constant":
            slope = float("nan")
        else:
            slope = float(
                np.polyfit(
                    np.log(smallest["epsilon"].to_numpy(float)),
                    np.log(smallest["remainder_l2"].to_numpy(float)),
                    deg=1,
                )[0]
            )
        summaries.append(
            {
                "perturbation": perturbation,
                "small_epsilon_loglog_slope": slope,
                "max_abs_threshold_difference": float(
                    ordered["max_abs_threshold_difference"].max()
                ),
                "max_abs_coverage_difference": float(
                    ordered["max_abs_coverage_difference"].max()
                ),
                "smallest_remainder_over_epsilon": float(
                    ordered.nsmallest(1, "epsilon")["remainder_over_epsilon"].iloc[0]
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    constant = summary.loc[summary["perturbation"] == "constant"].iloc[0]
    nonconstant = summary.loc[summary["perturbation"] != "constant"]
    acceptance = {
        "constant_shift_threshold_max_below_1e_10": bool(
            constant["max_abs_threshold_difference"] < 1e-10
        ),
        "constant_shift_coverage_max_below_1e_10": bool(
            constant["max_abs_coverage_difference"] < 1e-10
        ),
        "nonconstant_slopes_between_1_8_and_2_2": bool(
            nonconstant["small_epsilon_loglog_slope"].between(1.8, 2.2).all()
        ),
        "remainder_over_epsilon_decreases": bool(
            all(
                group.sort_values("epsilon", ascending=False)["remainder_over_epsilon"].iloc[-1]
                < group.sort_values("epsilon", ascending=False)["remainder_over_epsilon"].iloc[0]
                for name, group in frame.groupby("perturbation")
                if name != "constant"
            )
        ),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    return frame, summary, acceptance


def _role_rng(seed: int, role: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([20260805, int(seed), int(role)]))


def _sample_x(dgp: ScoreDGP, size: int, seed: int, role: int) -> np.ndarray:
    return dgp.sample_x(size, _role_rng(seed, role))


def _sample_true_score(
    dgp: ScoreDGP, x: np.ndarray, seed: int, role: int
) -> tuple[np.ndarray, np.ndarray]:
    z = _role_rng(seed, role).normal(size=len(x))
    return dgp.sample_score(x, _role_rng(seed, role + 10_000), z_shared=z)


def _sample_synthetic_score(
    dgp: ScoreDGP,
    x: np.ndarray,
    z_true: np.ndarray,
    seed: int,
    role: int,
) -> np.ndarray:
    return dgp.synthetic_score(x, z_true, "well_coupled", _role_rng(seed, role))


def _beta_second_moment(m: int, tau: float) -> float:
    k = int(math.ceil((m + 1) * tau))
    mean = k / (m + 1)
    variance = k * (m + 1 - k) / ((m + 1) ** 2 * (m + 2))
    return float(variance + (mean - tau) ** 2)


def _population_calibration_on_grid(
    dgp: ScoreDGP, q_values: np.ndarray, x_values: np.ndarray, tau: float
) -> float:
    def equation(gamma: float) -> float:
        return float(np.mean(dgp.cdf(q_values + gamma, x_values)) - tau)

    return float(brentq(equation, -20.0, 20.0, xtol=1e-13, rtol=1e-13))


def _run_scaling_task(
    *,
    sweep: str,
    n: int,
    n_pool: int,
    m: int,
    seed: int,
    dgp: ScoreDGP,
    target: dict[str, Any],
    constants: dict[str, float],
    fixed_power: float,
) -> dict[str, Any]:
    x = _sample_x(dgp, n, seed, 101)
    score, z = _sample_true_score(dgp, x, seed, 102)
    score_dagger = _sample_synthetic_score(dgp, x, z, seed, 103)
    x_pool = _sample_x(dgp, n_pool, seed, 201)
    score_pool_true, z_pool = _sample_true_score(dgp, x_pool, seed, 202)
    del score_pool_true
    score_pool = _sample_synthetic_score(dgp, x_pool, z_pool, seed, 203)
    x_cal = _sample_x(dgp, m, seed, 301)
    score_cal, _ = _sample_true_score(dgp, x_cal, seed, 302)

    feature_fn = lambda values: dgp.features(np.asarray(values), well_specified=True)
    phi = feature_fn(x)
    pilot = fit_linear_quantile(phi, score, dgp.tau)
    q = pilot.predict_features(phi)
    hessian, _, _ = estimate_geometry(phi, dgp.density(q, x), ridge=1e-4)
    psi = boundary_score(phi, score, q, dgp.tau)
    psi_dagger = boundary_score(phi, score_dagger, q, dgp.tau)
    phi_pool = feature_fn(x_pool)
    q_pool = pilot.predict_features(phi_pool)
    psi_pool = boundary_score(phi_pool, score_pool, q_pool, dgp.tau)

    x_eval = np.asarray(target["x_grid"], dtype=float)
    pi_target = np.asarray(target["pi"], dtype=float)
    result_rows: list[dict[str, Any]] = []
    for method, power in (("supervised", 0.0), ("oracle_power", fixed_power)):
        beta, update = one_step_update(
            pilot.coef_,
            hessian,
            psi,
            psi_dagger,
            psi_pool,
            power=power,
            max_update_norm=5.0,
        )
        q_cal = feature_fn(x_cal) @ beta
        finite_calibration = calibrate_scores(score_cal, q_cal, dgp.tau)
        q_eval = feature_fn(x_eval) @ beta
        gamma_population = _population_calibration_on_grid(dgp, q_eval, x_eval, dgp.tau)
        pi_population = dgp.cdf(q_eval + gamma_population, x_eval)
        pi_finite = dgp.cdf(q_eval + finite_calibration.correction, x_eval)
        finite_metrics = population_coverage_metrics(pi_finite, dgp.tau)
        population_metrics = population_coverage_metrics(pi_population, dgp.tau)
        predicted_learning = (
            constants["a"] - 2.0 * constants["b"] * power + constants["c"] * power**2
        ) / n + constants["c_pool"] * power**2 / n_pool
        predicted_calibration = constants["c_cal"] * _beta_second_moment(m, dgp.tau)
        result_rows.append(
            {
                "method": method,
                "lambda": power,
                "finite_calibration_error": float(np.mean((pi_finite - pi_target) ** 2)),
                "population_calibrated_error": float(
                    np.mean((pi_population - pi_target) ** 2)
                ),
                "finite_msce": finite_metrics["msce"],
                "population_calibrated_msce": population_metrics["msce"],
                "predicted_learning_error": float(predicted_learning),
                "predicted_calibration_error": float(predicted_calibration),
                "predicted_total_error": float(predicted_learning + predicted_calibration),
                "empirical_to_predicted_total_ratio": float(
                    np.mean((pi_finite - pi_target) ** 2)
                    / max(predicted_learning + predicted_calibration, 1e-24)
                ),
                "empirical_to_predicted_learning_ratio": float(
                    np.mean((pi_population - pi_target) ** 2)
                    / max(predicted_learning, 1e-24)
                ),
                "calibration_correction": finite_calibration.correction,
                "population_calibration_correction": gamma_population,
                "update_norm": update["update_norm"],
                "update_clipped": bool(update["update_clipped"]),
                "hessian_condition": update["hessian_condition"],
            }
        )
    return {
        "sweep": sweep,
        "seed": seed,
        "n": n,
        "N": n_pool,
        "m": m,
        "mechanism": "well_coupled",
        "rows": result_rows,
    }


def build_scaling_plan(seeds: Iterable[int] = DEFAULT_SEEDS) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for n in N_SAMPLE_VALUES:
        for seed in seeds:
            plan.append(
                {
                    "sweep": "n_scaling",
                    "n": n,
                    "N": 4 * n,
                    "m": n,
                    "seed": int(seed),
                }
            )
    for n_pool in N_VALUES:
        for seed in seeds:
            plan.append(
                {
                    "sweep": "N_scaling",
                    "n": REFERENCE_N,
                    "N": n_pool,
                    "m": N_SWEEP_CALIBRATION_SIZE,
                    "seed": int(seed),
                }
            )
    return plan


def _flatten_tasks(tasks: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        shared = {key: value for key, value in task.items() if key != "rows"}
        for row in task["rows"]:
            rows.append({**shared, **row})
    return pd.DataFrame(rows)


def _group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metrics = (
        "finite_calibration_error",
        "population_calibrated_error",
        "predicted_learning_error",
        "predicted_total_error",
        "empirical_to_predicted_total_ratio",
        "empirical_to_predicted_learning_ratio",
    )
    for keys, group in frame.groupby(["sweep", "method", "n", "N", "m"], sort=True):
        record = dict(zip(("sweep", "method", "n", "N", "m"), keys))
        record["seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            values = group[metric].to_numpy(float)
            mean = float(np.mean(values))
            se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = float(np.std(values, ddof=1))
            record[f"{metric}_ci_low"] = mean - 1.96 * se
            record[f"{metric}_ci_high"] = mean + 1.96 * se
        records.append(record)
    return pd.DataFrame(records)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / max(denominator, 1e-24)
    return float(slope), float(intercept), float(r_squared)


def _bootstrap_fit(
    frame: pd.DataFrame,
    sweep: str,
    method: str,
    metric: str,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    selected = frame[(frame["sweep"] == sweep) & (frame["method"] == method)].copy()
    seed_values = np.sort(selected["seed"].unique())
    estimates: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(seed_values, size=len(seed_values), replace=True)
        pieces = [selected[selected["seed"] == seed] for seed in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        if sweep == "n_scaling":
            means = boot.groupby("n", as_index=False)[metric].mean().sort_values("n")
            estimate = float(
                np.polyfit(np.log(means["n"]), np.log(means[metric]), deg=1)[0]
            )
        else:
            means = boot.groupby("N", as_index=False)[metric].mean().sort_values("N")
            estimate = _linear_fit(1.0 / means["N"].to_numpy(float), means[metric].to_numpy(float))[0]
        estimates.append(estimate)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {"bootstrap_ci_low": float(lower), "bootstrap_ci_high": float(upper)}


def _scaling_fits(
    frame: pd.DataFrame, constants: dict[str, float], fixed_power: float, draws: int
) -> pd.DataFrame:
    rng = np.random.default_rng(20260805)
    records: list[dict[str, Any]] = []
    for method in ("supervised", "oracle_power"):
        n_rows = frame[(frame["sweep"] == "n_scaling") & (frame["method"] == method)]
        n_means = (
            n_rows.groupby("n", as_index=False)["finite_calibration_error"].mean().sort_values("n")
        )
        slope = float(
            np.polyfit(
                np.log(n_means["n"].to_numpy(float)),
                np.log(n_means["finite_calibration_error"].to_numpy(float)),
                deg=1,
            )[0]
        )
        records.append(
            {
                "sweep": "n_scaling",
                "method": method,
                "fit": "log(error) ~ log(n)",
                "estimate": slope,
                "theory_target": -1.0,
                "intercept": float("nan"),
                "r_squared": float(
                    np.corrcoef(
                        np.log(n_means["n"]), np.log(n_means["finite_calibration_error"])
                    )[0, 1]
                    ** 2
                ),
                **_bootstrap_fit(
                    frame,
                    "n_scaling",
                    method,
                    "finite_calibration_error",
                    draws,
                    rng,
                ),
            }
        )
        N_rows = frame[(frame["sweep"] == "N_scaling") & (frame["method"] == method)]
        N_means = (
            N_rows.groupby("N", as_index=False)["population_calibrated_error"]
            .mean()
            .sort_values("N")
        )
        fit_slope, fit_intercept, fit_r2 = _linear_fit(
            1.0 / N_means["N"].to_numpy(float),
            N_means["population_calibrated_error"].to_numpy(float),
        )
        theory_slope = 0.0 if method == "supervised" else constants["c_pool"] * fixed_power**2
        records.append(
            {
                "sweep": "N_scaling",
                "method": method,
                "fit": "population-calibrated error ~ 1/N",
                "estimate": fit_slope,
                "theory_target": float(theory_slope),
                "intercept": fit_intercept,
                "r_squared": fit_r2,
                **_bootstrap_fit(
                    frame,
                    "N_scaling",
                    method,
                    "population_calibrated_error",
                    draws,
                    rng,
                ),
            }
        )
    return pd.DataFrame(records)


def run_scaling_experiment(
    output_dir: Path,
    config: FollowupConfig,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dgp = ScoreDGP(
        tau=config.tau,
        heterogeneity=config.heterogeneity,
        coupling=config.coupling,
        quadratic=config.quadratic,
        synthetic_bias=config.synthetic_bias,
    )
    target = population_target(dgp, well_specified=True)
    constants = population_constants(
        dgp,
        "well_coupled",
        target,
        n_mc=config.population_mc,
        seed=config.population_seed,
    )
    kappa_reference = REFERENCE_N / REFERENCE_N_POOL
    fixed_power = float(
        np.clip(
            constants["b"] / (constants["c"] + kappa_reference * constants["c_pool"]),
            0.0,
            1.5,
        )
    )
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    plan = build_scaling_plan(seeds)
    for descriptor in plan:
        task_id = stable_hash({"version": 16, **descriptor}, length=20)
        path = tasks_dir / f"{task_id}.json"
        if path.exists() and not force:
            payload = json.loads(path.read_text(encoding="utf-8"))
            completed.append(payload["result"])
            continue
        result = _run_scaling_task(
            sweep=descriptor["sweep"],
            n=descriptor["n"],
            n_pool=descriptor["N"],
            m=descriptor["m"],
            seed=descriptor["seed"],
            dgp=dgp,
            target=target,
            constants=constants,
            fixed_power=fixed_power,
        )
        atomic_write_json(
            path,
            {
                "task_id": task_id,
                "descriptor": descriptor,
                "fixed_population_oracle_power": fixed_power,
                "population_constants": constants,
                "result": result,
            },
        )
        completed.append(result)
    frame = _flatten_tasks(completed)
    summary = _group_summary(frame)
    fits = _scaling_fits(frame, constants, fixed_power, config.bootstrap_draws)
    n_fit = fits[(fits["sweep"] == "n_scaling") & (fits["method"] == "oracle_power")].iloc[0]
    N_fit = fits[(fits["sweep"] == "N_scaling") & (fits["method"] == "oracle_power")].iloc[0]
    N_supervised = fits[(fits["sweep"] == "N_scaling") & (fits["method"] == "supervised")].iloc[0]
    acceptance = {
        "logical_tasks_expected_270": len(plan) == 270,
        "logical_tasks_observed_270": len(completed) == 270,
        "unique_seed_setting_rows_540": len(frame) == 540,
        "no_clipped_updates": not bool(frame["update_clipped"].any()),
        "all_finite_required_metrics": bool(
            np.isfinite(
                frame[
                    [
                        "finite_calibration_error",
                        "population_calibrated_error",
                        "predicted_learning_error",
                        "predicted_total_error",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "oracle_n_slope_ci_contains_minus_one": bool(
            n_fit["bootstrap_ci_low"] <= -1.0 <= n_fit["bootstrap_ci_high"]
        ),
        "oracle_N_slope_positive": bool(N_fit["estimate"] > 0.0),
        "oracle_N_slope_ci_positive": bool(N_fit["bootstrap_ci_low"] > 0.0),
        "supervised_N_slope_ci_contains_zero": bool(
            N_supervised["bootstrap_ci_low"] <= 0.0 <= N_supervised["bootstrap_ci_high"]
        ),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    metadata = {
        "population_constants": constants,
        "fixed_population_oracle_power": fixed_power,
        "acceptance": acceptance,
    }
    return frame, summary, fits, metadata


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "axes.edgecolor": "#4C4C4C",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "savefig.bbox": "tight",
        }
    )


def plot_derivative(frame: pd.DataFrame, output_base: Path) -> None:
    _set_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for perturbation, label, color, marker in (
        ("standardized_linear", "linear perturbation", PALETTE["linear"], "o"),
        ("standardized_quadratic", "quadratic perturbation", PALETTE["quadratic"], "s"),
    ):
        subset = frame[frame["perturbation"] == perturbation].sort_values("epsilon")
        axes[0].loglog(
            subset["epsilon"],
            subset["remainder_l2"],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.5,
            label=label,
        )
        axes[1].loglog(
            subset["epsilon"],
            subset["remainder_over_epsilon"],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.5,
            label=label,
        )
    reference = frame[frame["perturbation"] == "standardized_linear"].sort_values("epsilon")
    scale = float(reference["remainder_l2"].iloc[-1] / reference["epsilon"].iloc[-1] ** 2)
    axes[0].loglog(
        reference["epsilon"],
        scale * reference["epsilon"] ** 2,
        color=PALETTE["reference"],
        linestyle="--",
        linewidth=1.1,
        label=r"reference $C\epsilon^2$",
    )
    axes[0].set_title("Post-conformal derivative remainder")
    axes[0].set_xlabel(r"perturbation size $\epsilon$")
    axes[0].set_ylabel(r"$\|r_\epsilon\|_{L_2(P_X)}$")
    axes[1].set_title("First-order normalization")
    axes[1].set_xlabel(r"perturbation size $\epsilon$")
    axes[1].set_ylabel(r"$\|r_\epsilon\|_2/\epsilon$")
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[1].legend(loc="upper left", fontsize=7.5)
    figure.suptitle("Direct numerical check of the density-weighted projection", y=1.03, fontsize=11)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def plot_scaling(summary: pd.DataFrame, fits: pd.DataFrame, output_base: Path) -> None:
    del fits
    _set_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    for method, label, color, marker in (
        ("oracle_power", "oracle synthetic power", PALETTE["oracle"], "o"),
        ("supervised", "supervised", PALETTE["supervised"], "s"),
    ):
        subset = summary[(summary["sweep"] == "n_scaling") & (summary["method"] == method)].sort_values("n")
        axes[0].loglog(
            subset["n"],
            subset["finite_calibration_error_mean"],
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=4,
            label=label,
        )
        axes[0].fill_between(
            subset["n"],
            subset["finite_calibration_error_ci_low"],
            subset["finite_calibration_error_ci_high"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axes[0].loglog(
            subset["n"],
            subset["predicted_total_error_mean"],
            color=color,
            linestyle="--",
            linewidth=1.0,
        )
    axes[0].set_title(r"Joint $n,N,m$ expansion")
    axes[0].set_xlabel(r"trusted learning size $n$ ($N=4n, m=n$)")
    axes[0].set_ylabel("stochastic deployment error")
    axes[0].set_xticks([200, 400, 800, 1600])
    axes[0].set_xticklabels(["200", "400", "800", "1600"])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(loc="lower left", fontsize=7.5)
    oracle = summary[(summary["sweep"] == "N_scaling") & (summary["method"] == "oracle_power")].sort_values("N")
    x = 1.0 / oracle["N"].to_numpy(float)
    y = oracle["population_calibrated_error_mean"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, deg=1)
    axes[1].errorbar(
        x,
        y,
        yerr=np.vstack(
            [
                y - oracle["population_calibrated_error_ci_low"].to_numpy(float),
                oracle["population_calibrated_error_ci_high"].to_numpy(float) - y,
            ]
        ),
        color=PALETTE["oracle"],
        marker="o",
        linestyle="none",
        linewidth=1.1,
        markersize=4,
        capsize=2,
        label="empirical mean (95% MC interval)",
    )
    x_line = np.linspace(0.0, max(x) * 1.06, 200)
    axes[1].plot(
        x_line,
        slope * x_line + intercept,
        color=PALETTE["reference"],
        linestyle="--",
        linewidth=1.1,
        label="empirical linear fit",
    )
    axes[1].plot(
        x,
        oracle["predicted_learning_error_mean"].to_numpy(float),
        color=PALETTE["oracle"],
        linewidth=1.4,
        marker="o",
        markersize=3,
        markerfacecolor="white",
        label="theorem leading term",
    )
    axes[1].set_title(r"Pool correction versus $1/N$")
    axes[1].set_xlabel(r"inverse synthetic-pool size $1/N$")
    axes[1].set_ylabel("population-calibrated error")
    axes[1].legend(loc="lower right", fontsize=7.0)
    figure.suptitle("Three-resource scaling in the well-specified residual-coupled cell", y=1.03, fontsize=11)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_all(output_dir: Path, config: FollowupConfig, force: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir = output_dir / "aggregate"
    figures_dir = output_dir / "figures"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    derivative, derivative_summary, derivative_acceptance = run_derivative_check(config)
    _write_csv(derivative, aggregate_dir / "post_conformal_derivative_rows.csv")
    _write_csv(derivative_summary, aggregate_dir / "post_conformal_derivative_summary.csv")
    plot_derivative(derivative, figures_dir / "post_conformal_derivative")
    scaling, scaling_summary, scaling_fits, scaling_metadata = run_scaling_experiment(
        output_dir, config, force=force
    )
    _write_csv(scaling, aggregate_dir / "three_resource_task_method_rows.csv")
    _write_csv(scaling_summary, aggregate_dir / "three_resource_scaling_summary.csv")
    _write_csv(scaling_fits, aggregate_dir / "three_resource_scaling_fits.csv")
    plot_scaling(scaling_summary, scaling_fits, figures_dir / "three_resource_scaling")
    acceptance = {
        "post_conformal_derivative": derivative_acceptance,
        "three_resource_scaling": scaling_metadata["acceptance"],
    }
    acceptance["passed"] = bool(
        derivative_acceptance["passed"] and scaling_metadata["acceptance"]["passed"]
    )
    payload = {
        "version": 16,
        "status": "COMPLETE" if acceptance["passed"] else "COMPLETE_WITH_FAILED_ACCEPTANCE",
        "config": asdict(config),
        "fixed_population_oracle_power": scaling_metadata["fixed_population_oracle_power"],
        "population_constants": scaling_metadata["population_constants"],
        "logical_randomized_tasks": 270,
        "task_method_rows": int(len(scaling)),
        "acceptance": acceptance,
    }
    atomic_write_json(aggregate_dir / "acceptance.json", acceptance)
    atomic_write_json(output_dir / "RUN_MANIFEST.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v16_theory_followup"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--population-mc", type=int, default=FollowupConfig.population_mc)
    parser.add_argument("--bootstrap-draws", type=int, default=FollowupConfig.bootstrap_draws)
    parser.add_argument("--quadrature-nodes", type=int, default=FollowupConfig.quadrature_nodes)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FollowupConfig(
        population_mc=args.population_mc,
        bootstrap_draws=args.bootstrap_draws,
        quadrature_nodes=args.quadrature_nodes,
    )
    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    payload = run_all(args.output_dir, config, force=args.force)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
