from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, root

from .conformal import calibrate_scores
from .dgp import ScoreDGP, mechanism_is_certified, mechanism_is_well_specified
from .io import atomic_write_json, read_json, stable_hash
from .metrics import population_coverage_metrics
from .models import (
    boundary_score,
    cross_fitted_power,
    estimate_geometry,
    fit_linear_quantile,
    one_step_update,
    projected_constants,
    projected_cross_trace,
)


def _feature_fn(dgp: ScoreDGP, well_specified: bool):
    return lambda x: dgp.features(np.asarray(x), well_specified=well_specified)


def population_target(dgp: ScoreDGP, well_specified: bool) -> dict[str, Any]:
    x_grid = np.linspace(-1.0, 1.0, 4001)
    phi = dgp.features(x_grid, well_specified=well_specified)
    if well_specified:
        beta = np.array([3.0, 0.6, dgp.quadratic], dtype=float)
    else:

        def equation(beta_arr: np.ndarray) -> np.ndarray:
            q = phi @ beta_arr
            residual = dgp.cdf(q, x_grid) - dgp.tau
            return np.mean(residual[:, None] * phi, axis=0)

        solution = root(equation, np.array([3.0 + dgp.quadratic / 3.0, 0.6]))
        if not solution.success:
            raise RuntimeError(f"Pseudo-true root failed: {solution.message}")
        beta = np.asarray(solution.x, dtype=float)

    q = phi @ beta

    def marginal_equation(gamma: float) -> float:
        return float(np.mean(dgp.cdf(q + gamma, x_grid)) - dgp.tau)

    gamma = float(brentq(marginal_equation, -20.0, 20.0))
    pi = dgp.cdf(q + gamma, x_grid)
    return {
        "beta": beta,
        "gamma": gamma,
        "x_grid": x_grid,
        "pi": pi,
        "q": q,
    }


def population_constants(
    dgp: ScoreDGP,
    mechanism: str,
    target: dict[str, Any],
    n_mc: int,
    seed: int,
) -> dict[str, float]:
    """Monte Carlo constants and the law-of-total-covariance decomposition."""

    rng = np.random.default_rng(seed)
    x = dgp.sample_x(n_mc, rng)
    score, z = dgp.sample_score(x, rng)
    score_d = dgp.synthetic_score(x, z, mechanism, rng)
    well = mechanism_is_well_specified(mechanism)
    phi = dgp.features(x, well_specified=well)
    q = phi @ np.asarray(target["beta"])
    f_h = dgp.density(q, x)
    _, g_c, _ = estimate_geometry(phi, f_h, ridge=1e-8)
    psi = boundary_score(phi, score, q, dgp.tau)
    psi_d = boundary_score(phi, score_d, q, dgp.tau)
    constants = projected_constants(psi, psi_d, g_c)

    p = dgp.cdf(q, x)
    p_d = dgp.synthetic_cdf(q, x, mechanism)
    mean_psi = (p - dgp.tau)[:, None] * phi
    mean_psi_d = (p_d - dgp.tau)[:, None] * phi
    residual_psi = psi - mean_psi
    residual_psi_d = psi_d - mean_psi_d
    b_residual = projected_cross_trace(residual_psi, residual_psi_d, g_c)
    b_alignment = projected_cross_trace(mean_psi, mean_psi_d, g_c)
    constants.update(
        {
            "b_residual": float(b_residual),
            "b_alignment": float(b_alignment),
            "b_decomposition_error": float(constants["b"] - b_residual - b_alignment),
            "c_pool": float(constants["c"]),
        }
    )
    f_cal = dgp.density(q + float(target["gamma"]), x)
    constants["c_cal"] = float(np.mean((f_cal / np.mean(f_cal)) ** 2))
    return constants


def _cached_population_constants(
    work_dir: Path,
    dgp: ScoreDGP,
    mechanism: str,
    target: dict[str, Any],
    n_mc: int,
    seed: int,
    use_cache: bool = True,
) -> dict[str, float]:
    descriptor = {
        "version": 5,
        "mechanism": mechanism,
        "tau": dgp.tau,
        "heterogeneity": dgp.heterogeneity,
        "coupling": dgp.coupling,
        "quadratic": dgp.quadratic,
        "synthetic_bias": dgp.synthetic_bias,
        "well_specified": mechanism_is_well_specified(mechanism),
        "n_mc": int(n_mc),
        "seed": int(seed),
    }
    cache_dir = Path(work_dir) / "outputs" / "_population_cache"
    cache_path = cache_dir / f"{stable_hash(descriptor, length=24)}.json"
    if use_cache and cache_path.exists():
        payload = read_json(cache_path)
        return {key: float(value) for key, value in payload["constants"].items()}
    constants = population_constants(dgp, mechanism, target, n_mc=n_mc, seed=seed)
    if use_cache:
        atomic_write_json(cache_path, {"descriptor": descriptor, "constants": constants})
    return constants


def _setting_from_task(params: dict[str, Any]) -> dict[str, Any]:
    setting = dict(params.get("setting", {}))
    out = dict(params)
    out.pop("setting", None)
    out.update(setting)
    return out


def _population_calibration_correction(
    dgp: ScoreDGP,
    q_eval: np.ndarray,
    x_eval: np.ndarray,
    tau: float,
) -> float:
    def equation(gamma: float) -> float:
        return float(np.mean(dgp.cdf(q_eval + gamma, x_eval)) - tau)

    return float(brentq(equation, -20.0, 20.0))


def _evaluate_beta(
    *,
    label: str,
    beta: np.ndarray,
    update_meta: dict[str, float],
    power: float | None,
    dgp: ScoreDGP,
    feature_fn: Any,
    x_cal: np.ndarray,
    score_cal: np.ndarray,
    x_eval: np.ndarray,
    pi_bar: np.ndarray,
    tau: float,
    theory_learning: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    phi_cal = feature_fn(x_cal)
    phi_eval = feature_fn(x_eval)
    q_cal = phi_cal @ beta
    calibration = calibrate_scores(score_cal, q_cal, tau)
    q_eval = phi_eval @ beta
    if calibration.infinite:
        pi_finite = np.ones_like(q_eval)
    else:
        pi_finite = dgp.cdf(q_eval + calibration.correction, x_eval)
    gamma_pop = _population_calibration_correction(dgp, q_eval, x_eval, tau)
    pi_pop = dgp.cdf(q_eval + gamma_pop, x_eval)
    metrics = population_coverage_metrics(pi_finite, tau)
    pop_metrics = population_coverage_metrics(pi_pop, tau)
    row = {
        "label": label,
        "lambda": None if power is None else float(power),
        **metrics,
        "population_calibrated_msce": pop_metrics["msce"],
        "population_calibrated_het": pop_metrics["het"],
        "population_calibrated_marginal_offset": pop_metrics["marginal_offset"],
        "stochastic_error": float(np.mean((pi_finite - pi_bar) ** 2)),
        "stochastic_error_population_calibrated": float(np.mean((pi_pop - pi_bar) ** 2)),
        "procedure_bias_proxy": float(np.mean(pi_finite - pi_bar)),
        "theory_learning": float(theory_learning),
        "calibration_correction": float(calibration.correction),
        "population_calibration_correction": float(gamma_pop),
        "calibration_k": int(calibration.order_index),
        "calibration_infinite": bool(calibration.infinite),
        **update_meta,
    }
    return row, pi_finite, pi_pop


def run_mechanism_task(task: dict[str, Any], work_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    params = _setting_from_task(task["params"])
    seed = int(params.get("seed", 0))
    mechanism = str(params.get("mechanism", "well_independent"))
    tau = float(params.get("tau", 0.9))
    n = int(params.get("n", 400))
    n_ratio = float(params.get("n_to_pool", 0.25))
    n_pool = int(params.get("N", max(50, round(n / max(n_ratio, 1e-9)))))
    m = int(params.get("m", n))
    dgp = ScoreDGP(
        tau=tau,
        heterogeneity=float(params.get("heterogeneity", 0.35)),
        coupling=float(params.get("coupling", 0.75)),
        quadratic=float(params.get("quadratic", 0.9)),
        synthetic_bias=float(params.get("synthetic_bias", 0.5)),
    )
    well = mechanism_is_well_specified(mechanism)
    feature_fn = _feature_fn(dgp, well)
    target = population_target(dgp, well_specified=well)
    constants = _cached_population_constants(
        work_dir,
        dgp,
        mechanism,
        target,
        n_mc=int(params.get("population_mc", 30000)),
        seed=int(params.get("population_seed", 99173)),
        use_cache=bool(params.get("cache_population", True)),
    )
    kappa = n / n_pool
    denominator = constants["c"] + kappa * constants["c_pool"]
    lambda_oracle = constants["b"] / denominator if denominator > 1e-12 else 0.0
    allow_signed = bool(params.get("allow_signed", mechanism.startswith("signed_")))
    lambda_max = float(params.get("lambda_max", 1.5))
    lambda_oracle = float(
        np.clip(lambda_oracle, -lambda_max if allow_signed else 0.0, lambda_max)
    )

    rng = np.random.default_rng(seed)
    x = dgp.sample_x(n, rng)
    score, z = dgp.sample_score(x, rng)
    score_d = dgp.synthetic_score(x, z, mechanism, rng)
    x_pool = dgp.sample_x(n_pool, rng)
    score_pool, z_pool = dgp.sample_score(x_pool, rng)
    syn_pool = (
        score_pool
        if mechanism == "paired_oracle"
        else dgp.synthetic_score(x_pool, z_pool, mechanism, rng)
    )
    x_cal = dgp.sample_x(m, rng)
    score_cal, _ = dgp.sample_score(x_cal, rng)

    phi = feature_fn(x)
    pilot = fit_linear_quantile(phi, score, tau)
    q = pilot.predict_features(phi)
    f_train = dgp.density(q, x)
    hessian, g_c, _ = estimate_geometry(
        phi, f_train, ridge=float(params.get("ridge", 1e-4))
    )
    psi = boundary_score(phi, score, q, tau)
    psi_d = boundary_score(phi, score_d, q, tau)
    phi_pool = feature_fn(x_pool)
    q_pool = pilot.predict_features(phi_pool)
    psi_pool = boundary_score(phi_pool, syn_pool, q_pool, tau)
    sample_constants = projected_constants(psi, psi_d, g_c, psi_pool)

    plugin = cross_fitted_power(
        x,
        score,
        score_d,
        x_pool,
        syn_pool,
        feature_fn=feature_fn,
        tau=tau,
        n_splits=int(params.get("power_folds", 2)),
        ridge=float(params.get("ridge", 1e-4)),
        gate_rate=float(params.get("gate_rate", 1.0)),
        lambda_max=lambda_max,
        allow_signed=allow_signed,
        seed=seed + 100,
        density_fn=dgp.density,
    )

    if "lambda_grid" in params:
        lambda_grid = [float(value) for value in params["lambda_grid"]]
    elif allow_signed:
        lambda_grid = np.linspace(-lambda_max, lambda_max, 13).tolist()
    else:
        lambda_grid = np.linspace(0.0, lambda_max, 13).tolist()
    named_powers = {
        "supervised": 0.0,
        "unit": 1.0,
        "oracle": lambda_oracle,
        "deterministic_gate": plugin.selected_power,
    }
    all_powers: list[tuple[str, float]] = [
        (f"grid_{index:02d}", value) for index, value in enumerate(lambda_grid)
    ]
    all_powers.extend(named_powers.items())

    x_eval = np.asarray(target["x_grid"], dtype=float)
    pi_bar = np.asarray(target["pi"], dtype=float)
    rows: list[dict[str, Any]] = []
    curves: dict[str, np.ndarray] = {"x_eval": x_eval, "pseudo_truth": pi_bar}
    seen: set[tuple[str, float]] = set()
    for label, power in all_powers:
        key = (label, round(float(power), 12))
        if key in seen:
            continue
        seen.add(key)
        beta, update_meta = one_step_update(
            pilot.coef_,
            hessian,
            psi,
            psi_d,
            psi_pool,
            power=float(power),
            max_update_norm=float(params.get("max_update_norm", 5.0)),
        )
        theory_learning = (
            constants["a"] - 2 * constants["b"] * power + constants["c"] * power**2
        ) / n + constants["c_pool"] * power**2 / n_pool
        row, pi_finite, pi_pop = _evaluate_beta(
            label=label,
            beta=beta,
            update_meta=update_meta,
            power=float(power),
            dgp=dgp,
            feature_fn=feature_fn,
            x_cal=x_cal,
            score_cal=score_cal,
            x_eval=x_eval,
            pi_bar=pi_bar,
            tau=tau,
            theory_learning=float(theory_learning),
        )
        rows.append(row)
        curves[f"pi__{label}"] = pi_finite
        curves[f"pi_popcal__{label}"] = pi_pop

    if mechanism_is_certified(mechanism):
        pooled_phi = np.vstack([phi, phi_pool])
        pooled_scores = np.concatenate([score, syn_pool])
        pooled_model = fit_linear_quantile(pooled_phi, pooled_scores, tau)
        row, pi_finite, pi_pop = _evaluate_beta(
            label="certified_pooling",
            beta=pooled_model.coef_,
            update_meta={
                "update_norm": 0.0,
                "update_clipped": 0.0,
                "hessian_condition": float("nan"),
            },
            power=None,
            dgp=dgp,
            feature_fn=feature_fn,
            x_cal=x_cal,
            score_cal=score_cal,
            x_eval=x_eval,
            pi_bar=pi_bar,
            tau=tau,
            theory_learning=float(constants["a"] / (n + n_pool)),
        )
        rows.append(row)
        curves["pi__certified_pooling"] = pi_finite
        curves["pi_popcal__certified_pooling"] = pi_pop

    result = {
        "experiment": task["experiment"],
        "mechanism": mechanism,
        "seed": seed,
        "n": n,
        "N": n_pool,
        "m": m,
        "tau": tau,
        "well_specified": well,
        "certified_synthetic_law": mechanism_is_certified(mechanism),
        "population_constants": constants,
        "sample_constants": sample_constants,
        "lambda_oracle": lambda_oracle,
        "power_estimate": asdict(plugin),
        "pseudo_truth_msce": float(np.mean((pi_bar - tau) ** 2)),
        "pseudo_truth_het": float(np.var(pi_bar)),
        "rows": rows,
    }
    return result, curves


def run_conformal_null_task(
    task: dict[str, Any], work_dir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Raw-positive, post-conformal-zero influence-space diagnostic."""

    del work_dir
    params = _setting_from_task(task["params"])
    seed = int(params.get("seed", 0))
    n_mc = int(params.get("n_mc", 50000))
    strength = float(params.get("strength", 1.0))
    kappa = float(params.get("kappa", 0.25))
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n_mc)
    phi = np.column_stack([np.ones(n_mc), x])
    hessian, g_c, _ = estimate_geometry(phi, np.ones(n_mc), ridge=1e-10)
    # A constant threshold perturbation w=e_1 is mapped from score space as Jw.
    # Using this exact finite-sample direction avoids confusing an intercept
    # coordinate with the coordinate-free conformal null direction.
    null_direction = hessian @ np.array([1.0, 0.0])
    null_direction /= np.linalg.norm(null_direction)
    orthogonal_direction = np.array([-null_direction[1], null_direction[0]])
    shared = rng.normal(size=n_mc)
    orthogonal = rng.normal(size=n_mc)
    pool_noise = rng.normal(size=n_mc)
    psi = shared[:, None] * null_direction[None, :] + orthogonal[:, None] * orthogonal_direction[None, :]
    psi_d = (strength * shared + 0.25 * pool_noise)[:, None] * null_direction[None, :]
    psi_pool = rng.normal(scale=np.sqrt(strength**2 + 0.25**2), size=n_mc)[:, None] * null_direction[None, :]
    projected = projected_constants(psi, psi_d, g_c, psi_pool)
    identity = np.eye(2)
    raw = projected_constants(psi, psi_d, identity, psi_pool)
    lambdas = np.linspace(0.0, 1.5, 61)
    raw_risk = raw["a"] - 2 * raw["b"] * lambdas + (raw["c"] + kappa * raw["c_pool"]) * lambdas**2
    post_risk = projected["a"] - 2 * projected["b"] * lambdas + (
        projected["c"] + kappa * projected["c_pool"]
    ) * lambdas**2
    result = {
        "experiment": task["experiment"],
        "seed": seed,
        "strength": strength,
        "n_mc": n_mc,
        "raw_constants": raw,
        "projected_constants": projected,
        "raw_gain": float(raw_risk[0] - np.min(raw_risk)),
        "postconformal_gain": float(post_risk[0] - np.min(post_risk)),
        "raw_minimizer": float(lambdas[int(np.argmin(raw_risk))]),
        "postconformal_minimizer": float(lambdas[int(np.argmin(post_risk))]),
        "acceptance_raw_positive": bool(raw["b"] > 0.05),
        "acceptance_projected_zero": bool(abs(projected["b"]) < 5e-4),
    }
    curves = {"lambda": lambdas, "raw_risk": raw_risk, "postconformal_risk": post_risk}
    return result, curves


def signed_pinball_contribution(
    t: np.ndarray, s: float, s_dagger: float, tau: float, power: float
) -> np.ndarray:
    t = np.asarray(t, dtype=float)

    def pin(y: float) -> np.ndarray:
        return np.maximum(tau * (y - t), (tau - 1.0) * (y - t))

    return pin(s) - power * pin(s_dagger)


def run_objective_stability(
    task: dict[str, Any], work_dir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    del work_dir
    params = task["params"]
    tau = float(params.get("tau", 0.9))
    s = float(params.get("s", 0.0))
    s_d = float(params.get("s_dagger", 1.0))
    powers = [float(value) for value in params.get("powers", [0.5, 1.0, 1.25, 1.5])]
    t = np.linspace(-100.0, 100.0, 4001)
    rows = []
    curves = {"t": t}
    for power in powers:
        value = signed_pinball_contribution(t, s, s_d, tau, power)
        left_slope = float(np.polyfit(t[:200], value[:200], 1)[0])
        right_slope = float(np.polyfit(t[-200:], value[-200:], 1)[0])
        rows.append(
            {
                "lambda": power,
                "left_tail_slope": left_slope,
                "right_tail_slope": right_slope,
                "min_on_grid": float(np.min(value)),
                "noncoercive_or_unbounded": bool(power >= 1.0),
            }
        )
        curves[f"objective__{power:g}"] = value
    return {"experiment": "objective_stability", "rows": rows}, curves
