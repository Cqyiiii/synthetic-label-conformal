from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class LinearQuantileModel:
    coef_: np.ndarray
    tau: float

    def predict_features(self, phi: np.ndarray) -> np.ndarray:
        return np.asarray(phi, dtype=float) @ self.coef_


@dataclass(frozen=True)
class ConditionalDensityEstimate:
    density: np.ndarray
    lower_quantile: np.ndarray
    upper_quantile: np.ndarray
    lower_level: float
    upper_level: float
    floor_frequency: float
    crossing_frequency: float


def _constant_intercept_column(phi: np.ndarray, tolerance: float = 1e-10) -> bool:
    return bool(phi.shape[1] >= 1 and np.max(np.abs(phi[:, 0] - 1.0)) <= tolerance)


def fit_linear_quantile(
    phi: np.ndarray,
    scores: np.ndarray,
    tau: float,
    alpha: float = 1e-6,
    sample_weight: np.ndarray | None = None,
) -> LinearQuantileModel:
    """Fit a linear quantile head with an explicitly unpenalized intercept.

    All audited feature maps include a leading column of ones.  Scikit-learn's
    ``QuantileRegressor`` does not penalize its native intercept, whereas treating
    the constant column as an ordinary coefficient would penalize it.  We therefore
    remove the leading constant during fitting and reconstruct the full coefficient
    vector afterwards.
    """

    phi = np.asarray(phi, dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if phi.ndim != 2 or phi.shape[0] != scores.size:
        raise ValueError("phi and scores have incompatible shapes.")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights is not None and weights.size != scores.size:
        raise ValueError("sample_weight has incompatible shape.")
    has_intercept = _constant_intercept_column(phi)
    design = phi[:, 1:] if has_intercept else phi
    model = QuantileRegressor(
        quantile=float(tau),
        alpha=float(alpha),
        fit_intercept=has_intercept,
        solver="highs",
    )
    model.fit(design, scores, sample_weight=weights)
    if has_intercept:
        coef = np.concatenate([[float(model.intercept_)], np.asarray(model.coef_, dtype=float)])
    else:
        coef = np.asarray(model.coef_, dtype=float)
    return LinearQuantileModel(coef, float(tau))


def fit_weighted_linear_quantile(
    phi_trusted: np.ndarray,
    score_trusted: np.ndarray,
    phi_synthetic: np.ndarray,
    score_synthetic: np.ndarray,
    tau: float,
    synthetic_total_weight: float = 0.5,
    alpha: float = 1e-6,
) -> LinearQuantileModel:
    """Weighted trusted-plus-synthetic quantile fit.

    ``synthetic_total_weight`` is the total synthetic weight relative to the total
    trusted weight.  This keeps the meaning stable as the synthetic-pool size changes.
    The method is intentionally not bias robust and is reported as a distinct empirical
    strategy rather than as a consequence of the PPI theorem.
    """

    phi_t = np.asarray(phi_trusted, dtype=float)
    phi_s = np.asarray(phi_synthetic, dtype=float)
    y_t = np.asarray(score_trusted, dtype=float).reshape(-1)
    y_s = np.asarray(score_synthetic, dtype=float).reshape(-1)
    if phi_t.shape[1] != phi_s.shape[1]:
        raise ValueError("Trusted and synthetic feature dimensions differ.")
    if synthetic_total_weight < 0:
        raise ValueError("synthetic_total_weight must be nonnegative.")
    trusted_weight = np.ones(y_t.size, dtype=float)
    per_synthetic = (float(synthetic_total_weight) * max(y_t.size, 1) / max(y_s.size, 1))
    synthetic_weight = np.full(y_s.size, per_synthetic, dtype=float)
    return fit_linear_quantile(
        np.vstack([phi_t, phi_s]),
        np.concatenate([y_t, y_s]),
        tau,
        alpha=alpha,
        sample_weight=np.concatenate([trusted_weight, synthetic_weight]),
    )


def pinball_loss(scores: np.ndarray, predictions: np.ndarray, tau: float) -> np.ndarray:
    residual = np.asarray(scores, dtype=float).reshape(-1) - np.asarray(predictions, dtype=float).reshape(-1)
    return np.maximum(float(tau) * residual, (float(tau) - 1.0) * residual)


def gaussian_kernel_density_at_boundary(residuals: np.ndarray, minimum: float = 1e-3) -> float:
    """Scalar fallback retained only for diagnostics and backwards compatibility."""

    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    n = max(residuals.size, 2)
    scale = np.std(residuals, ddof=1) if residuals.size > 1 else 1.0
    iqr = np.subtract(*np.percentile(residuals, [75, 25])) if residuals.size > 3 else scale
    robust = min(scale, iqr / 1.349) if iqr > 0 else scale
    bandwidth = max(0.9 * max(robust, 1e-8) * n ** (-1 / 5), 1e-3)
    density = np.mean(np.exp(-0.5 * (residuals / bandwidth) ** 2)) / (
        np.sqrt(2 * np.pi) * bandwidth
    )
    return float(max(density, minimum))


def estimate_conditional_boundary_density(
    phi_train: np.ndarray,
    scores_train: np.ndarray,
    tau: float,
    phi_eval: np.ndarray | None = None,
    delta_tau: float = 0.04,
    alpha: float = 1e-5,
    minimum: float = 1e-3,
    maximum: float = 100.0,
) -> ConditionalDensityEstimate:
    """Estimate f_{S|X}(q_tau(X)|X) from nearby conditional quantile heads.

    The reciprocal-spacing estimate is ``(u-l)/(Q_u-Q_l)``.  Negative or tiny
    spacings are floored and explicitly counted rather than silently replacing
    the vector with a global scalar density.
    """

    tau = float(tau)
    max_delta = 0.49 * min(tau, 1.0 - tau)
    delta = min(float(delta_tau), max_delta)
    if delta <= 0:
        raise ValueError("delta_tau is incompatible with tau.")
    lower_level = tau - delta
    upper_level = tau + delta
    phi_train = np.asarray(phi_train, dtype=float)
    phi_eval = phi_train if phi_eval is None else np.asarray(phi_eval, dtype=float)
    low_model = fit_linear_quantile(phi_train, scores_train, lower_level, alpha=alpha)
    high_model = fit_linear_quantile(phi_train, scores_train, upper_level, alpha=alpha)
    q_low = low_model.predict_features(phi_eval)
    q_high = high_model.predict_features(phi_eval)
    spacing = q_high - q_low
    crossing = spacing <= 0
    # Separate quantile heads can cross in the scarce-label regime.  We keep the
    # crossing rate as an audit diagnostic, but avoid replacing every crossing
    # by the maximum possible density.  The magnitude of the local spacing is
    # used when informative, with a robust unconditional/median fallback for
    # near-zero spacings.  This remains a projected-density surrogate rather
    # than a claim of noncrossing conditional-density consistency.
    level_width = upper_level - lower_level
    unconditional_spacing = float(
        np.quantile(np.asarray(scores_train, dtype=float), upper_level)
        - np.quantile(np.asarray(scores_train, dtype=float), lower_level)
    )
    absolute_spacing = np.abs(spacing)
    positive_magnitudes = absolute_spacing[absolute_spacing > level_width / maximum]
    median_spacing = (
        float(np.median(positive_magnitudes))
        if positive_magnitudes.size
        else 0.0
    )
    fallback_spacing = max(
        unconditional_spacing,
        median_spacing,
        level_width / maximum,
    )
    repaired_spacing = np.where(
        spacing > level_width / maximum,
        spacing,
        np.maximum(absolute_spacing, fallback_spacing),
    )
    safe_spacing = np.maximum(repaired_spacing, level_width / maximum)
    density = level_width / safe_spacing
    floor = density <= minimum
    density = np.clip(density, minimum, maximum)
    return ConditionalDensityEstimate(
        density=np.asarray(density, dtype=float),
        lower_quantile=np.asarray(q_low, dtype=float),
        upper_quantile=np.asarray(q_high, dtype=float),
        lower_level=float(lower_level),
        upper_level=float(upper_level),
        floor_frequency=float(np.mean(floor)),
        crossing_frequency=float(np.mean(crossing)),
    )


def estimate_geometry(
    phi: np.ndarray,
    f_boundary: np.ndarray | float,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional design matrix.")
    n, p = phi.shape
    raw_f = np.asarray(f_boundary, dtype=float).reshape(-1)
    if raw_f.size == 1:
        f = np.full(n, float(raw_f[0]), dtype=float)
    elif raw_f.size == n:
        f = raw_f
    else:
        raise ValueError("f_boundary must be scalar or have one value per row.")
    f = np.maximum(f, 1e-12)
    bar_f = float(np.mean(f))
    h = np.mean(f[:, None] * phi, axis=0)
    d_c = phi - h[None, :] / max(bar_f, 1e-12)
    penalty = np.eye(p)
    if _constant_intercept_column(phi):
        penalty[0, 0] = 0.0
    hessian = (phi.T * f) @ phi / n + float(ridge) * penalty
    a_c = (d_c.T * (f**2)) @ d_c / n
    h_inv = np.linalg.pinv(hessian, rcond=1e-10)
    g_c = h_inv @ a_c @ h_inv
    g_c = (g_c + g_c.T) / 2
    eigvals, eigvecs = np.linalg.eigh(g_c)
    eigvals = np.maximum(eigvals, 0.0)
    g_c = (eigvecs * eigvals) @ eigvecs.T
    return hessian, g_c, d_c


def boundary_score(phi: np.ndarray, score: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    indicator = (np.asarray(score).reshape(-1) <= np.asarray(q).reshape(-1)).astype(float)
    return (indicator - float(tau))[:, None] * np.asarray(phi, dtype=float)


def _cov_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xc = x - np.mean(x, axis=0, keepdims=True)
    if y is None:
        return xc.T @ xc / max(x.shape[0] - 1, 1)
    y = np.asarray(y, dtype=float)
    yc = y - np.mean(y, axis=0, keepdims=True)
    return xc.T @ yc / max(x.shape[0] - 1, 1)


def projected_constants(
    psi: np.ndarray,
    psi_dagger: np.ndarray,
    g_c: np.ndarray,
    psi_pool: np.ndarray | None = None,
) -> dict[str, float]:
    cov = _cov_matrix(psi, psi_dagger)
    a = float(np.trace(g_c @ _cov_matrix(psi)))
    b = float(np.trace(g_c @ cov))
    c = float(np.trace(g_c @ _cov_matrix(psi_dagger)))
    c_pool = c if psi_pool is None else float(np.trace(g_c @ _cov_matrix(psi_pool)))
    rho = b / np.sqrt(max(a * c, 1e-24))
    return {
        "a": a,
        "b": b,
        "c": c,
        "c_pool": c_pool,
        "rho": float(rho),
        "raw_covariance_trace": float(np.trace(cov)),
        "raw_covariance_norm": float(np.linalg.norm(cov, ord="fro")),
    }


def projected_cross_trace(left: np.ndarray, right: np.ndarray, g_c: np.ndarray) -> float:
    return float(np.trace(np.asarray(g_c, dtype=float) @ _cov_matrix(left, right)))


def one_step_update(
    beta: np.ndarray,
    hessian: np.ndarray,
    psi: np.ndarray,
    psi_dagger: np.ndarray,
    psi_pool: np.ndarray,
    power: float,
    max_update_norm: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    score = np.mean(psi - float(power) * psi_dagger, axis=0) + float(power) * np.mean(
        psi_pool, axis=0
    )
    update = -np.linalg.pinv(hessian, rcond=1e-10) @ score
    norm = float(np.linalg.norm(update))
    clipped = False
    if norm > float(max_update_norm):
        update *= float(max_update_norm) / norm
        clipped = True
    return np.asarray(beta, dtype=float) + update, {
        "update_norm": norm,
        "update_clipped": float(clipped),
        "hessian_condition": float(np.linalg.cond(hessian)),
    }


@dataclass(frozen=True)
class PowerEstimate:
    b_hat: float
    c_hat: float
    c_paired_hat: float
    c_pool_hat: float
    a_hat: float
    se_b: float
    denominator_hat: float
    gate_threshold: float
    raw_power: float
    selected_power: float
    sign_passed: bool
    density_floor_frequency: float
    density_crossing_frequency: float


def cross_fitted_power(
    x: np.ndarray,
    score: np.ndarray,
    score_dagger: np.ndarray,
    x_pool: np.ndarray,
    score_pool: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    tau: float,
    n_splits: int = 2,
    ridge: float = 1e-4,
    gate_rate: float = 1.0,
    lambda_max: float = 1.5,
    allow_signed: bool = False,
    seed: int = 0,
    density_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    density_delta_tau: float = 0.04,
    quantile_alpha: float = 1e-5,
) -> PowerEstimate:
    """Cross-fitted projected power with the Theory V5 deterministic gate.

    ``se_b`` is reported only as a diagnostic.  Selection uses

        n^{-1/4} sqrt(a_hat c_hat)

    rather than an unsupported row-level normal confidence bound.
    """

    x = np.asarray(x)
    score = np.asarray(score).reshape(-1)
    score_dagger = np.asarray(score_dagger).reshape(-1)
    x_pool = np.asarray(x_pool)
    score_pool = np.asarray(score_pool).reshape(-1)
    if len(x) < max(2 * n_splits, 6):
        raise ValueError("Too few paired observations for cross-fitted power estimation.")
    contributions: list[float] = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    c_pair_vals: list[float] = []
    c_pool_vals: list[float] = []
    density_floor: list[float] = []
    density_crossing: list[float] = []
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    for train, valid in kf.split(x):
        phi_train = feature_fn(x[train])
        model = fit_linear_quantile(phi_train, score[train], tau, alpha=quantile_alpha)
        q_train = model.predict_features(phi_train)
        phi_valid = feature_fn(x[valid])
        q_valid = model.predict_features(phi_valid)
        phi_pool = feature_fn(x_pool)
        q_pool = model.predict_features(phi_pool)
        if density_fn is None:
            density = estimate_conditional_boundary_density(
                phi_train,
                score[train],
                tau=tau,
                phi_eval=phi_train,
                delta_tau=density_delta_tau,
                alpha=quantile_alpha,
            )
            f_train = density.density
            density_floor.append(density.floor_frequency)
            density_crossing.append(density.crossing_frequency)
        else:
            f_train = np.asarray(density_fn(q_train, x[train]), dtype=float)
            density_floor.append(float(np.mean(f_train <= 1.001e-3)))
            density_crossing.append(0.0)
        _, g_c, _ = estimate_geometry(phi_train, f_train, ridge=ridge)
        psi = boundary_score(phi_valid, score[valid], q_valid, tau)
        psi_d = boundary_score(phi_valid, score_dagger[valid], q_valid, tau)
        # The pool score is recomputed from this fold-specific pilot.
        psi_p = boundary_score(phi_pool, score_pool, q_pool, tau)
        constants = projected_constants(psi, psi_d, g_c, psi_p)
        a_vals.append(constants["a"])
        b_vals.append(constants["b"])
        c_pair_vals.append(constants["c"])
        c_pool_vals.append(constants["c_pool"])
        pc = psi - np.mean(psi, axis=0, keepdims=True)
        pdc = psi_d - np.mean(psi_d, axis=0, keepdims=True)
        contributions.extend(np.einsum("ij,jk,ik->i", pc, g_c, pdc).tolist())
    a_hat = float(np.mean(a_vals))
    b_hat = float(np.mean(b_vals))
    c_paired_hat = float(np.mean(c_pair_vals))
    c_pool_hat = float(np.mean(c_pool_vals))
    kappa = len(x) / max(len(x_pool), 1)
    denominator = max(c_paired_hat + kappa * c_pool_hat, 1e-12)
    raw = b_hat / denominator
    gate_threshold = float(gate_rate) * len(x) ** (-0.25) * np.sqrt(
        max(a_hat * c_paired_hat, 0.0)
    )
    if allow_signed:
        shrunk_b = np.sign(b_hat) * max(abs(b_hat) - gate_threshold, 0.0)
        passed = abs(b_hat) > gate_threshold
        selected = float(np.clip(shrunk_b / denominator, -lambda_max, lambda_max))
    else:
        shrunk_b = max(b_hat - gate_threshold, 0.0)
        passed = b_hat > gate_threshold
        selected = float(np.clip(shrunk_b / denominator, 0.0, lambda_max))
    contrib = np.asarray(contributions, dtype=float)
    se_b = (
        float(np.std(contrib, ddof=1) / np.sqrt(contrib.size))
        if contrib.size > 1
        else float("inf")
    )
    return PowerEstimate(
        b_hat=b_hat,
        c_hat=c_paired_hat,
        c_paired_hat=c_paired_hat,
        c_pool_hat=c_pool_hat,
        a_hat=a_hat,
        se_b=se_b,
        denominator_hat=float(denominator),
        gate_threshold=float(gate_threshold),
        raw_power=float(raw),
        selected_power=selected,
        sign_passed=bool(passed),
        density_floor_frequency=float(np.mean(density_floor)),
        density_crossing_frequency=float(np.mean(density_crossing)),
    )
