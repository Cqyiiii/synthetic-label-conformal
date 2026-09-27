from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln
from sklearn.neighbors import NearestNeighbors

from .conformal import CalibrationResult, exact_order_statistic
from .models import (
    boundary_score,
    estimate_conditional_boundary_density,
    estimate_geometry,
    fit_linear_quantile,
    one_step_update,
)


def _log_comb(n: np.ndarray | float, k: np.ndarray | float) -> np.ndarray:
    n_arr = np.asarray(n, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    valid = (n_arr >= 0) & (k_arr >= 0) & (n_arr >= k_arr)
    out = np.full(np.broadcast(n_arr, k_arr).shape, -np.inf, dtype=float)
    n_b, k_b = np.broadcast_arrays(n_arr, k_arr)
    out[valid] = gammaln(n_b[valid] + 1.0) - gammaln(k_b[valid] + 1.0) - gammaln(
        n_b[valid] - k_b[valid] + 1.0
    )
    return out


def compute_spi_windows(m: int, n_synthetic: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Numerically stable rank windows used by the scalar SPI comparator.

    This implementation is adapted from the project's original experimental code.
    It is kept in a separate comparator module because SPI changes the calibration
    rule, whereas the paper's main theory analyzes an untouched final RCP calibration.
    """

    m = int(m)
    n_synthetic = int(n_synthetic)
    beta = float(beta)
    if m < 1 or n_synthetic < 1:
        raise ValueError("SPI requires nonempty trusted and synthetic score samples.")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1).")
    r_minus = np.zeros(m + 1, dtype=int)
    r_plus = np.zeros(m + 1, dtype=int)
    log_denom = float(_log_comb(n_synthetic + m + 1, m + 1))
    ks = np.arange(1, n_synthetic + 2, dtype=float)
    for r in range(1, m + 2):
        n1 = ks + r - 2
        k1 = float(r - 1)
        n2 = n_synthetic + m - ks - r + 2
        k2 = float(m - r + 1)
        log_pmf = _log_comb(n1, k1) + _log_comb(n2, k2) - log_denom
        max_log = float(np.max(log_pmf))
        if not np.isfinite(max_log):
            raise RuntimeError("Degenerate SPI rank distribution.")
        pmf = np.exp(log_pmf - max_log)
        pmf /= np.sum(pmf)
        cdf = np.cumsum(pmf)
        valid_minus = np.flatnonzero(cdf <= beta / 2.0)
        r_min = 1 if valid_minus.size == 0 else int(ks[valid_minus[-1]] + 1)
        valid_plus = np.flatnonzero(cdf >= 1.0 - beta / 2.0)
        r_max = n_synthetic + 1 if valid_plus.size == 0 else int(ks[valid_plus[0]])
        r_minus[r - 1] = r_min
        r_plus[r - 1] = r_max
    return r_minus, r_plus


def spi_score_threshold(
    trusted_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    tau: float,
    beta: float = 0.4,
) -> tuple[float, dict[str, Any]]:
    """Return the scalar SPI score threshold for a regression-score comparator."""

    trusted = np.sort(np.asarray(trusted_scores, dtype=float).reshape(-1))
    synthetic = np.sort(np.asarray(synthetic_scores, dtype=float).reshape(-1))
    m, n_synthetic = trusted.size, synthetic.size
    r_minus, r_plus = compute_spi_windows(m, n_synthetic, beta)
    cutoff = int(np.ceil(float(tau) * (n_synthetic + 1)))
    q_prime_index = min(max(cutoff, 1), n_synthetic) - 1
    q_prime = float(synthetic[q_prime_index])
    minus_candidates = np.flatnonzero(r_minus <= cutoff)
    plus_candidates = np.flatnonzero(r_plus <= cutoff)
    r_bar_minus = int(minus_candidates[-1]) if minus_candidates.size else 0
    r_bar_plus = int(plus_candidates[-1]) if plus_candidates.size else 0

    def trusted_order(index_zero_based: int) -> float:
        return float(trusted[index_zero_based]) if index_zero_based < m else float("inf")

    threshold = max(min(q_prime, trusted_order(r_bar_minus)), trusted_order(r_bar_plus))
    return float(threshold), {
        "beta": float(beta),
        "trusted_scores": int(m),
        "synthetic_scores": int(n_synthetic),
        "synthetic_order_index": int(q_prime_index + 1),
        "trusted_lower_index": int(r_bar_minus + 1),
        "trusted_upper_index": int(r_bar_plus + 1),
    }


def semicp_nnm_threshold(
    paired_true_scores: np.ndarray,
    paired_synthetic_scores: np.ndarray,
    trusted_calibration_scores: np.ndarray,
    synthetic_pool_scores: np.ndarray,
    tau: float,
) -> tuple[float, dict[str, Any]]:
    """SemiCP-style nearest-neighbor correction of synthetic proxy scores.

    The mapping is fitted on a dedicated paired role and the quantile is computed from
    an independent trusted calibration role plus corrected synthetic scores.  This is
    an empirical comparator; unlike the main RCP methods, it is not claimed to be robust
    to arbitrary teacher error.
    """

    true_pair = np.asarray(paired_true_scores, dtype=float).reshape(-1)
    syn_pair = np.asarray(paired_synthetic_scores, dtype=float).reshape(-1)
    trusted_cal = np.asarray(trusted_calibration_scores, dtype=float).reshape(-1)
    syn_pool = np.asarray(synthetic_pool_scores, dtype=float).reshape(-1)
    if true_pair.size != syn_pair.size or true_pair.size < 2:
        raise ValueError("SemiCP-NNM requires matched paired score arrays.")
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(syn_pair[:, None])
    distances, indices = nn.kneighbors(syn_pool[:, None])
    matched = indices[:, 0]
    corrected = syn_pool + true_pair[matched] - syn_pair[matched]
    combined = np.concatenate([trusted_cal, corrected])
    calibration = exact_order_statistic(combined, tau)
    return float(calibration.correction), {
        "paired_scores": int(true_pair.size),
        "trusted_calibration_scores": int(trusted_cal.size),
        "synthetic_scores": int(syn_pool.size),
        "mean_match_distance": float(np.mean(distances)),
        "mean_proxy_correction": float(np.mean(corrected - syn_pool)),
        "calibration_k": int(calibration.order_index),
        "calibration_infinite": bool(calibration.infinite),
    }


@dataclass(frozen=True)
class LinearCQRModel:
    lower_coef: np.ndarray
    upper_coef: np.ndarray
    lower_level: float
    upper_level: float
    metadata: dict[str, Any]

    def predict_features(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = np.asarray(phi, dtype=float)
        lower = design @ np.asarray(self.lower_coef, dtype=float).T
        upper = design @ np.asarray(self.upper_coef, dtype=float).T
        return np.minimum(lower, upper), np.maximum(lower, upper)


def fit_cqr_supervised(
    phi: np.ndarray,
    outcomes: np.ndarray,
    alpha: float,
    quantile_alpha: float = 1e-5,
) -> LinearCQRModel:
    design = np.asarray(phi, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    lower_level = float(alpha) / 2.0
    upper_level = 1.0 - float(alpha) / 2.0
    lower = []
    upper = []
    for column in range(y.shape[1]):
        lower.append(
            fit_linear_quantile(design, y[:, column], lower_level, alpha=quantile_alpha).coef_
        )
        upper.append(
            fit_linear_quantile(design, y[:, column], upper_level, alpha=quantile_alpha).coef_
        )
    return LinearCQRModel(
        lower_coef=np.asarray(lower, dtype=float),
        upper_coef=np.asarray(upper, dtype=float),
        lower_level=lower_level,
        upper_level=upper_level,
        metadata={"method": "supervised", "output_dimension": int(y.shape[1])},
    )


def fit_cqr_ppi_one_step(
    phi: np.ndarray,
    outcomes: np.ndarray,
    synthetic_outcomes: np.ndarray,
    phi_pool: np.ndarray,
    synthetic_pool_outcomes: np.ndarray,
    alpha: float,
    quantile_alpha: float = 1e-5,
    density_delta_tau: float = 0.02,
    ridge: float = 1e-3,
    max_update_norm: float = 5.0,
) -> LinearCQRModel:
    design = np.asarray(phi, dtype=float)
    design_pool = np.asarray(phi_pool, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    y_d = np.asarray(synthetic_outcomes, dtype=float)
    y_pool = np.asarray(synthetic_pool_outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
        y_d = y_d[:, None]
        y_pool = y_pool[:, None]
    if y.shape != y_d.shape or y_pool.shape[1] != y.shape[1]:
        raise ValueError("CQR trusted and synthetic outcomes have incompatible shapes.")
    levels = (float(alpha) / 2.0, 1.0 - float(alpha) / 2.0)
    coefficients: list[list[np.ndarray]] = [[], []]
    update_norms: list[float] = []
    clipped: list[float] = []
    conditions: list[float] = []
    floors: list[float] = []
    crossings: list[float] = []
    for level_index, level in enumerate(levels):
        for column in range(y.shape[1]):
            pilot = fit_linear_quantile(design, y[:, column], level, alpha=quantile_alpha)
            q = pilot.predict_features(design)
            q_pool = pilot.predict_features(design_pool)
            density = estimate_conditional_boundary_density(
                design,
                y[:, column],
                tau=level,
                phi_eval=design,
                delta_tau=density_delta_tau,
                alpha=quantile_alpha,
            )
            hessian, _, _ = estimate_geometry(design, density.density, ridge=ridge)
            psi = boundary_score(design, y[:, column], q, level)
            psi_d = boundary_score(design, y_d[:, column], q, level)
            psi_pool = boundary_score(design_pool, y_pool[:, column], q_pool, level)
            coef, meta = one_step_update(
                pilot.coef_,
                hessian,
                psi,
                psi_d,
                psi_pool,
                power=1.0,
                max_update_norm=max_update_norm,
            )
            coefficients[level_index].append(coef)
            update_norms.append(float(meta["update_norm"]))
            clipped.append(float(meta["update_clipped"]))
            conditions.append(float(meta["hessian_condition"]))
            floors.append(float(density.floor_frequency))
            crossings.append(float(density.crossing_frequency))
    return LinearCQRModel(
        lower_coef=np.asarray(coefficients[0], dtype=float),
        upper_coef=np.asarray(coefficients[1], dtype=float),
        lower_level=levels[0],
        upper_level=levels[1],
        metadata={
            "method": "ppi_unit_one_step",
            "output_dimension": int(y.shape[1]),
            "maximum_update_norm": float(np.max(update_norms)) if update_norms else 0.0,
            "update_clipped_count": int(np.sum(clipped)),
            "maximum_hessian_condition": float(np.max(conditions)) if conditions else float("nan"),
            "density_floor_frequency": float(np.mean(floors)) if floors else float("nan"),
            "density_crossing_frequency": float(np.mean(crossings)) if crossings else float("nan"),
        },
    )


def cqr_calibration_score(
    outcomes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    y = np.asarray(outcomes, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    return np.max(np.maximum(lo - y, y - hi), axis=1)


def calibrate_cqr(
    outcomes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tau: float,
) -> CalibrationResult:
    return exact_order_statistic(cqr_calibration_score(outcomes, lower, upper), tau)
