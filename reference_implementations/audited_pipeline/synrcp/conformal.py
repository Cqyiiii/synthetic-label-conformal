from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    correction: float
    order_index: int
    calibration_size: int
    tau: float
    infinite: bool


def exact_order_statistic(values: np.ndarray, tau: float) -> CalibrationResult:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Calibration residuals must be nonempty.")
    if not 0 < tau < 1:
        raise ValueError("tau must lie strictly between zero and one.")
    m = int(arr.size)
    k = int(math.ceil((m + 1) * tau))
    if k > m:
        return CalibrationResult(float("inf"), k, m, float(tau), True)
    correction = float(np.partition(arr, k - 1)[k - 1])
    return CalibrationResult(correction, k, m, float(tau), False)


def calibrate_scores(scores: np.ndarray, predicted_quantiles: np.ndarray, tau: float) -> CalibrationResult:
    residuals = np.asarray(scores, dtype=float).reshape(-1) - np.asarray(
        predicted_quantiles, dtype=float
    ).reshape(-1)
    return exact_order_statistic(residuals, tau)


def covered(scores: np.ndarray, predicted_quantiles: np.ndarray, correction: float) -> np.ndarray:
    threshold = np.asarray(predicted_quantiles, dtype=float).reshape(-1) + correction
    return np.asarray(scores, dtype=float).reshape(-1) <= threshold


def radius_summary(predicted_quantiles: np.ndarray, correction: float) -> dict[str, float]:
    radii = np.asarray(predicted_quantiles, dtype=float).reshape(-1) + correction
    finite = np.isfinite(radii)
    positive = finite & (radii >= 0)
    return {
        "mean_radius": float(np.mean(radii[positive])) if positive.any() else float("nan"),
        "median_radius": float(np.median(radii[positive])) if positive.any() else float("nan"),
        "empty_frequency": float(np.mean(radii < 0)),
        "full_frequency": float(np.mean(~finite)),
    }


def box_volume_summary(
    predicted_quantiles: np.ndarray,
    correction: float,
    outcome_scale: np.ndarray,
) -> dict[str, float]:
    """Original-scale volume summary for an L-infinity residual score.

    A nonnegative standardized radius ``r`` yields side lengths
    ``2 r outcome_scale[j]``.  Empty and full sets are recorded explicitly.
    """

    radii = np.asarray(predicted_quantiles, dtype=float).reshape(-1) + correction
    scales = np.asarray(outcome_scale, dtype=float).reshape(-1)
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("outcome_scale must contain finite positive entries.")
    finite_positive = np.isfinite(radii) & (radii > 0)
    log_volume = np.full(radii.shape, np.nan, dtype=float)
    if finite_positive.any():
        log_scale = float(np.sum(np.log(scales)))
        log_volume[finite_positive] = (
            scales.size * np.log(2.0 * radii[finite_positive]) + log_scale
        )
    summary = radius_summary(predicted_quantiles, correction)
    summary.update(
        {
            "mean_log_volume": float(np.nanmean(log_volume))
            if finite_positive.any()
            else float("nan"),
            "median_log_volume": float(np.nanmedian(log_volume))
            if finite_positive.any()
            else float("nan"),
            "outcome_dimension": int(scales.size),
        }
    )
    return summary


def covered_box(
    outcomes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    correction: float,
) -> np.ndarray:
    y = np.asarray(outcomes, dtype=float)
    lo = np.asarray(lower, dtype=float) - float(correction)
    hi = np.asarray(upper, dtype=float) + float(correction)
    if y.ndim == 1:
        y = y[:, None]
    if np.isposinf(correction):
        return np.ones(y.shape[0], dtype=bool)
    return np.all((y >= lo) & (y <= hi), axis=1)


def interval_box_volume_summary(
    lower: np.ndarray,
    upper: np.ndarray,
    correction: float,
    outcome_scale: np.ndarray,
) -> dict[str, float]:
    """Original-scale box volume for asymmetric CQR endpoints."""

    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    scales = np.asarray(outcome_scale, dtype=float).reshape(-1)
    if lo.ndim == 1:
        lo = lo[:, None]
        hi = hi[:, None]
    if lo.shape != hi.shape or lo.shape[1] != scales.size:
        raise ValueError("Endpoint and outcome-scale dimensions are incompatible.")
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("outcome_scale must contain finite positive entries.")
    if np.isposinf(correction):
        return {
            "mean_radius": float("nan"),
            "median_radius": float("nan"),
            "empty_frequency": 0.0,
            "full_frequency": 1.0,
            "mean_log_volume": float("nan"),
            "median_log_volume": float("nan"),
            "outcome_dimension": int(scales.size),
        }
    widths = hi - lo + 2.0 * float(correction)
    nonempty = np.all(widths > 0, axis=1)
    log_volume = np.full(widths.shape[0], np.nan, dtype=float)
    if nonempty.any():
        log_volume[nonempty] = np.sum(np.log(widths[nonempty] * scales[None, :]), axis=1)
    half_width = np.max(widths / 2.0, axis=1)
    return {
        "mean_radius": float(np.mean(half_width[nonempty])) if nonempty.any() else float("nan"),
        "median_radius": float(np.median(half_width[nonempty])) if nonempty.any() else float("nan"),
        "empty_frequency": float(np.mean(~nonempty)),
        "full_frequency": 0.0,
        "mean_log_volume": float(np.nanmean(log_volume)) if nonempty.any() else float("nan"),
        "median_log_volume": float(np.nanmedian(log_volume)) if nonempty.any() else float("nan"),
        "outcome_dimension": int(scales.size),
    }
