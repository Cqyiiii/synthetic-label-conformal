from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrozenRepresentation:
    """Small supervised random-feature map fitted on an external role.

    A larger bank of random ReLU features is generated from covariates, then
    the features most correlated with the external-role score are retained.
    Downstream paired learning only refits fixed-dimensional quantile heads.
    This avoids both representation leakage and a costly nested neural fit.
    """

    weights: np.ndarray
    bias: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    tau: float
    training_summary: dict[str, Any]

    @property
    def input_dim(self) -> int:
        return int(self.weights.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.weights.shape[1])

    def features(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        z = (array - self.x_mean) / self.x_scale
        hidden = np.maximum(z @ self.weights + self.bias, 0.0)
        return np.column_stack([np.ones(hidden.shape[0]), hidden])


def _pinball(prediction: np.ndarray, target: np.ndarray, tau: float) -> float:
    residual = np.asarray(target, dtype=float).reshape(-1) - np.asarray(prediction, dtype=float).reshape(-1)
    return float(np.mean(np.maximum(tau * residual, (tau - 1.0) * residual)))


def train_quantile_representation(
    x: np.ndarray,
    score: np.ndarray,
    tau: float,
    hidden: int = 32,
    feature_dim: int = 12,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> FrozenRepresentation:
    """Fit and freeze a cheap external-role representation.

    ``epochs``, ``learning_rate`` and ``weight_decay`` are retained in the API
    for configuration compatibility.  Theory V5 intentionally replaces the
    former iterative neural representation with a deterministic-size random
    feature screening step, reducing the real-data compute budget sharply.
    """

    del learning_rate, weight_decay
    array = np.asarray(x, dtype=float)
    target = np.asarray(score, dtype=float).reshape(-1)
    if array.ndim != 2 or array.shape[0] != target.size:
        raise ValueError("x and score have incompatible shapes.")
    x_mean = np.mean(array, axis=0)
    x_scale = np.std(array, axis=0, ddof=0)
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
    z = (array - x_mean) / x_scale
    rng = np.random.default_rng(seed)
    bank_size = max(int(hidden), int(feature_dim) * 4, int(feature_dim))
    weights_bank = rng.normal(scale=1.0 / np.sqrt(max(array.shape[1], 1)), size=(array.shape[1], bank_size))
    bias_bank = rng.normal(scale=0.5, size=bank_size)
    bank = np.maximum(z @ weights_bank + bias_bank, 0.0)
    centered_bank = bank - np.mean(bank, axis=0, keepdims=True)
    centered_target = target - np.mean(target)
    denominator = np.sqrt(np.sum(centered_bank**2, axis=0) * np.sum(centered_target**2))
    correlations = np.divide(
        centered_bank.T @ centered_target,
        denominator,
        out=np.zeros(bank_size, dtype=float),
        where=denominator > 1e-12,
    )
    selected = np.argsort(np.abs(correlations))[-int(feature_dim) :]
    selected = np.sort(selected)
    weights = weights_bank[:, selected]
    bias = bias_bank[selected]
    selected_features = bank[:, selected]
    design = np.column_stack([np.ones(array.shape[0]), selected_features])
    # A least-squares head is used only for an inexpensive training diagnostic.
    head = np.linalg.pinv(design, rcond=1e-8) @ target
    baseline = np.full_like(target, np.quantile(target, tau))
    fitted = design @ head
    return FrozenRepresentation(
        weights=np.asarray(weights, dtype=float),
        bias=np.asarray(bias, dtype=float),
        x_mean=np.asarray(x_mean, dtype=float),
        x_scale=np.asarray(x_scale, dtype=float),
        tau=float(tau),
        training_summary={
            "representation_type": "screened_random_relu",
            "requested_epochs_ignored": int(epochs),
            "bank_size": int(bank_size),
            "feature_dim": int(feature_dim),
            "initial_pinball": _pinball(baseline, target, tau),
            "screened_head_pinball": _pinball(fitted, target, tau),
            "max_absolute_screening_correlation": float(np.max(np.abs(correlations))),
            "parameter_count": int(weights.size + bias.size),
        },
    )



def train_covariate_representation(
    x: np.ndarray,
    tau: float,
    hidden: int = 32,
    feature_dim: int = 12,
    seed: int = 0,
) -> FrozenRepresentation:
    """Fit a label-free frozen random-feature map.

    A random ReLU bank is generated from the dedicated covariate-only role.  Features
    are selected by empirical variance, so no outcome or nonconformity score is read.
    This is the default representation in the V6 low-label benchmark; the supervised
    screening route is retained for exact reproduction of the completed V5 wave.
    """

    array = np.asarray(x, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("x must be a nontrivial two-dimensional array.")
    x_mean = np.mean(array, axis=0)
    x_scale = np.std(array, axis=0, ddof=0)
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
    z = (array - x_mean) / x_scale
    rng = np.random.default_rng(seed)
    bank_size = max(int(hidden), int(feature_dim) * 4, int(feature_dim))
    weights_bank = rng.normal(
        scale=1.0 / np.sqrt(max(array.shape[1], 1)),
        size=(array.shape[1], bank_size),
    )
    bias_bank = rng.normal(scale=0.5, size=bank_size)
    bank = np.maximum(z @ weights_bank + bias_bank, 0.0)
    variances = np.var(bank, axis=0)
    selected = np.argsort(variances)[-int(feature_dim) :]
    selected = np.sort(selected)
    return FrozenRepresentation(
        weights=np.asarray(weights_bank[:, selected], dtype=float),
        bias=np.asarray(bias_bank[selected], dtype=float),
        x_mean=np.asarray(x_mean, dtype=float),
        x_scale=np.asarray(x_scale, dtype=float),
        tau=float(tau),
        training_summary={
            "representation_type": "covariate_only_random_relu",
            "outcomes_accessed": 0,
            "bank_size": int(bank_size),
            "feature_dim": int(feature_dim),
            "minimum_selected_variance": float(np.min(variances[selected])),
            "maximum_selected_variance": float(np.max(variances[selected])),
            "parameter_count": int(weights_bank[:, selected].size + selected.size),
        },
    )
