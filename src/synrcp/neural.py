from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class NeuralThresholdFit:
    """Audited legacy-capacity threshold fit used only by the bridge experiment."""

    predict: Callable[[np.ndarray], np.ndarray]
    diagnostics: dict[str, Any]


def fit_neural_threshold(
    x_learning: np.ndarray,
    score_learning: np.ndarray,
    *,
    tau: float,
    seed: int,
    x_pool: np.ndarray | None = None,
    score_pool: np.ndarray | None = None,
    score_paired_synthetic: np.ndarray | None = None,
    objective: str = "supervised",
    power: float = 0.0,
    hidden: int = 128,
    epochs: int = 100,
    learning_rate: float = 2e-3,
) -> NeuralThresholdFit:
    """Fit the original two-hidden-layer threshold network under audited roles.

    ``positive`` adds a positive synthetic empirical-risk term whose total
    coefficient is ``power``. ``signed`` reproduces the legacy PPI objective
    trusted + power * (pool synthetic - paired synthetic).  The signed path is
    diagnostic: divergence is recorded rather than repaired or silently
    discarded.
    """

    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised only without torch
        raise RuntimeError("The neural bridge requires PyTorch.") from error
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work
        # begins. Subsequent bridge fits in the same task retain the first limit.
        pass

    x_l = np.asarray(x_learning, dtype=np.float32)
    s_l = np.asarray(score_learning, dtype=np.float32).reshape(-1, 1)
    if x_l.ndim != 2 or x_l.shape[0] != s_l.shape[0]:
        raise ValueError("Learning covariates and scores have incompatible shapes.")
    if objective not in {"supervised", "positive", "signed"}:
        raise ValueError(f"Unknown neural threshold objective: {objective}")
    if objective != "supervised":
        if x_pool is None or score_pool is None:
            raise ValueError(f"{objective} objective requires a synthetic pool.")
    if objective == "signed" and score_paired_synthetic is None:
        raise ValueError("Signed objective requires paired synthetic scores.")

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = nn.Sequential(
        nn.Linear(x_l.shape[1], int(hidden)),
        nn.ReLU(),
        nn.Linear(int(hidden), int(hidden)),
        nn.ReLU(),
        nn.Linear(int(hidden), 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    tx_l = torch.from_numpy(x_l)
    ts_l = torch.from_numpy(s_l)
    tx_p = torch.from_numpy(np.asarray(x_pool, dtype=np.float32)) if x_pool is not None else None
    ts_p = (
        torch.from_numpy(np.asarray(score_pool, dtype=np.float32).reshape(-1, 1))
        if score_pool is not None
        else None
    )
    ts_d = (
        torch.from_numpy(
            np.asarray(score_paired_synthetic, dtype=np.float32).reshape(-1, 1)
        )
        if score_paired_synthetic is not None
        else None
    )

    def torch_pinball(prediction: Any, target: Any) -> Any:
        residual = target - prediction
        return torch.maximum(float(tau) * residual, (float(tau) - 1.0) * residual)

    loss_curve: list[float] = []
    trusted_curve: list[float] = []
    pool_curve: list[float] = []
    paired_curve: list[float] = []
    divergence_epoch: int | None = None
    model.train()
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction_l = model(tx_l)
        trusted_loss = torch_pinball(prediction_l, ts_l).mean()
        pool_loss = torch.zeros((), dtype=trusted_loss.dtype)
        paired_loss = torch.zeros((), dtype=trusted_loss.dtype)
        loss = trusted_loss
        if objective in {"positive", "signed"}:
            assert tx_p is not None and ts_p is not None
            pool_loss = torch_pinball(model(tx_p), ts_p).mean()
        if objective == "positive":
            loss = trusted_loss + float(power) * pool_loss
        elif objective == "signed":
            assert ts_d is not None
            paired_loss = torch_pinball(prediction_l, ts_d).mean()
            loss = trusted_loss + float(power) * (pool_loss - paired_loss)
        loss_value = float(loss.detach().cpu())
        loss_curve.append(loss_value)
        trusted_curve.append(float(trusted_loss.detach().cpu()))
        pool_curve.append(float(pool_loss.detach().cpu()))
        paired_curve.append(float(paired_loss.detach().cpu()))
        if not np.isfinite(loss_value):
            divergence_epoch = epoch
            break
        loss.backward()
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            divergence_epoch = epoch
            break

    model.eval()
    parameter_norm = float(
        np.sqrt(
            sum(
                float(torch.sum(parameter.detach().double() ** 2).cpu())
                for parameter in model.parameters()
            )
        )
    )

    def predict(array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        with torch.no_grad():
            return model(torch.from_numpy(values)).detach().cpu().numpy().reshape(-1).astype(float)

    train_predictions = predict(x_l)
    pool_predictions = predict(np.asarray(x_pool, dtype=float)) if x_pool is not None else np.array([])
    finite = bool(
        divergence_epoch is None
        and np.isfinite(parameter_norm)
        and np.all(np.isfinite(train_predictions))
        and np.all(np.isfinite(pool_predictions))
    )
    diagnostics = {
        "objective": objective,
        "power": float(power),
        "hidden": int(hidden),
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(loss_curve)),
        "learning_rate": float(learning_rate),
        "loss_curve": loss_curve,
        "trusted_loss_curve": trusted_curve,
        "pool_loss_curve": pool_curve,
        "paired_loss_curve": paired_curve,
        "divergence_epoch": divergence_epoch,
        "finite": finite,
        "parameter_norm": parameter_norm,
        "learning_prediction_min": float(np.min(train_predictions)),
        "learning_prediction_max": float(np.max(train_predictions)),
        "pool_prediction_min": (
            float(np.min(pool_predictions)) if pool_predictions.size else None
        ),
        "pool_prediction_max": (
            float(np.max(pool_predictions)) if pool_predictions.size else None
        ),
    }
    return NeuralThresholdFit(predict=predict, diagnostics=diagnostics)
