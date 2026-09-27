from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class V11NeuralFit:
    predict: Callable[[np.ndarray], np.ndarray]
    diagnostics: dict[str, Any]


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("V11 neural experiments require PyTorch.") from error
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch, nn


def _prepare_torch(seed: int) -> tuple[Any, Any]:
    torch, nn = _torch_modules()
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch, nn


def _network(nn: Any, input_dim: int, output_dim: int, hidden: int) -> Any:
    return nn.Sequential(
        nn.Linear(int(input_dim), int(hidden)),
        nn.ReLU(),
        nn.Linear(int(hidden), int(hidden)),
        nn.ReLU(),
        nn.Linear(int(hidden), int(output_dim)),
    )


def _parameter_norm(torch: Any, model: Any) -> float:
    return float(
        np.sqrt(
            sum(
                float(torch.sum(parameter.detach().double() ** 2).cpu())
                for parameter in model.parameters()
            )
        )
    )


def _make_predictor(
    torch: Any,
    model: Any,
    transform: Callable[[Any], Any] | None = None,
    chunk_size: int = 8192,
) -> Callable[[np.ndarray], np.ndarray]:
    model.eval()

    def predict(array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), int(chunk_size)):
                result = model(torch.from_numpy(values[start : start + int(chunk_size)]))
                if transform is not None:
                    result = transform(result)
                outputs.append(result.detach().cpu().numpy())
        if not outputs:
            return np.empty((0, 1), dtype=float)
        return np.concatenate(outputs, axis=0).astype(float)

    return predict


def fit_mean_network(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    hidden: int = 128,
    epochs: int = 100,
    learning_rate: float = 2e-3,
) -> V11NeuralFit:
    torch, nn = _prepare_torch(seed)
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    model = _network(nn, x_arr.shape[1], y_arr.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    tx = torch.from_numpy(x_arr)
    ty = torch.from_numpy(y_arr)
    loss_fn = nn.HuberLoss(delta=1.0)
    curve: list[float] = []
    divergence_epoch: int | None = None
    model.train()
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(tx), ty)
        value = float(loss.detach().cpu())
        if epoch % 5 == 0 or epoch == int(epochs) - 1:
            curve.append(value)
        if not np.isfinite(value):
            divergence_epoch = epoch
            break
        loss.backward()
        optimizer.step()
    predict = _make_predictor(torch, model)
    train_prediction = predict(x_arr)
    diagnostics = {
        "objective": "huber_mean",
        "seed": int(seed),
        "hidden": int(hidden),
        "epochs_requested": int(epochs),
        "epochs_completed": int(epoch + 1),
        "learning_rate": float(learning_rate),
        "loss_curve_every_5_epochs": curve,
        "divergence_epoch": divergence_epoch,
        "finite": bool(
            divergence_epoch is None
            and np.all(np.isfinite(train_prediction))
            and np.isfinite(_parameter_norm(torch, model))
        ),
        "parameter_norm": _parameter_norm(torch, model),
        "train_rmse": float(np.sqrt(np.mean((train_prediction - y_arr) ** 2))),
    }
    return V11NeuralFit(predict=predict, diagnostics=diagnostics)


def fit_threshold_network(
    x_trusted: np.ndarray,
    score_trusted: np.ndarray,
    *,
    tau: float,
    seed: int,
    objective: str = "supervised",
    power: float = 0.0,
    x_pool: np.ndarray | None = None,
    score_pool: np.ndarray | None = None,
    score_paired: np.ndarray | None = None,
    hidden: int = 128,
    epochs: int = 100,
    learning_rate: float = 2e-3,
    output_bound: float | None = None,
    pretrain_epochs: int | None = None,
) -> V11NeuralFit:
    """Fit one V11 RCP threshold learner from a shared initialization.

    Objectives:
    - ``supervised``: trusted pinball loss.
    - ``positive``: trusted + power * synthetic-pool pinball loss.
    - ``signed``: trusted + power * (pool synthetic - paired synthetic).
    - ``pool``: equal per-observation pooling of trusted and synthetic samples.
    - ``pretrain_finetune``: synthetic pretraining followed by trusted fine-tuning.
    """

    if objective not in {
        "supervised",
        "positive",
        "signed",
        "pool",
        "pretrain_finetune",
    }:
        raise ValueError(f"Unknown threshold objective: {objective}")
    if objective != "supervised" and (x_pool is None or score_pool is None):
        raise ValueError(f"{objective} requires a pool.")
    if objective == "signed" and score_paired is None:
        raise ValueError("signed objective requires paired synthetic scores.")

    torch, nn = _prepare_torch(seed)
    x_t = np.asarray(x_trusted, dtype=np.float32)
    s_t = np.asarray(score_trusted, dtype=np.float32).reshape(-1, 1)
    x_p = None if x_pool is None else np.asarray(x_pool, dtype=np.float32)
    s_p = (
        None
        if score_pool is None
        else np.asarray(score_pool, dtype=np.float32).reshape(-1, 1)
    )
    s_d = (
        None
        if score_paired is None
        else np.asarray(score_paired, dtype=np.float32).reshape(-1, 1)
    )
    model = _network(nn, x_t.shape[1], 1, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    tx_t = torch.from_numpy(x_t)
    ts_t = torch.from_numpy(s_t)
    tx_p = None if x_p is None else torch.from_numpy(x_p)
    ts_p = None if s_p is None else torch.from_numpy(s_p)
    ts_d = None if s_d is None else torch.from_numpy(s_d)
    bound = None if output_bound is None else float(max(output_bound, 1e-6))

    def transform(raw: Any) -> Any:
        if bound is None:
            return raw
        return bound * torch.sigmoid(raw)

    def pinball(prediction: Any, target: Any) -> Any:
        residual = target - prediction
        return torch.maximum(float(tau) * residual, (float(tau) - 1.0) * residual)

    curve: list[dict[str, float]] = []
    divergence_epoch: int | None = None
    model.train()
    total_epochs = int(epochs)
    pre_epochs = (
        int(pretrain_epochs)
        if pretrain_epochs is not None
        else max(1, int(epochs) // 2)
    )
    for epoch in range(total_epochs):
        optimizer.zero_grad(set_to_none=True)
        pred_t = transform(model(tx_t))
        trusted_loss = pinball(pred_t, ts_t).mean()
        pool_loss = torch.zeros((), dtype=trusted_loss.dtype)
        paired_loss = torch.zeros((), dtype=trusted_loss.dtype)
        if tx_p is not None and ts_p is not None:
            pool_loss = pinball(transform(model(tx_p)), ts_p).mean()
        if objective == "supervised":
            loss = trusted_loss
        elif objective == "positive":
            loss = trusted_loss + float(power) * pool_loss
        elif objective == "signed":
            assert ts_d is not None
            paired_loss = pinball(pred_t, ts_d).mean()
            loss = trusted_loss + float(power) * (pool_loss - paired_loss)
        elif objective == "pool":
            assert s_p is not None
            n_t = float(len(s_t))
            n_p = float(len(s_p))
            loss = (n_t * trusted_loss + n_p * pool_loss) / (n_t + n_p)
        else:
            loss = pool_loss if epoch < pre_epochs else trusted_loss
        value = float(loss.detach().cpu())
        if epoch % 5 == 0 or epoch == total_epochs - 1:
            curve.append(
                {
                    "epoch": float(epoch),
                    "loss": value,
                    "trusted": float(trusted_loss.detach().cpu()),
                    "pool": float(pool_loss.detach().cpu()),
                    "paired": float(paired_loss.detach().cpu()),
                }
            )
        if not np.isfinite(value):
            divergence_epoch = epoch
            break
        loss.backward()
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            divergence_epoch = epoch
            break

    predict_2d = _make_predictor(torch, model, transform=transform)

    def predict(array: np.ndarray) -> np.ndarray:
        return predict_2d(array).reshape(-1)

    train_prediction = predict(x_t)
    pool_prediction = predict(x_p) if x_p is not None else np.array([], dtype=float)
    saturation = (
        float(np.mean((train_prediction < 0.01 * bound) | (train_prediction > 0.99 * bound)))
        if bound is not None
        else 0.0
    )
    diagnostics = {
        "objective": objective,
        "power": float(power),
        "seed": int(seed),
        "hidden": int(hidden),
        "epochs_requested": total_epochs,
        "epochs_completed": int(epoch + 1),
        "pretrain_epochs": int(pre_epochs) if objective == "pretrain_finetune" else None,
        "learning_rate": float(learning_rate),
        "output_bound": bound,
        "loss_curve_every_5_epochs": curve,
        "divergence_epoch": divergence_epoch,
        "finite": bool(
            divergence_epoch is None
            and np.all(np.isfinite(train_prediction))
            and np.all(np.isfinite(pool_prediction))
            and np.isfinite(_parameter_norm(torch, model))
        ),
        "parameter_norm": _parameter_norm(torch, model),
        "train_prediction_min": float(np.min(train_prediction)),
        "train_prediction_max": float(np.max(train_prediction)),
        "pool_prediction_min": float(np.min(pool_prediction)) if pool_prediction.size else None,
        "pool_prediction_max": float(np.max(pool_prediction)) if pool_prediction.size else None,
        "bound_saturation": saturation,
    }
    return V11NeuralFit(predict=predict, diagnostics=diagnostics)


def fit_cqr_network(
    x_trusted: np.ndarray,
    y_trusted: np.ndarray,
    *,
    alpha: float,
    seed: int,
    objective: str = "supervised",
    power: float = 1.0,
    x_pool: np.ndarray | None = None,
    y_pool: np.ndarray | None = None,
    y_paired: np.ndarray | None = None,
    hidden: int = 128,
    epochs: int = 200,
    learning_rate: float = 2e-3,
) -> V11NeuralFit:
    if objective not in {"supervised", "signed"}:
        raise ValueError(f"Unknown CQR objective: {objective}")
    if objective == "signed" and (x_pool is None or y_pool is None or y_paired is None):
        raise ValueError("Signed CQR requires paired and pool synthetic outcomes.")

    torch, nn = _prepare_torch(seed)
    x_t = np.asarray(x_trusted, dtype=np.float32)
    y_t = np.asarray(y_trusted, dtype=np.float32)
    if y_t.ndim == 1:
        y_t = y_t[:, None]
    d = int(y_t.shape[1])
    x_p = None if x_pool is None else np.asarray(x_pool, dtype=np.float32)
    y_p = None if y_pool is None else np.asarray(y_pool, dtype=np.float32)
    y_d = None if y_paired is None else np.asarray(y_paired, dtype=np.float32)
    if y_p is not None and y_p.ndim == 1:
        y_p = y_p[:, None]
    if y_d is not None and y_d.ndim == 1:
        y_d = y_d[:, None]

    model = _network(nn, x_t.shape[1], 2 * d, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    tx_t = torch.from_numpy(x_t)
    ty_t = torch.from_numpy(y_t)
    tx_p = None if x_p is None else torch.from_numpy(x_p)
    ty_p = None if y_p is None else torch.from_numpy(y_p)
    ty_d = None if y_d is None else torch.from_numpy(y_d)
    levels = (float(alpha) / 2.0, 1.0 - float(alpha) / 2.0)

    def pinball(prediction: Any, target: Any, tau: float) -> Any:
        residual = target - prediction
        return torch.maximum(float(tau) * residual, (float(tau) - 1.0) * residual)

    def loss_for(x_tensor: Any, y_tensor: Any) -> Any:
        prediction = model(x_tensor)
        lower, upper = prediction[:, :d], prediction[:, d:]
        return (
            pinball(lower, y_tensor, levels[0])
            + pinball(upper, y_tensor, levels[1])
        ).mean()

    curve: list[dict[str, float]] = []
    divergence_epoch: int | None = None
    model.train()
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        trusted_loss = loss_for(tx_t, ty_t)
        pool_loss = torch.zeros((), dtype=trusted_loss.dtype)
        paired_loss = torch.zeros((), dtype=trusted_loss.dtype)
        loss = trusted_loss
        if objective == "signed":
            assert tx_p is not None and ty_p is not None and ty_d is not None
            pool_loss = loss_for(tx_p, ty_p)
            paired_loss = loss_for(tx_t, ty_d)
            loss = trusted_loss + float(power) * (pool_loss - paired_loss)
        value = float(loss.detach().cpu())
        if epoch % 5 == 0 or epoch == int(epochs) - 1:
            curve.append(
                {
                    "epoch": float(epoch),
                    "loss": value,
                    "trusted": float(trusted_loss.detach().cpu()),
                    "pool": float(pool_loss.detach().cpu()),
                    "paired": float(paired_loss.detach().cpu()),
                }
            )
        if not np.isfinite(value):
            divergence_epoch = epoch
            break
        loss.backward()
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            divergence_epoch = epoch
            break

    raw_predict = _make_predictor(torch, model)

    def predict(array: np.ndarray) -> np.ndarray:
        raw = raw_predict(array)
        lower, upper = raw[:, :d], raw[:, d:]
        return np.concatenate([np.minimum(lower, upper), np.maximum(lower, upper)], axis=1)

    train_raw = raw_predict(x_t)
    crossing = float(np.mean(train_raw[:, :d] > train_raw[:, d:]))
    train_prediction = predict(x_t)
    diagnostics = {
        "objective": objective,
        "power": float(power),
        "seed": int(seed),
        "hidden": int(hidden),
        "epochs_requested": int(epochs),
        "epochs_completed": int(epoch + 1),
        "learning_rate": float(learning_rate),
        "output_dimension": d,
        "loss_curve_every_5_epochs": curve,
        "divergence_epoch": divergence_epoch,
        "finite": bool(
            divergence_epoch is None
            and np.all(np.isfinite(train_prediction))
            and np.isfinite(_parameter_norm(torch, model))
        ),
        "parameter_norm": _parameter_norm(torch, model),
        "training_crossing_rate_before_sort": crossing,
        "training_lower_min": float(np.min(train_prediction[:, :d])),
        "training_upper_max": float(np.max(train_prediction[:, d:])),
    }
    return V11NeuralFit(predict=predict, diagnostics=diagnostics)
