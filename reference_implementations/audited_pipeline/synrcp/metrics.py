from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler




@dataclass(frozen=True)
class NumpyKMeansPartition:
    """Small deterministic NumPy k-means model used for coverage partitions.

    The execution environment used for the audited artifact can deadlock inside
    scikit-learn/OpenMP KMeans on some BLAS builds.  The partition role is small,
    so a bounded NumPy Lloyd implementation is both faster and easier to audit.
    """

    cluster_centers_: np.ndarray

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_centers_.shape[0])

    def predict(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=float)
        distances = np.sum(
            (values[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2
        )
        return np.argmin(distances, axis=1).astype(int)


def _fit_numpy_kmeans(
    z: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 8,
    max_iter: int = 80,
) -> NumpyKMeansPartition:
    values = np.asarray(z, dtype=float)
    n, p = values.shape
    k_eff = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    for _ in range(max(1, int(n_init))):
        centers = np.empty((k_eff, p), dtype=float)
        first = int(rng.integers(0, n))
        centers[0] = values[first]
        closest = np.sum((values - centers[0]) ** 2, axis=1)
        for center_id in range(1, k_eff):
            total = float(np.sum(closest))
            if total <= 1e-15:
                candidate = int(rng.integers(0, n))
            else:
                candidate = int(rng.choice(n, p=closest / total))
            centers[center_id] = values[candidate]
            closest = np.minimum(closest, np.sum((values - centers[center_id]) ** 2, axis=1))
        previous_labels: np.ndarray | None = None
        for _iteration in range(max(1, int(max_iter))):
            distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            previous_labels = labels.copy()
            nearest_distance = distances[np.arange(n), labels]
            for center_id in range(k_eff):
                members = values[labels == center_id]
                if members.size:
                    centers[center_id] = np.mean(members, axis=0)
                else:
                    farthest = int(np.argmax(nearest_distance))
                    centers[center_id] = values[farthest]
                    nearest_distance[farthest] = -np.inf
        final_distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(final_distances, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
    assert best_centers is not None
    return NumpyKMeansPartition(best_centers)


@dataclass(frozen=True)
class GroupMetric:
    k: int
    raw: float
    debiased: float
    cross_product: float
    heterogeneity: float
    marginal_offset: float
    coverage: float
    min_group_size: int


def population_coverage_metrics(pi: np.ndarray, tau: float) -> dict[str, float]:
    pi = np.asarray(pi, dtype=float).reshape(-1)
    marginal = float(np.mean(pi))
    return {
        "msce": float(np.mean((pi - tau) ** 2)),
        "het": float(np.var(pi)),
        "marginal_offset": float((marginal - tau) ** 2),
        "marginal_coverage": marginal,
    }


def fit_partition(
    x_partition: np.ndarray, k: int, seed: int
) -> tuple[StandardScaler, NumpyKMeansPartition]:
    scaler = StandardScaler().fit(np.asarray(x_partition, dtype=float))
    z = scaler.transform(np.asarray(x_partition, dtype=float))
    model = _fit_numpy_kmeans(z, k=k, seed=seed)
    return scaler, model


def grouped_coverage_metrics(
    x_partition: np.ndarray,
    x_eval: np.ndarray,
    covered_eval: np.ndarray,
    tau: float,
    k: int,
    seed: int,
) -> GroupMetric:
    scaler, model = fit_partition(x_partition, k=k, seed=seed)
    groups = model.predict(scaler.transform(np.asarray(x_eval, dtype=float)))
    covered_arr = np.asarray(covered_eval, dtype=float).reshape(-1)
    n = covered_arr.size
    raw = 0.0
    deb = 0.0
    cross = 0.0
    weighted_var = 0.0
    min_size = n
    group_ps: list[tuple[float, float]] = []
    rng = np.random.default_rng(seed + 1701)
    for group_id in np.unique(groups):
        idx = np.flatnonzero(groups == group_id)
        if idx.size == 0:
            continue
        min_size = min(min_size, int(idx.size))
        weight = idx.size / n
        p = float(np.mean(covered_arr[idx]))
        group_ps.append((weight, p))
        raw += weight * (p - tau) ** 2
        correction = p * (1.0 - p) / (idx.size - 1) if idx.size > 1 else 0.0
        deb += weight * ((p - tau) ** 2 - correction)
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        split = max(1, idx.size // 2)
        left = shuffled[:split]
        right = shuffled[split:]
        if right.size == 0:
            cross += weight * (p - tau) ** 2
        else:
            p1 = float(np.mean(covered_arr[left]))
            p2 = float(np.mean(covered_arr[right]))
            cross += weight * (p1 - tau) * (p2 - tau)
    marginal = float(np.mean(covered_arr))
    for weight, p in group_ps:
        weighted_var += weight * (p - marginal) ** 2
    return GroupMetric(
        k=int(model.n_clusters),
        raw=float(raw),
        debiased=float(deb),
        cross_product=float(cross),
        heterogeneity=float(weighted_var),
        marginal_offset=float((marginal - tau) ** 2),
        coverage=marginal,
        min_group_size=int(min_size),
    )


def select_worst_slab(
    x_select: np.ndarray,
    covered_select: np.ndarray,
    min_mass: float = 0.1,
    directions: int = 256,
    quantile_grid: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    x = np.asarray(x_select, dtype=float)
    y = np.asarray(covered_select, dtype=float).reshape(-1)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    rng = np.random.default_rng(seed)
    best: dict[str, Any] = {
        "coverage": 1.0,
        "direction": None,
        "lower": None,
        "upper": None,
        "mass": 1.0,
    }
    quantiles = np.linspace(0.0, 1.0 - min_mass, quantile_grid)
    widths = np.linspace(min_mass, min(0.75, 1.0), quantile_grid)
    for _ in range(int(directions)):
        direction = rng.normal(size=z.shape[1])
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction /= norm
        projection = z @ direction
        pairs: list[tuple[float, float]] = []
        q_needed: set[float] = set()
        for start in quantiles:
            for width in widths:
                end = min(start + width, 1.0)
                if end - start < min_mass:
                    continue
                pairs.append((float(start), float(end)))
                q_needed.add(float(start))
                q_needed.add(float(end))
        q_list = sorted(q_needed)
        q_values = np.quantile(projection, q_list)
        q_map = dict(zip(q_list, q_values, strict=True))
        for start, end in pairs:
            lower, upper = q_map[start], q_map[end]
            mask = (projection >= lower) & (projection <= upper)
            if np.mean(mask) + 1e-12 < min_mass or not mask.any():
                continue
            coverage = float(np.mean(y[mask]))
            if coverage < best["coverage"]:
                best = {
                    "coverage": coverage,
                    "direction": direction.copy(),
                    "lower": float(lower),
                    "upper": float(upper),
                    "mass": float(np.mean(mask)),
                }
    best["scaler_mean"] = scaler.mean_.tolist()
    best["scaler_scale"] = scaler.scale_.tolist()
    return best


def evaluate_slab(
    x_eval: np.ndarray, covered_eval: np.ndarray, slab: dict[str, Any]
) -> dict[str, float]:
    if slab.get("direction") is None:
        return {"wsc": float(np.mean(covered_eval)), "wsc_mass": 1.0}
    x = np.asarray(x_eval, dtype=float)
    mean = np.asarray(slab["scaler_mean"], dtype=float)
    scale = np.asarray(slab["scaler_scale"], dtype=float)
    z = (x - mean) / np.where(scale == 0, 1.0, scale)
    direction = np.asarray(slab["direction"], dtype=float)
    projection = z @ direction
    mask = (projection >= float(slab["lower"])) & (projection <= float(slab["upper"]))
    if not mask.any():
        return {"wsc": float("nan"), "wsc_mass": 0.0}
    return {
        "wsc": float(np.mean(np.asarray(covered_eval).reshape(-1)[mask])),
        "wsc_mass": float(np.mean(mask)),
    }


def cross_fitted_ert(
    x: np.ndarray,
    covered_arr: np.ndarray,
    tau: float,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Optional proper-risk diagnostic; not the primary Theory V6 endpoint.

    The classifier is unweighted, so its probabilities retain their coverage
    interpretation.  ``ert_brier_gain`` compares the constant predictor tau
    with the cross-fitted conditional predictor; positive values mean the
    covariates explain coverage heterogeneity.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(covered_arr, dtype=int).reshape(-1)
    if np.unique(y).size < 2 or min(np.bincount(y)) < 2:
        p = np.full(y.size, np.mean(y), dtype=float)
    else:
        n_splits = min(int(folds), int(min(np.bincount(y))))
        cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
        p = np.empty(y.size, dtype=float)
        for train, valid in cv.split(x, y):
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(scaler.transform(x[train]), y[train])
            p[valid] = model.predict_proba(scaler.transform(x[valid]))[:, 1]
    conditional_risk = float(np.mean((p - y) ** 2))
    constant_risk = float(np.mean((float(tau) - y) ** 2))
    return {
        "ert_brier_gain": constant_risk - conditional_risk,
        "ert_conditional_brier": conditional_risk,
        "ert_constant_brier": constant_risk,
    }


def paired_bootstrap_difference(
    baseline: np.ndarray,
    method: np.ndarray,
    seed: int = 0,
    draws: int = 2000,
    lower_is_better: bool = True,
) -> dict[str, float]:
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)
    diff = method - baseline
    rng = np.random.default_rng(seed)
    if diff.size == 0:
        return {
            "mean": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "win_rate": float("nan"),
        }
    idx = rng.integers(0, diff.size, size=(int(draws), diff.size))
    boot = np.mean(diff[idx], axis=1)
    wins = diff < 0 if lower_is_better else diff > 0
    return {
        "mean": float(np.mean(diff)),
        "low": float(np.quantile(boot, 0.025)),
        "high": float(np.quantile(boot, 0.975)),
        "win_rate": float(np.mean(wins)),
    }
