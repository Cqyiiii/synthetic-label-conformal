from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def finite_array(value, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains nonfinite values.")
    return array


def pinball(residual: np.ndarray, tau: float) -> np.ndarray:
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1).")
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def conformal_offset(residuals: np.ndarray, tau: float) -> float:
    """Exact conformal order statistic; no interpolation."""
    residuals = finite_array(residuals, 1, "residuals")
    if len(residuals) == 0:
        raise ValueError("Calibration sample is empty.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1).")

    # Interpret the supplied decimal coverage level exactly.
    fraction = Fraction(str(float(tau)))
    numerator = fraction.numerator * (len(residuals) + 1)
    denominator = fraction.denominator
    rank = (numerator + denominator - 1) // denominator

    if rank > len(residuals):
        return float("inf")
    return float(np.partition(residuals, rank - 1)[rank - 1])


def bounded_fourier_features(
    x: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """
    frequencies must be fixed before development/evaluation outcomes are used.
    Use training-only preprocessing and a predeclared bandwidth.

    Includes a constant feature. Every returned row has norm one.
    These are evaluation features, NOT a linear quantile learner.
    """
    x = finite_array(x, 2, "x")
    frequencies = finite_array(frequencies, 2, "frequencies")
    if x.shape[1] != frequencies.shape[0]:
        raise ValueError("Feature/frequency dimensions do not match.")
    r = frequencies.shape[1]
    if r < 1:
        raise ValueError("At least one frequency is required.")

    phase = x @ frequencies
    return np.concatenate(
        [
            np.ones((len(x), 1)),
            np.cos(phase) / np.sqrt(r),
            np.sin(phase) / np.sqrt(r),
        ],
        axis=1,
    ) / np.sqrt(2.0)


def mean_and_noise(vector_rows: np.ndarray):
    """
    Return the sample mean and tr(unbiased sample covariance) / sample size.
    """
    values = finite_array(vector_rows, 2, "vector_rows")
    n = len(values)
    if n < 2:
        raise ValueError("The quadratic estimator requires at least two rows.")

    mean = values.mean(axis=0)
    centered = values - mean
    noise = float(np.sum(centered * centered) / (n * (n - 1)))
    return mean, noise


def quadratic_moment_estimate(
    z_true: np.ndarray,
    phi: np.ndarray,
    *,
    beta: float = 0.0,
    z_synthetic: np.ndarray | None = None,
    pool_phi: np.ndarray | None = None,
    pool_z_synthetic: np.ndarray | None = None,
) -> float:
    """
    Unbiased squared-moment estimator for a FIXED candidate/threshold.

    beta=0 does not read synthetic values.
    Do not clip a negative estimate to zero.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("This implementation requires beta in [0, 1].")

    z_true = finite_array(z_true, 1, "z_true")
    phi = finite_array(phi, 2, "phi")
    if len(z_true) != len(phi):
        raise ValueError("z_true and phi have different row counts.")

    if beta == 0.0:
        mean_a, correction_a = mean_and_noise(phi * z_true[:, None])
        return float(mean_a @ mean_a - correction_a)

    if any(
        item is None
        for item in (z_synthetic, pool_phi, pool_z_synthetic)
    ):
        raise ValueError("Synthetic validation inputs are required for beta > 0.")

    z_synthetic = finite_array(z_synthetic, 1, "z_synthetic")
    pool_phi = finite_array(pool_phi, 2, "pool_phi")
    pool_z_synthetic = finite_array(
        pool_z_synthetic, 1, "pool_z_synthetic"
    )
    if len(z_synthetic) != len(z_true):
        raise ValueError("Paired true/synthetic row counts do not match.")
    if len(pool_phi) != len(pool_z_synthetic):
        raise ValueError("Synthetic-pool row counts do not match.")
    if phi.shape[1] != pool_phi.shape[1]:
        raise ValueError("Paired/pool evaluation feature dimensions differ.")

    a = phi * (z_true - beta * z_synthetic)[:, None]
    b = pool_phi * pool_z_synthetic[:, None]
    mean_a, correction_a = mean_and_noise(a)
    mean_b, correction_b = mean_and_noise(b)

    mean = mean_a + beta * mean_b
    return float(
        mean @ mean
        - correction_a
        - beta * beta * correction_b
    )


def selection_scores(
    q_dev: np.ndarray,
    s_dev: np.ndarray,
    phi_dev: np.ndarray,
    *,
    tau: float = 0.9,
    seed: int = 0,
    sdag_dev: np.ndarray | None = None,
    q_pool: np.ndarray | None = None,
    sdag_pool: np.ndarray | None = None,
    phi_pool: np.ndarray | None = None,
    fold_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Two-fold scores for fixed candidates; contains no final-outcome input."""
    q = finite_array(q_dev, 2, "q_dev")
    s = finite_array(s_dev, 1, "s_dev")
    phi = finite_array(phi_dev, 2, "phi_dev")
    if q.shape[1] != len(s) or len(phi) != len(s) or q.shape[0] < 1:
        raise ValueError("Candidate/development shapes do not match.")
    if len(s) < 4 or phi.shape[1] == 0:
        raise ValueError("Need at least four development rows and one feature.")
    pinball(np.zeros(1), tau)
    if fold_indices is None:
        perm = np.random.default_rng(seed).permutation(len(s))
        folds = (perm[:len(s)//2], perm[len(s)//2:])
    else:
        folds = tuple(np.asarray(f, dtype=int) for f in fold_indices)
        if len(folds) != 2 or any(len(f)<2 for f in folds):
            raise ValueError("Two nonempty folds with at least two rows required.")
        if not np.array_equal(np.sort(np.concatenate(folds)), np.arange(len(s))):
            raise ValueError("Folds must partition all development rows.")
    synthetic = any(v is not None for v in (sdag_dev,q_pool,sdag_pool,phi_pool))
    if synthetic:
        if any(v is None for v in (sdag_dev,q_pool,sdag_pool,phi_pool)):
            raise ValueError("All four synthetic validation inputs required.")
        sd = finite_array(sdag_dev,1,"sdag_dev")
        qp = finite_array(q_pool,2,"q_pool")
        sp = finite_array(sdag_pool,1,"sdag_pool")
        pp = finite_array(phi_pool,2,"phi_pool")
        if len(sd)!=len(s) or qp.shape!=(len(q),len(sp)) or len(pp)!=len(sp) or pp.shape[1]!=phi.shape[1]:
            raise ValueError("Synthetic validation dimensions do not match.")
    keys = ["raw_pinball","calibrated_pinball","coverage_beta0"]
    if synthetic:
        keys.append("coverage_beta1")
    scores = {k:np.zeros(len(q)) for k in keys}
    offsets = np.empty((len(q),2))
    fold_scores = []
    for fold in range(2):
        calibration, validation = folds[fold], folds[1-fold]
        per_fold = {k:[] for k in keys}
        for j in range(len(q)):
            gamma = conformal_offset(s[calibration]-q[j,calibration],tau)
            if not np.isfinite(gamma):
                raise ValueError("Temporary calibration too small for finite offset.")
            offsets[j,fold] = gamma
            z = (s[validation] <= q[j,validation]+gamma).astype(float)-tau
            vals = {
                "raw_pinball":float(pinball(s[validation]-q[j,validation],tau).mean()),
                "calibrated_pinball":float(pinball(s[validation]-q[j,validation]-gamma,tau).mean()),
                "coverage_beta0":quadratic_moment_estimate(z,phi[validation]),
            }
            if synthetic:
                vals["coverage_beta1"] = quadratic_moment_estimate(
                    z,phi[validation],beta=1.0,
                    z_synthetic=(sd[validation]<=q[j,validation]+gamma).astype(float)-tau,
                    pool_phi=pp,
                    pool_z_synthetic=(sp<=qp[j]+gamma).astype(float)-tau,
                )
            for k,v in vals.items():
                scores[k][j] += v/2.0
                per_fold[k].append(v)
        fold_scores.append(per_fold)
    return {
        "scores":{k:v.tolist() for k,v in scores.items()},
        "selected":{k:int(np.argmin(v)) for k,v in scores.items()},
        "fold0":folds[0].tolist(),"fold1":folds[1].tolist(),
        "offsets":offsets.tolist(),"fold_scores":fold_scores,
    }

