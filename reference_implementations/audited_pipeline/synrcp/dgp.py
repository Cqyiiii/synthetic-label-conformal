from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

Mechanism = Literal[
    "well_independent",
    "well_coupled",
    "misspecified_independent",
    "misspecified_coupled",
    "certified_exact_independent",
    "paired_oracle",
    "signed_positive",
    "signed_zero",
    "signed_negative",
]


@dataclass(frozen=True)
class ScoreDGP:
    """Clean crossed DGP used by the Theory V5 mechanism experiments.

    The target score is

        S = q*(X) + sigma(X) (Z - z_tau),

    and the synthetic score is

        S^dagger = q*(X) + Delta(X)
                   + sigma(X) {rho Z + sqrt(1-rho^2) Z^dagger - z_tau}.

    Changing ``rho`` changes only paired residual coupling.  The conditional
    synthetic marginal is otherwise unchanged.  Changing the learner feature
    map creates the specification factor, so the 2 x 2 design is genuinely
    crossed.
    """

    tau: float = 0.9
    heterogeneity: float = 0.35
    coupling: float = 0.75
    quadratic: float = 0.9
    synthetic_bias: float = 0.5

    @property
    def z_tau(self) -> float:
        return float(norm.ppf(self.tau))

    def q_star(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return 3.0 + 0.6 * x + self.quadratic * x**2

    def sigma(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        raw = 0.45 * np.exp(self.heterogeneity * x)
        return np.clip(raw, 0.12, 1.5)

    def delta(self, x: np.ndarray, mechanism: str) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if mechanism in {"certified_exact_independent", "paired_oracle", "signed_positive", "signed_zero", "signed_negative"}:
            return np.zeros_like(x)
        return self.synthetic_bias * (x**2 - 1.0 / 3.0)

    def rho(self, mechanism: str) -> float:
        if mechanism in {"well_coupled", "misspecified_coupled", "signed_positive"}:
            return abs(self.coupling)
        if mechanism == "signed_negative":
            return -abs(self.coupling)
        if mechanism in {
            "well_independent",
            "misspecified_independent",
            "certified_exact_independent",
            "signed_zero",
        }:
            return 0.0
        if mechanism == "paired_oracle":
            return 1.0
        raise ValueError(f"Unknown mechanism: {mechanism}")

    def features(self, x: np.ndarray, well_specified: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if well_specified:
            return np.column_stack([np.ones_like(x), x, x**2])
        return np.column_stack([np.ones_like(x), x])

    def sample_x(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=int(n))

    def sample_score(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        z_shared: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        z = rng.normal(size=x.size) if z_shared is None else np.asarray(z_shared, dtype=float).reshape(-1)
        score = self.q_star(x) + self.sigma(x) * (z - self.z_tau)
        return score, z

    def synthetic_score(
        self,
        x: np.ndarray,
        true_z: np.ndarray,
        mechanism: Mechanism | str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        true_z = np.asarray(true_z, dtype=float).reshape(-1)
        if mechanism == "paired_oracle":
            return self.q_star(x) + self.sigma(x) * (true_z - self.z_tau)
        rho = self.rho(str(mechanism))
        z2 = rng.normal(size=x.size)
        z_syn = rho * true_z + np.sqrt(max(1.0 - rho**2, 0.0)) * z2
        return self.q_star(x) + self.delta(x, str(mechanism)) + self.sigma(x) * (z_syn - self.z_tau)

    def cdf(self, threshold: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        threshold = np.asarray(threshold, dtype=float).reshape(-1)
        z = (threshold - self.q_star(x)) / self.sigma(x) + self.z_tau
        return norm.cdf(z)

    def synthetic_cdf(self, threshold: np.ndarray, x: np.ndarray, mechanism: str) -> np.ndarray:
        """Conditional marginal CDF of the synthetic score.

        It is independent of the paired coupling coefficient.  This property is
        the key invariant of the clean crossed factorial.
        """

        x = np.asarray(x, dtype=float).reshape(-1)
        threshold = np.asarray(threshold, dtype=float).reshape(-1)
        z = (threshold - self.q_star(x) - self.delta(x, mechanism)) / self.sigma(x) + self.z_tau
        return norm.cdf(z)

    def density(self, threshold: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        threshold = np.asarray(threshold, dtype=float).reshape(-1)
        z = (threshold - self.q_star(x)) / self.sigma(x) + self.z_tau
        return norm.pdf(z) / self.sigma(x)


def mechanism_is_well_specified(mechanism: str) -> bool:
    return mechanism in {
        "well_independent",
        "well_coupled",
        "certified_exact_independent",
        "paired_oracle",
        "signed_positive",
        "signed_zero",
        "signed_negative",
    }


def mechanism_is_certified(mechanism: str) -> bool:
    return mechanism == "certified_exact_independent"
