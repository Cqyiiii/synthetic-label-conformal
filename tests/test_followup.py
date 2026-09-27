from __future__ import annotations

import numpy as np

from synrcp.followup import FollowupConfig, build_scaling_plan, run_derivative_check


def test_scaling_plan_has_270_unique_tasks() -> None:
    plan = build_scaling_plan()
    keys = {(row["sweep"], row["n"], row["N"], row["m"], row["seed"]) for row in plan}
    assert len(plan) == 270
    assert len(keys) == 270
    assert sum(row["sweep"] == "n_scaling" for row in plan) == 120
    assert sum(row["sweep"] == "N_scaling" for row in plan) == 150


def test_derivative_check_has_exact_constant_null_and_quadratic_remainder() -> None:
    rows, summary, acceptance = run_derivative_check(FollowupConfig(quadrature_nodes=512))
    assert len(rows) == 24
    assert acceptance["constant_shift_threshold_max_below_1e_10"]
    assert acceptance["constant_shift_coverage_max_below_1e_10"]
    slopes = summary.loc[
        summary["perturbation"] != "constant", "small_epsilon_loglog_slope"
    ].to_numpy(float)
    assert np.all((slopes > 1.8) & (slopes < 2.2))

