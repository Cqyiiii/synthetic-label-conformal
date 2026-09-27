from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from .io import atomic_write_json
from .real_data import run_real_data_task
from .simulation import run_conformal_null_task, run_mechanism_task, run_objective_stability
from .system import apply_thread_limit, system_metadata
from .v11 import run_v11_task


def _resolve_diagnostic_params(task: dict[str, Any]) -> dict[str, Any]:
    params = dict(task["params"])
    experiment = task["experiment"]
    if experiment == "real_data_nuisance_randomness":
        mode = str(params.get("nuisance_mode", "fixed_fit_seeds"))
        seed = int(params.get("seed", 0))
        # Both modes use identical role and dataset seeds for paired comparison.
        params.setdefault("role_seed", seed + 1)
        params.setdefault("dataset_sample_seed", 710)
        if mode == "fixed_fit_seeds":
            params["teacher_fit_seed"] = 4242
            params["mean_fit_seed"] = 4343
            params["representation_fit_seed"] = 4444
        elif mode == "retrained_fit_seeds":
            params["teacher_fit_seed"] = seed + 4242
            params["mean_fit_seed"] = seed + 4343
            params["representation_fit_seed"] = seed + 4444
        else:
            raise ValueError(f"Unknown nuisance mode: {mode}")
    task_copy = dict(task)
    task_copy["params"] = params
    return task_copy


def execute_task(task: dict[str, Any], project_root: Path, force: bool = False) -> Path:
    task = _resolve_diagnostic_params(task)
    threads = int(task.get("threads", 1))
    apply_thread_limit(threads)
    output_root = Path(task["output_root"])
    if not output_root.is_absolute():
        output_root = project_root / output_root
    task_path = output_root / "tasks" / f"{task['task_id']}.json"
    if task_path.exists() and not force:
        return task_path
    start = time.perf_counter()
    experiment = task["experiment"]
    # Environment variables set after NumPy import are not sufficient for every
    # BLAS backend.  The runtime guard prevents pathological oversubscription,
    # especially for the 49-output WEC benchmark.
    with threadpool_limits(limits=threads):
        if experiment in {
            "mechanism_2x2",
            "paired_score_oracle",
            "calibration_density",
            "signed_power",
            "smoke",
        }:
            result, curves = run_mechanism_task(task, project_root)
        elif experiment == "conformal_null":
            result, curves = run_conformal_null_task(task, project_root)
        elif experiment == "objective_stability":
            result, curves = run_objective_stability(task, project_root)
        elif (
            experiment in {
                "real_data_main",
                "real_data_teacher_strength",
                "real_data_nuisance_randomness",
                "smoke_real",
            }
            or experiment.startswith("synthetic_")
            or experiment.startswith("v9_")
            or experiment.startswith("v10_")
        ):
            result, curves = run_real_data_task(task, project_root)
        elif experiment.startswith("v11_"):
            result, curves = run_v11_task(task, project_root)
        else:
            raise ValueError(f"Unknown experiment type: {experiment}")
    curves_path = None
    if curves:
        curves_dir = output_root / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        curves_path_obj = curves_dir / f"{task['task_id']}.npz"
        np.savez_compressed(curves_path_obj, **curves)
        try:
            curves_path = str(curves_path_obj.relative_to(project_root))
        except ValueError:
            curves_path = str(curves_path_obj)
    payload = {
        "task": task,
        "result": result,
        "runtime_seconds": float(time.perf_counter() - start),
        "system": system_metadata(threads),
        "curves_path": curves_path,
    }
    atomic_write_json(task_path, payload)
    return task_path


def load_plan(path: str | Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks
